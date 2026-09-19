"""Freeze attribution-blind cohorts from a saved UniProt snapshot.

No explanation files are accepted. Reruns validate an existing immutable bundle.
"""

import argparse
from pathlib import Path

import json
import subprocess

from mapexploc.baseline import search_similar, similarity_groups
from mapexploc.execution import atomic_json, fingerprint
from mapexploc.provenance import file_sha256, sequence_sha256
from mapexploc.study import (
    FAMILIES,
    audit_manifests,
    eligible_features,
    planned_power,
    required_groups,
    split_group,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--historical", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--retrieved-at", required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "freeze.json").exists():
        for name, expected in json.loads((args.output / "freeze.json").read_text())[
            "checksums"
        ].items():
            if file_sha256(args.output / name) != expected:
                raise ValueError("Frozen cohort was modified")
        print("Existing frozen cohort checksums verified; no rewrite")
        return
    source = json.loads(args.source.read_text())["results"]
    historical = json.loads(args.historical.read_text())["results"]
    historical_ids = {r["protein"]["protein_id"] for r in historical}
    eligible = {}
    for record in source:
        sequence = record["sequence"]["value"]
        if not 10 <= len(sequence) <= 1022 or set(sequence) - set(
            "ACDEFGHIKLMNPQRSTVWY"
        ):
            continue
        families = eligible_features(record)
        if families:
            eligible[record["primaryAccession"]] = {
                "record": record,
                "families": families,
            }
    sequences = {k: v["record"]["sequence"]["value"] for k, v in eligible.items()}
    sequences.update(
        {r["protein"]["protein_id"]: r["protein"]["sequence"] for r in historical}
    )
    ids = sorted(sequences)
    fasta = args.output / "grouping-input.fasta"
    fasta.write_text("".join(f">{pid}\n{sequences[pid]}\n" for pid in ids))
    pair_file = args.output / "pairs.tsv"
    pairs = search_similar(fasta, fasta, pair_file, 4)
    canonical = "\n".join(sorted(set(pair_file.read_text().splitlines()))) + "\n"
    pair_file.write_text(canonical)
    groups = dict(zip(ids, similarity_groups(ids, pairs)))
    historical_groups = {groups[pid] for pid in historical_ids}
    rows = {name: [] for name in ("development", "reference_pool", "evaluation")}
    represented = set()
    excluded = []
    for pid in sorted(eligible):
        group = groups[pid]
        if group in historical_groups:
            excluded.append({"protein_id": pid, "reason": "historical_sequence_group"})
            continue
        # One representative per component across families, selected by accession.
        if group in represented:
            excluded.append(
                {"protein_id": pid, "reason": "nonrepresentative_homologue"}
            )
            continue
        represented.add(group)
        split = split_group(group)
        item = eligible[pid]
        rows[split].append(
            dict(
                protein_id=pid,
                sequence=sequences[pid],
                sequence_sha256=sequence_sha256(sequences[pid]),
                group=group,
                split=split,
                families=item["families"],
            )
        )
    manifests = [
        dict(
            cohort_id="v2x-" + split,
            selection_criteria=(
                "Attribution-blind joint 30% identity / 80% bidirectional coverage"
                " components; deterministic hash split; first accession per component"
            ),
            members=members,
        )
        for split, members in rows.items()
    ]
    audit_manifests(manifests, {pid: groups[pid] for pid in historical_ids})
    required = required_groups()
    families = {}
    for family in FAMILIES:
        counts = {
            split: sum(family in m["families"] for m in members)
            for split, members in rows.items()
        }
        families[family] = {
            "output_class": FAMILIES[family],
            "counts": counts,
            "required_independent_evaluation_groups": required,
            "planned_standardized_effect": 0.5,
            "planned_power": planned_power(counts["evaluation"]),
            "status": (
                "candidate_pending_controls_and_resolution"
                if counts["evaluation"] >= required
                else "exploratory_insufficient_independent_groups"
            ),
            "controls_and_resolution": (
                "Not established by sample size; must pass preregistered eligibility"
                " before final inference"
            ),
        }
    protocol = dict(
        protocol_id="v2x-biological-development-v1",
        seed=20260919,
        hypotheses=12,
        alpha=0.05,
        multiplicity="Holm in evaluation; conservative Bonferroni power planning",
        minimum_groups=60,
        required_groups=required,
        standardized_effect=0.5,
        power_target=0.8,
        bootstrap_replicates=1000,
        randomizations=1000,
        historical_excluded=True,
        grouping=(
            "MMseqs2 identity>=0.3 coverage>=0.8 both sequences; connected components"
        ),
        representative="Lexicographically first accession per eligible component",
        primary_statistic=(
            "Within-protein positive output-class SHAP density in complete regions"
            " wholly within experimentally annotated determinants minus matched"
            " length/terminal-stratum non-determinant regions"
        ),
        fixed_terminal_controls=(
            "Matched sequence controls and independent motif/order interventions"
            " required; annotation-pair overlap alone is invalid"
        ),
        degeneracy_rule=(
            "Exclude formal annotation-pair inference when fewer than 20 distinct"
            " permutations of region eligibility masks exist"
        ),
        resolution_rule=(
            "Formal interval analysis requires at least one fully contained region and"
            " one matched non-overlapping region per protein; minimum powered count"
            " applies after exclusions"
        ),
        method_selection=(
            "Development data only; no historical/final-evaluation overlap scores; no"
            " promoted replacement without both-mode stability and held-out"
            " faithfulness improvement"
        ),
        source_query=(
            "reviewed:true AND taxonomy_id:2759 AND length:[10 TO 1022] AND"
            ' (ft_signal:* OR ft_transit:mitochondrion OR ft_motif:"nuclear'
            ' localization" OR ft_transmem:* OR ft_motif:peroxisom* OR'
            " ft_motif:retention)"
        ),
        retrieved_at=args.retrieved_at,
        source_sha256=file_sha256(args.source),
        source_records=len(source),
        eligible_records=len(eligible),
        mmseqs_version=subprocess.check_output(
            ["mmseqs", "version"], text=True
        ).strip(),
        families=families,
    )
    outputs = {
        "protocol.json": protocol,
        "group-membership.json": groups,
        "exclusions.json": excluded,
        "eligible-records.json": [v["record"] for _, v in sorted(eligible.items())],
    }
    outputs.update({m["cohort_id"] + ".json": m for m in manifests})
    for name, value in outputs.items():
        atomic_json(args.output / name, value)
    atomic_json(
        args.output / "freeze.json",
        {
            "scope": (
                "Independent curation inputs; no evaluation attributions generated"
            ),
            "protocol_sha256": fingerprint(protocol),
            "checksums": {
                name: file_sha256(args.output / name)
                for name in list(outputs) + ["pairs.tsv", "grouping-input.fasta"]
            },
        },
    )
    print(json.dumps(families, indent=2))


if __name__ == "__main__":
    main()
