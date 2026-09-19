"""Recreate the frozen positive-control cohort from its exact UniProt snapshot.

No explanations are read. Run before inference. Current API results are not a
substitute for the archived snapshot identified in the release manifest.
"""

import argparse
import hashlib
from pathlib import Path

import json
import pandas as pd
import subprocess

from mapexploc.annotations import import_uniprot
from mapexploc.baseline import search_similar, similarity_groups
from mapexploc.provenance import file_sha256, sequence_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retrieved-at", required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--sorting-signals", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    records = json.loads(args.source.read_text())["results"]
    minimum, maximum = protocol["selection"]["length_range"]
    eligible = []
    for record in sorted(records, key=lambda r: r["primaryAccession"]):
        sequence = record["sequence"]["value"]
        if not minimum <= len(sequence) <= maximum or set(sequence) - set(
            "ACDEFGHIKLMNPQRSTVWY"
        ):
            continue
        for feature in record.get("features", []):
            if feature["type"] != "Signal":
                continue
            start, end = feature["location"]["start"], feature["location"]["end"]
            if (
                start.get("value") == 1
                and 20 <= (end.get("value") or 0) <= 40
                and all(p.get("modifier", "EXACT") == "EXACT" for p in [start, end])
                and any(
                e.get("evidenceCode") == "ECO:0000269"
                for e in feature.get("evidences", [])
            )
            ):
                eligible.append(record)
                break
    args.output.mkdir(parents=True, exist_ok=True)
    fasta = args.output / "eligible.fasta"
    fasta.write_text(
        "".join(
            f">{r['primaryAccession']}\n{r['sequence']['value']}\n" for r in eligible
        )
    )
    pair_file = args.output / "pairs.tsv"
    pairs = search_similar(fasta, fasta, pair_file, 4)
    groups = similarity_groups([r["primaryAccession"] for r in eligible], pairs)
    selected, members, seen = [], [], set()
    for record, group in zip(eligible, groups):
        if group in seen:
            continue
        seen.add(group)
        selected.append(record)
        members.append(
            {
                "protein_id": record["primaryAccession"],
                "sequence_sha256": sequence_sha256(record["sequence"]["value"]),
                "group": group,
                "split": "biological_model_behavior_control",
            }
        )
        if len(selected) == protocol["selection"]["target_count"]:
            break
    if len(selected) < protocol["selection"]["target_count"]:
        raise ValueError(
            "Insufficient independent proteins; do not change criteria "
            "after inspecting explanations"
        )
    snapshot = file_sha256(args.source)
    annotations = []
    for record in selected:
        numeric = {
            **record,
            "features": [
                f
                for f in record.get("features", [])
                if all(
                    isinstance(f["location"][p].get("value"), int)
                    for p in ["start", "end"]
                )
            ],
        }
        annotations.extend(
            import_uniprot(
                numeric,
                protein_id=record["primaryAccession"],
                sequence=record["sequence"]["value"],
                release="snapshot-sha256:" + snapshot,
                retrieved_at=args.retrieved_at,
                source_url="https://rest.uniprot.org/uniprotkb/"
                           + record["primaryAccession"],
            )
        )
    manifest = {
        "cohort_id": "signal-concordance-v2-" + args.retrieved_at[:10],
        "selection_criteria": (
            "Accession-ordered first 30 independent representatives meeting "
            "the frozen protocol; no attribution-based selection"
        ),
        "members": members,
        "source_snapshot_sha256": snapshot,
        "candidate_count": len(records),
        "eligible_count": len(eligible),
        "mmseqs_version": subprocess.check_output(
            ["mmseqs", "version"], text=True
        ).strip(),
        "pair_table_sha256": file_sha256(pair_file),
        "sorted_pair_table_sha256": hashlib.sha256(
            ("\n".join(sorted(pair_file.read_text().splitlines())) + "\n").encode()
        ).hexdigest(),
        "protocol_sha256": file_sha256(args.protocol),
        "grouping": (
            "MMseqs2 30% identity, 80% bidirectional coverage, connected "
            "components; first accession per group"
        ),
        "source_query": (
            "reviewed:true AND taxonomy_id:2759 AND length:[100 TO 250] "
            "AND ft_signal:*"
        ),
        "annotation_unknown_endpoints": (
            "Not drawable; original entries retained in uniprot-records.json; "
            "excluded from overlays and statistics"
        ),
    }
    sources = {}
    for name, path, url in [
        ("localization", args.training, "Swissprot_Train_Validation_dataset.csv"),
        ("sorting_signals", args.sorting_signals, "SortingSignalsSwissprot.csv"),
    ]:
        frame = pd.read_csv(path)
        accessions, sequences = set(frame.ACC), set(frame.Sequence)
        sources[name] = {
            "url": "https://services.healthtech.dtu.dk/services/DeepLoc-2.1/data/"
                   + url,
            "sha256": file_sha256(path),
            "rows": len(frame),
            "matches": [
                {
                    "protein_id": r["primaryAccession"],
                    "accession_overlap": r["primaryAccession"] in accessions,
                    "exact_sequence_overlap": r["sequence"]["value"] in sequences,
                }
                for r in selected
            ],
        }
    outputs = {
        "uniprot-records.json": selected,
        "cohort-manifest.json": manifest,
        "annotations.json": [a.model_dump() for a in annotations],
        "group-membership.json": dict(
            zip([r["primaryAccession"] for r in eligible], groups)
        ),
        "supervision-overlap.json": {
            "sources": sources,
            "interpretation": (
                "Model-behavior concordance only. Exact overlap does not rule out "
                "homologous supervision overlap; foundation pretraining overlap "
                "remains unknown."
            ),
        },
    }
    for name, data in outputs.items():
        (args.output / name).write_text(json.dumps(data, indent=2) + "\n")
    (args.output / "cohort.fasta").write_text(
        "".join(
            f">{r['primaryAccession']}\n{r['sequence']['value']}\n" for r in selected
        )
    )


if __name__ == "__main__":
    main()
