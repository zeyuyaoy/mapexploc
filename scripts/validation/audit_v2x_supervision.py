"""Exact, accession, global-homology and local-domain supervision audit."""

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from mapexploc.baseline import search_similar
from mapexploc.execution import atomic_json
from mapexploc.provenance import file_sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohorts", type=Path, required=True)
    p.add_argument("--localization", type=Path, required=True)
    p.add_argument("--sorting-signals", type=Path, required=True)
    p.add_argument("--membrane", type=Path)
    p.add_argument("--hpa", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    members = [
        m
        for name in ["development", "reference_pool", "evaluation"]
        for m in json.loads((args.cohorts / f"v2x-{name}.json").read_text())["members"]
    ]
    query = args.output / "query.fasta"
    query.write_text("".join(f">{m['protein_id']}\n{m['sequence']}\n" for m in members))
    sources = {}
    for name, path in [
        ("localization", args.localization),
        ("sorting_signals", args.sorting_signals),
        ("membrane", args.membrane),
        ("hpa_test", args.hpa),
    ]:
        if path is None:
            continue
        frame = pd.read_csv(path)
        if name == "hpa_test":
            frame = frame.rename(columns={"sid": "ACC", "fasta": "Sequence"})
        target = args.output / f"{name}.fasta"
        target.write_text(
            "".join(
                f">{i}\n{row.Sequence}\n" for i, row in enumerate(frame.itertuples())
            )
        )
        global_file = args.output / f"{name}-global.tsv"
        pairs = search_similar(query, target, global_file, 4)
        global_ids = {a for a, b in pairs}
        local_file = args.output / f"{name}-local.tsv"
        command = [
            "mmseqs",
            "easy-search",
            str(query),
            str(target),
            str(local_file),
            str(args.output / f"{name}-local.tmp"),
            "--min-seq-id",
            "0.3",
            "-c",
            "0",
            "--alignment-mode",
            "3",
            "-s",
            "7.5",
            "-e",
            "0.001",
            "--max-seqs",
            "10000",
            "--threads",
            "4",
            "--format-output",
            "query,target,fident,alnlen,qcov,tcov,evalue",
            "-v",
            "1",
        ]
        subprocess.run(command, check=True)
        local_links = {}
        for line in local_file.read_text().splitlines():
            fields = line.split("\t")
            if len(fields) == 7 and int(fields[3]) >= 50:
                local_links.setdefault(fields[0], []).append(
                    {
                        "accession": str(frame.iloc[int(fields[1])].ACC),
                        "aligned_residues": int(fields[3]),
                        "identity": float(fields[2]),
                        "evalue": float(fields[6]),
                    }
                )
        ids = set(frame.ACC)
        sequences = set(frame.Sequence)
        for output in (global_file, local_file):
            output.write_text(
                "\n".join(sorted(set(output.read_text().splitlines()))) + "\n"
            )
        sources[name] = {
            "source_sha256": file_sha256(path),
            "source_rows": len(frame),
            "global_search": "identity>=0.3, coverage>=0.8 both sequences",
            "local_search": (
                "identity>=0.3, alignment>=50 residues, E<=0.001, no global-coverage"
                " requirement; audit links, not automatic split grouping"
            ),
            "local_command": command,
            "global_pairs_sha256": file_sha256(global_file),
            "local_pairs_sha256": file_sha256(local_file),
            "matches": [
                {
                    "protein_id": m["protein_id"],
                    "split": m["split"],
                    "accession_overlap": (
                        None if name == "hpa_test" else m["protein_id"] in ids
                    ),
                    "accession_namespace": (
                        "Ensembl protein ID; accession comparison unavailable"
                        if name == "hpa_test"
                        else "UniProt accession"
                    ),
                    "exact_sequence_overlap": m["sequence"] in sequences,
                    "global_homology_overlap": m["protein_id"] in global_ids,
                    "local_domain_links": local_links.get(m["protein_id"], []),
                }
                for m in members
            ],
        }
    atomic_json(
        args.output / "supervision-overlap.json",
        {
            "sources": sources,
            "unavailable_sources": {
                **(
                    {"independent_HPA_test": "unknown_not_installed"}
                    if args.hpa is None
                    else {}
                ),
                **(
                    {"membrane_supervision": "unknown_not_installed"}
                    if args.membrane is None
                    else {}
                ),
                "foundation_model_pretraining": "unknown",
            },
            "interpretation": (
                "Model-behavior concordance only. Missing source datasets remain"
                " unknown; absence of detected similarity is not proof of independence."
            ),
            "mmseqs_version": (
                subprocess.check_output(["mmseqs", "version"], text=True).strip()
            ),
        },
    )
    print(
        json.dumps(
            {
                k: {
                    "rows": v["source_rows"],
                    "exact_matches": sum(
                        r["exact_sequence_overlap"] for r in v["matches"]
                    ),
                    "global_overlap": sum(
                        r["global_homology_overlap"] for r in v["matches"]
                    ),
                    "local_domain_overlap": sum(
                        bool(r["local_domain_links"]) for r in v["matches"]
                    ),
                }
                for k, v in sources.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
