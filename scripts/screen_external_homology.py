"""Sequence/bridge screening; separate remote and provenance audits are required."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from pathlib import Path

from mapexploc.external_validation import digest, sequence_input_digest, write_new
from mapexploc.features import normalize_protein_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument(
        "--references",
        type=Path,
        default=Path("examples/validation/external-v1/annotation"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = json.loads(args.cohort.read_text())["records"]
    ids = [r["case_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case IDs")
    # Safe opaque FASTA IDs, never parse free-text identifiers in MMseqs output.
    query_ids, fragments, unsupported = {}, [], []
    for i, record in enumerate(records):
        try:
            sequence = normalize_protein_sequence(record["sequence"])
        except (ValueError, TypeError):
            unsupported.append(
                {
                    "case_id": record["case_id"],
                    "sequence_status": "unsupported",
                    "sequence_component": None,
                    "connected_to_history": None,
                    "remote_domain_status": "unresolved",
                }
            )
            continue
        query_ids[f"E_{i}"] = record["case_id"]
        fragments.append(f">E_{i}\n{sequence}\n")
    queries = "".join(fragments)
    ref = args.references / "history-reference.fasta"
    bridges = args.references / "bridge-reference.fasta.gz"
    combined = args.output / "combined.fasta"
    combined.write_text(
        queries + ref.read_text() + gzip.decompress(bridges.read_bytes()).decode()
    )
    total = sum(line.startswith(">") for line in combined.read_text().splitlines())
    hits = args.output / "alignments.tsv"
    command = [
        "mmseqs",
        "easy-search",
        str(combined),
        str(combined),
        str(hits),
        str(args.output / "tmp"),
        "--min-seq-id",
        "0",
        "-c",
        "0",
        "--alignment-mode",
        "3",
        "-s",
        "7.5",
        "-e",
        "0.001",
        "--max-seqs",
        str(total),
        "--max-accept",
        str(total),
        "--mask",
        "1",
        "--comp-bias-corr",
        "1",
        "--threads",
        str(args.threads),
        "--format-output",
        "query,target,fident,qcov,tcov,alnlen,evalue",
        "-v",
        "1",
    ]
    tool_version = subprocess.check_output(["mmseqs", "version"], text=True).strip()
    with (args.output / "search.log").open("x") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    parent: dict[str, str] = {}

    def find(i: str) -> str:
        parent.setdefault(i, i)
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edge_count = 0
    with hits.open() as handle:
        for line in handle:
            a, b, identity, qcov, tcov, length, evalue = line.rstrip().split("\t")
            identity, qcov, tcov = float(identity), float(qcov), float(tcov)
            global_hit = identity >= 0.30 and min(qcov, tcov) >= 0.80
            local_hit = (
                identity >= 0.30
                and max(qcov, tcov) >= 0.50
                and int(length) >= 80
                and float(evalue) <= 1e-5
            )
            if a != b and (global_hit or local_hit):
                parent[find(a)] = find(b)
                edge_count += 1
    development = {
        line[1:] for line in ref.read_text().splitlines() if line.startswith(">")
    }
    contaminated = {find(i) for i in development}
    results = [
        {
            "case_id": case,
            "sequence_status": "screened",
            "sequence_component": find(i),
            "connected_to_history": find(i) in contaminated,
            "remote_domain_status": "unresolved",
        }
        for i, case in query_ids.items()
    ] + unsupported
    write_new(
        args.output / "sequence-screen.json",
        {
            "cohort_sha256": digest(args.cohort),
            "sequence_input_sha256": sequence_input_digest(records),
            "reference_sha256": digest(ref),
            "bridge_sha256": digest(bridges),
            "mmseqs_version": tool_version,
            "command": command,
            "accepted_directed_edges": edge_count,
            "alignments_sha256": digest(hits),
            "records": results,
            "limitations": [
                "Search sensitivity is finite; no-hit is not absence of homology",
                "Domain-profile, gene, publication and assay audits are "
                "separate mandatory gates",
                "Bridge universe is the frozen human source; "
                "unobserved interspecies bridges may exist",
            ],
        },
    )


if __name__ == "__main__":
    main()
