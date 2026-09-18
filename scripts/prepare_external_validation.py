"""Prepare blinded annotation forms and dependence references, never predictions.

Run from the repository root. Custodian files must not be sent to annotators.
The whole accepted cohort is included so uncertain evidence is not screened out
using predictions or a fragile keyword rule.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def publications(value: Any) -> set[str]:
    found = set()
    if isinstance(value, dict):
        if value.get("source") == "PubMed" and value.get("id"):
            found.add("PMID:" + str(value["id"]))
        for item in value.values():
            found.update(publications(item))
    elif isinstance(value, list):
        for item in value:
            found.update(publications(item))
    return found


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    data_path = Path("examples/experiments/research-revision/dataset.csv")
    errors_path = Path(
        "examples/experiments/research-revision/diagnostics/final-error-analysis.csv"
    )
    data = pd.read_csv(data_path).sort_values("accession")
    errors = pd.read_csv(errors_path).set_index("accession")
    packet, private = [], []
    rng = np.random.default_rng(20260919)
    for number, i in enumerate(rng.permutation(len(data))):
        row = data.iloc[i]
        case = f"INT-{number + 1:05d}"
        annotation = json.loads(row.localization_annotations)
        notes = [
            t["value"]
            for c in annotation
            for t in c.get("note", {}).get("texts", [])
            if t.get("value", "").strip()
        ]
        sources = sorted(
            publications(annotation) | publications(json.loads(row.evidence))
        )
        cn = False
        if row.accession in errors.index:
            predictions = str(errors.loc[row.accession, "predicted_labels"]).split(";")
            cn = any(
                {row.label, label} == {"Cytoplasm", "Nucleus"} for label in predictions
            )
        packet.append(
            {
                "case_id": case,
                "accession": row.accession,
                "sequence": row.sequence,
                "primary_publication_ids": ";".join(sources),
                "blinded_dossier_ref": "",
                "reviewer_id": "",
                "context": "",
                "assay_sequence_attribution": "",
                "detection_breadth": "",
                "supported_label_set": "",
                "other_locations": "",
                "status": "",
                "note_and_ambiguity_review_complete": "",
                "evidence_rationale": "",
                "submitted_at": "",
                "model_blind_attestation": "",
            }
        )
        private.append(
            {
                "case_id": case,
                "accession": row.accession,
                "previous_label": row.label,
                "split": row.split,
                "cytoplasm_nucleus_disagreement": cn,
                "note_bearing": bool(notes),
                "free_text_notes": notes,
                "stored_annotations": annotation,
                "review_status": "pending_independent_review",
            }
        )
    for name in ("reviewer-a.csv", "reviewer-b.csv"):
        write_csv(output / name, packet)
    (output / "custodian-key.json").write_text(json.dumps(private, indent=2) + "\n")
    history = {
        "accessions": data.accession.tolist(),
        "sequence_sha256": [
            hashlib.sha256(s.encode()).hexdigest() for s in data.sequence
        ],
        "study_ids": sorted(
            set().union(
                *(
                    publications(json.loads(r.localization_annotations))
                    | publications(json.loads(r.evidence))
                    for r in data.itertuples()
                )
            )
        ),
        "includes": (
            "All 2107 accepted records: development plus historical-related; "
            "includes all 1822 original baseline records"
        ),
        "gene_mapping_status": (
            "not_available_in_snapshot_requires_independent_crosswalk_before_external_seal"
        ),
    }
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output / "history-reference.fasta").write_text(
        "".join(f">D_{r.accession}\n{r.sequence}\n" for r in data.itertuples())
    )
    source_path = Path("artifacts/human-v2/source.json")
    source = json.loads(source_path.read_text())["results"]
    # Include rejected records as possible bridges, not as model-training cases.
    fasta = "".join(
        f">B_{r['primaryAccession']}\n{r['sequence']['value']}\n" for r in source
    )
    (output / "bridge-reference.fasta.gz").write_bytes(
        gzip.compress(fasta.encode(), mtime=0)
    )
    counts = {
        "review_scope": (
            "all accepted internal records, not an external validation cohort"
        ),
        "records": len(packet),
        "cytoplasm_nucleus_disagreements_any_saved_oof_repeat": sum(
            r["cytoplasm_nucleus_disagreement"] for r in private
        ),
        "note_bearing_records": sum(r["note_bearing"] for r in private),
        "either_trigger": sum(
            r["note_bearing"] or r["cytoplasm_nucleus_disagreement"] for r in private
        ),
        "reviewed_by_independent_humans": 0,
        "historical_predictions_generated": False,
        "new_model_predictions_generated": False,
        "bridge_records": len(source),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "input_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (data_path, errors_path)
        },
    }
    (output / "preparation.json").write_text(json.dumps(counts, indent=2) + "\n")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args().output), indent=2))
