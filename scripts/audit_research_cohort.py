"""Audit stored annotation evidence, without reading historical predictions."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from mapexploc.baseline import checksum
from mapexploc.experiments import atomic_json, read_json
from mapexploc.research import (
    has_location_note,
    load_development,
    publications,
    study_groups,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("examples/experiments/research-revision/dataset.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame, audit = load_development(args.source)
    all_rows = pd.read_csv(args.source)
    reserved = all_rows.loc[all_rows.split != "development"]
    support = Counter(p for evidence in frame.evidence for p in publications(evidence))
    reserved_studies = {
        p for evidence in reserved.evidence for p in publications(evidence)
    }
    combined = study_groups(frame)
    annotations = []
    for row in frame.itertuples():
        if has_location_note(row.localization_annotations):
            notes = [
                t["value"]
                for c in json.loads(row.localization_annotations)
                for t in c.get("note", {}).get("texts", [])
                if t.get("value")
            ]
            annotations.append(
                {"accession": row.accession, "label": row.label, "notes": notes}
            )
    manifest = read_json(Path("examples/baseline/manifest.json"))
    preparation_path = args.source.parent / "source-preparation.json"
    if not preparation_path.exists():
        preparation_path = args.source.parent / "preparation.json"
    preparation = read_json(preparation_path)
    audit.update(
        {
            "dataset_sha256": checksum(args.source),
            "original_source_sha256": manifest["source_sha256"],
            "expanded_source_sha256": preparation["source_sha256"],
            "identical_source_snapshots": manifest["source_sha256"]
            == preparation["source_sha256"],
            "accepted_curated": len(all_rows),
            "exclusions": preparation["exclusions"],
            "supporting_studies_shared_with_reserved": len(
                set(support) & reserved_studies
            ),
            "largest_supporting_studies": support.most_common(10),
            "combined_sequence_study_groups": len(set(combined)),
            "largest_combined_group": int(pd.Series(combined).value_counts().max()),
            "length_summary": frame.sequence.str.len().describe().to_dict(),
            "note_free_class_counts": frame.loc[~frame.has_location_note]
            .label.value_counts()
            .to_dict(),
            "notes_present_class_counts": frame.loc[frame.has_location_note]
            .label.value_counts()
            .to_dict(),
            "limitations": [
                "No site, assay, tissue, gene or fragment-status fields in snapshot",
                "Missing localization is not a negative label",
                "Notes present does not mean labels are wrong; "
                "notes absent does not establish exclusive localization",
                "Only accepted records form the homology graph; "
                "remote/domain/bridge links may remain",
            ],
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "cohort-audit.json", audit)
    atomic_json(args.output / "development-localization-notes.json", annotations)
    print(
        json.dumps(
            {
                "development": len(frame),
                "notes": len(annotations),
                "study_groups": len(set(combined)),
            }
        )
    )


if __name__ == "__main__":
    main()
