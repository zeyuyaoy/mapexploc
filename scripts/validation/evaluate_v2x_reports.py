"""Untouched evaluation of frozen reports; never changes the selected method."""

import argparse
import json
from pathlib import Path

from mapexploc.biological_v2x import evaluate_family
from mapexploc.execution import atomic_json, fingerprint
from mapexploc.report_v3 import load_report
from mapexploc.study import FAMILIES, holm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-profile", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fast-report", type=Path, required=True)
    parser.add_argument("--accurate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = json.loads(args.selected_profile.read_text())
    checksum = selected.pop("selection_sha256")
    if (
        fingerprint(selected) != checksum
        or selected.get("stage") != "resolution_selection_complete"
    ):
        raise ValueError(
            "Complete frozen development selection required before evaluation"
        )
    manifest = json.loads(args.manifest.read_text())
    results = []
    for mode, path in [("fast", args.fast_report), ("accurate", args.accurate_report)]:
        report = load_report(path)
        if report.model.provenance.get("mode") != mode:
            raise ValueError("Evaluation model mode mismatch")
        if report.model.model_dump(mode="json") != selected["models"][mode]:
            raise ValueError("Evaluation model/runtime differs from frozen development")
        config = report.configuration.model_dump()
        expected = selected["configuration"]
        for key, value in expected.items():
            if (
                key not in {"cohort_id", "selection_criteria"}
                and config.get(key) != value
            ):
                raise ValueError(
                    "Evaluation method differs from the frozen selected configuration"
                )
        for family in FAMILIES:
            result = evaluate_family(report, manifest, family)
            result["mode"] = mode
            results.append(result)
    # All twelve hypotheses remain in the correction family; excluded tests
    # contribute p=1 rather than reducing multiplicity after seeing outcomes.
    adjusted = holm(
        [r.get("formal_p") if r.get("formal_p") is not None else 1.0 for r in results]
    )
    for row, p in zip(results, adjusted):
        row["holm_p"] = p if row.get("formal_p") is not None else None
    atomic_json(
        args.output,
        {
            "selected_profile_sha256": checksum,
            "evaluation_manifest_sha256": fingerprint(manifest),
            "results": results,
            "interpretation": (
                "All prespecified outcomes retained. Null findings narrow conclusions;"
                " the selected method is unchanged."
            ),
        },
    )


if __name__ == "__main__":
    main()
