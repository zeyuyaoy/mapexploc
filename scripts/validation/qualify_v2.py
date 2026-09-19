"""Real-checkpoint release gate; run with an installed wheel outside the checkout.

Unlike offline CI, this command never skips unavailable DeepLoc assets. It runs
CLI analyses for the external model and RF, then compares native probabilities
and complete attributions against frozen reference reports.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import mapexploc
from mapexploc import AnalysisReport


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--deeploc-config", type=Path, required=True)
    parser.add_argument("--rf-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not Path(mapexploc.__file__)
        .resolve()
        .is_relative_to(Path(sys.prefix).resolve())
    ):
        raise RuntimeError("This gate requires an installed wheel, not source imports")
    reference = AnalysisReport.model_validate_json(args.reference.read_text())
    # The predetermined first two proteins; do not select based on concordance.
    proteins = reference.results[:2]
    args.output.mkdir(parents=True, exist_ok=True)
    fasta = args.output / "proteins.fasta"
    fasta.write_text(
        "".join(f">{p.protein.protein_id}\n{p.protein.sequence}\n" for p in proteins)
    )
    annotations = args.output / "annotations.json"
    annotations.write_text(
        json.dumps([a.model_dump() for p in proteins for a in p.annotations])
    )
    configuration = args.output / "analysis.json"
    configuration.write_text(
        json.dumps(
            {
                "cohort_id": "clean-release-repeat",
                "selection_criteria": (
                    "First two frozen accession-ordered biological control proteins"
                ),
                "seed": 42,
                "coalition_budget": 512,
                "references": 4,
            }
        )
    )
    rf_config = args.output / "rf.json"
    rf_config.write_text(json.dumps({"path": str(args.rf_checkpoint.resolve())}))
    reports = {}
    for name, adapter, config in [
        ("deeploc", "deeploc2", args.deeploc_config),
        ("rf", "feature_artifact", rf_config),
    ]:
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "mapexploc.cli",
                "analyze",
                "--adapter",
                adapter,
                "--adapter-config",
                str(config.resolve()),
                "--fasta",
                str(fasta.resolve()),
                "--configuration",
                str(configuration.resolve()),
                "--annotations",
                str(annotations.resolve()),
                "--output-dir",
                str((args.output / name).resolve()),
            ],
            check=True,
        )
        reports[name] = AnalysisReport.model_validate_json(
            (args.output / name / "report.json").read_text()
        )
        if (
            reports[name].cohort.included_count != 2
            or not (args.output / name / "report.html").is_file()
        ):
            raise RuntimeError("Incomplete report")
    repeated = reports["deeploc"]
    if repeated.model.checkpoint_sha256 != reference.model.checkpoint_sha256:
        raise RuntimeError("External checkpoint differs from the frozen reference")
    probabilities = max(
        float(np.max(np.abs(np.array(a.probabilities) - b.probabilities)))
        for a, b in zip(repeated.results, proteins)
    )
    attributions = max(
        float(np.max(np.abs(np.array(a.attributions) - b.attributions)))
        for a, b in zip(repeated.results, proteins)
    )
    if probabilities > 1e-5 or attributions > 1e-5:
        raise RuntimeError("Clean reproduction exceeds declared numerical tolerance")
    result = {
        "status": "passed",
        "installed_package": str(Path(mapexploc.__file__).resolve()),
        "models": [r.model.model_id for r in reports.values()],
        "checkpoint_sha256": repeated.model.checkpoint_sha256,
        "protein_ids": [p.protein.protein_id for p in proteins],
        "probability_max_absolute_delta": probabilities,
        "attribution_max_absolute_delta": attributions,
        "tolerance": 1e-5,
        "software": repeated.software,
        "scope": (
            "Installed wheel CLI, two real pretrained models, two-protein "
            "local/global JSON/HTML/CSV, full native preprocessing and "
            "sequence-region SHAP. No checkpoint test skipped."
        ),
    }
    (args.output / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
