"""Post-selection calibration falsification for the exact frozen candidate.

This is exploratory development evidence, not another independent test. Preserve
the original completed nested run and write the final artifact separately.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from mapexploc.baseline import checksum
from mapexploc.experiments import atomic_json, read_json
from mapexploc.research import (
    fit_temperature,
    group_bootstrap,
    inner_fit,
    load_development,
    read_record,
    research_features,
    score,
    temperature_scale,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(
            "Use a new finalization directory; do not overwrite a frozen candidate"
        )
    protocol = read_json(args.run / "protocol.json")
    source = Path(protocol["source"])
    if checksum(source) != protocol["source_sha256"]:
        raise ValueError("Source changed")
    frame, _ = load_development(source)
    artifact = joblib.load(args.run / "research-model.joblib")
    candidate = artifact["candidate"]
    design = read_json(args.run / "design.json")
    x, y = research_features(frame.sequence), frame.label.to_numpy()
    raw = np.zeros((len(protocol["seeds"]), len(frame), 5))
    calibrated = raw.copy()
    temperatures = []
    with threadpool_limits(limits=1):
        for fold in design["folds"]:
            train, valid = np.array(fold["train"]), np.array(fold["validation"])
            inner = [
                (np.array(f["train"]), np.array(f["validation"])) for f in fold["inner"]
            ]
            result = inner_fit(candidate, x.iloc[train], y[train], inner, fold["seed"])
            temperature = fit_temperature(y[train], np.array(result["probabilities"]))
            record = read_record(
                args.run / "folds" / f'{fold["seed"]}-{fold["fold"]}.json'
            )
            p = np.array(record["probabilities"][candidate["id"]])
            seed_index = protocol["seeds"].index(fold["seed"])
            raw[seed_index, valid] = p
            calibrated[seed_index, valid] = temperature_scale(p, temperature)
            temperatures.append(
                {"seed": fold["seed"], "fold": fold["fold"], "temperature": temperature}
            )
    truth = np.tile(y, len(raw))
    groups = np.tile(frame.group.to_numpy(), len(raw))
    raw, calibrated = raw.reshape(-1, 5), calibrated.reshape(-1, 5)
    raw_metrics, cal_metrics = score(truth, raw), score(truth, calibrated)
    uncertainty = group_bootstrap(truth, calibrated, raw, groups)
    gain = raw_metrics["log_loss"] - cal_metrics["log_loss"]
    retain = bool(
        gain >= 0.02 and uncertainty["log_loss_difference_97_5_interval"][1] < 0
    )
    report = {
        "candidate": candidate,
        "status": "post_selection_development_diagnostic",
        "reason": (
            "Primary outer selection used a different model family in five of six folds"
        ),
        "source_sha256": checksum(source),
        "script_sha256": checksum(Path(__file__)),
        "primary_protocol_sha256": checksum(args.run / "protocol.json"),
        "raw": raw_metrics,
        "calibrated": cal_metrics,
        "paired_bootstrap": uncertainty,
        "log_loss_reduction": gain,
        "retain_calibration": retain,
        "outer_temperatures": temperatures,
        "primary_final_temperature": artifact["temperature"],
        "decision_rule": (
            "Retain only if loss reduction >=0.02 and paired interval excludes zero"
        ),
        "limitation": (
            "Candidate was chosen using all development data; "
            "intervals are exploratory, not confirmatory"
        ),
    }
    if not retain:
        artifact["temperature"] = 1.0
    artifact["finalization_status"] = (
        "development_candidate_no_independent_confirmation"
    )
    artifact["calibration_decision"] = report["status"]
    args.output.mkdir(parents=True)
    path = args.output / "model.joblib"
    joblib.dump(artifact, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(
        artifact["model"].predict_proba(x.iloc[:10]),
        restored["model"].predict_proba(x.iloc[:10]),
        atol=0,
        rtol=0,
    )
    report.update(
        {
            "final_temperature": artifact["temperature"],
            "artifact_sha256": checksum(path),
            "artifact_bytes": path.stat().st_size,
            "promoted": False,
        }
    )
    atomic_json(args.output / "finalization.json", report)
    np.savez_compressed(
        args.output / "calibration-predictions.npz",
        raw=raw,
        calibrated=calibrated,
        truth=truth,
        groups=groups,
    )
    print(
        {
            "candidate": candidate["id"],
            "gain": gain,
            "retain_calibration": retain,
            "final_temperature": artifact["temperature"],
        }
    )


if __name__ == "__main__":
    main()
