"""Pin the already fitted artifacts, protocol and execution code without refitting.

Exclusive creation only. This is a local lock, not independent preregistration.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np

from mapexploc.external_validation import STUDY_STATE, digest, read, utc_now, write_new
from mapexploc.features import AMINO_ACIDS, build_feature_matrix
from mapexploc.research import feature_columns

LR = "examples/experiments/research-revision/final/model.joblib"
RF = "examples/models/human-baseline.joblib"
EXPECTED = {
    LR: "e760d2ec4ab99d64d495cbb81307c4bf4302e04366fd807745d4680c83e67943",
    RF: "cd8ecb31c3c177efcf136fa84791839c6ca185ce3525a77aae9155a87b062082",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=STUDY_STATE / "freeze-v1.1.json",
    )
    parser.add_argument("--supersedes", type=Path, default=STUDY_STATE / "freeze.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Already frozen; never overwrite the protocol lock")
    if (STUDY_STATE / "primary-evaluation-started.json").exists():
        raise ValueError(
            "A study evaluation has started; a maintenance re-lock is forbidden"
        )
    previous = read(args.supersedes)
    if any(previous["files"].get(p) != h for p, h in EXPECTED.items()):
        raise ValueError("Previous manifest does not identify the same fitted models")
    for path, expected in EXPECTED.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Wrong exact fitted model: {path}")
    fitted = joblib.load(LR)
    model = fitted["model"]
    classifier = model.named_steps["classifier"]
    if fitted["temperature"] != 1 or classifier.C != 0.1:
        raise ValueError("Unexpected logistic configuration")
    # Check legacy RF deserialization without scoring any real protein.
    # RF serialization used sklearn 1.9.0; the locked runtime uses 1.9.1.
    with warnings.catch_warnings(record=True) as caught:
        reference = joblib.load(RF)["model"]
    rng = np.random.default_rng(20260919)
    sequences = [
        "".join(rng.choice(list(AMINO_ACIDS), n))
        for n in (51, 100, 250, 500, 1000, 2000)
    ]
    x = build_feature_matrix(sequences)
    scaled = reference.named_steps["scaler"].transform(x).astype(np.float32)
    manual = np.zeros((len(x), 5))
    forest = reference.named_steps["rf"]
    for estimator in forest.estimators_:
        tree = estimator.tree_
        for i, values in enumerate(scaled):
            node = 0
            while tree.children_left[node] != tree.children_right[node]:
                node = (
                    tree.children_left[node]
                    if values[tree.feature[node]] <= tree.threshold[node]
                    else tree.children_right[node]
                )
            votes = tree.value[node, 0]
            manual[i] += votes / votes.sum()
    manual /= len(forest.estimators_)
    observed = reference.predict_proba(x)
    np.testing.assert_allclose(manual, observed, atol=1e-12, rtol=0)
    files = list(Path("src/mapexploc").rglob("*.py"))
    files += [Path(p) for p in EXPECTED]
    files += [
        Path("config/external-validation-v1.json"),
        Path("docs/external-validation.md"),
    ]
    files += list(Path("examples/validation/external-v1/annotation").glob("*"))
    files += list(Path("examples/validation/external-v1/templates").glob("*"))
    files += [Path("examples/validation/external-v1/precision-design.json")]
    files += [
        Path("scripts") / name
        for name in (
            "lock_external_validation.py",
            "prepare_external_validation.py",
            "screen_external_homology.py",
            "external_precision_design.py",
        )
    ]
    files += [
        Path("tests") / name
        for name in (
            "test_external_validation.py",
            "test_evaluation.py",
            "test_training_contracts.py",
        )
    ]
    files += [Path("docs/research-software.md"), args.supersedes]
    parameters = {}
    for name, values in (
        ("coefficients", classifier.coef_),
        ("intercepts", classifier.intercept_),
        ("scaler_mean", model.named_steps["scaler"].mean_),
        ("scaler_scale", model.named_steps["scaler"].scale_),
    ):
        parameters[name] = {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }
    result = {
        "study_id": "mapexploc-external-v1",
        "execution_revision": "1.1",
        "supersedes": {"path": str(args.supersedes), "sha256": digest(args.supersedes)},
        "amendment": (
            "Pre-evaluation software maintenance: formatting, shared generic metrics, "
            "training guards, label-set equality and documentation corrections. "
            "Exact fitted models, features, estimand, cohort gates and primary "
            "decision rules are unchanged. No external evaluation has started."
        ),
        "locked_at": utc_now(),
        "status": "local_lock_pending_independent_registration_and_cohort",
        "protocol": "config/external-validation-v1.json",
        "history": "examples/validation/external-v1/annotation/history.json",
        "logistic_model": LR,
        "reference_model": RF,
        "candidate": fitted["candidate"],
        "temperature": fitted["temperature"],
        "classes": fitted["classes"],
        "selected_features": feature_columns(fitted["candidate"]),
        "fitted_parameters": parameters,
        "training": (
            "Already fitted on 1741 development proteins; "
            "no retraining or recalibration in this stage"
        ),
        "inference": (
            "Frozen log1p length, fitted standardization, multinomial logistic "
            "softmax, temperature 1, argmax with stored class order for ties"
        ),
        "python": platform.python_version(),
        "platform_record": platform.platform(),
        "dependencies": {
            p: version(p)
            for p in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "biopython",
                "joblib",
                "threadpoolctl",
            )
        },
        "legacy_reference_check": {
            "synthetic_sequences": len(sequences),
            "max_difference_from_direct_tree_traversal": float(
                np.abs(manual - observed).max()
            ),
            "load_warning_types": sorted({type(w.message).__name__ for w in caught}),
            "interpretation": (
                "Numerical fixture check under pinned runtime; not an external "
                "biological result or blanket cross-version compatibility guarantee"
            ),
        },
        "files": {str(p): digest(p) for p in sorted(files)},
        "external_predictions_generated": False,
        "wet_lab_outcomes_produced": False,
    }
    write_new(args.output, result)
    print(f"Exact fitted models and protocol locked: {args.output}")


if __name__ == "__main__":
    main()
