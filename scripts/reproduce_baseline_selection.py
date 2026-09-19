"""Reproduce version 1 selection using training data only, preserving its test."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from mapexploc.baseline import checksum
from mapexploc.experiments import atomic_json, read_json, software
from mapexploc.features import build_feature_matrix
from mapexploc.validation import grouped_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path("examples/baseline/dataset.csv")
    manifest = read_json(source.with_name("manifest.json"))
    if checksum(source) != manifest["dataset_sha256"]:
        raise ValueError("Frozen baseline source changed")
    data = pd.read_csv(source)
    data = data.loc[data.split == "train"].reset_index(drop=True)
    x = build_feature_matrix(data.sequence)
    splits = grouped_splits(data.label, data.group, 3, manifest["seed"])
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    class_weight="balanced", random_state=manifest["seed"], n_jobs=1
                ),
            ),
        ]
    )
    search = GridSearchCV(
        model,
        {
            "rf__n_estimators": [128, 256],
            "rf__max_depth": [12, 24],
            "rf__min_samples_leaf": [1, 3],
        },
        scoring="f1_macro",
        cv=splits,
        n_jobs=1,
        error_score="raise",
    )
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        search.fit(x, data.label)
    # Read selection metadata without scoring historical holdout outcomes.
    recorded = read_json(Path("examples/models/human-baseline.report.json"))
    report = {
        "dataset_sha256": checksum(source),
        "train_rows": len(data),
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
        "recorded_cv_score": recorded["best_cv_score"],
        "recorded_params": recorded["best_params"],
        "params_reproduced": search.best_params_ == recorded["best_params"],
        "score_reproduced_at_1e_12": bool(
            np.isclose(
                search.best_score_, recorded["best_cv_score"], rtol=0, atol=1e-12
            )
        ),
        "software": software(),
        "seconds": time.perf_counter() - started,
        "test_evaluated": False,
        "comparisons": [
            {"params": p, "mean_macro_f1": float(m), "std_macro_f1": float(s)}
            for p, m, s in zip(
                search.cv_results_["params"],
                search.cv_results_["mean_test_score"],
                search.cv_results_["std_test_score"],
            )
        ],
    }
    atomic_json(args.output, report)
    print(
        {
            k: report[k]
            for k in (
            "best_cv_score",
            "params_reproduced",
            "score_reproduced_at_1e_12",
            "seconds",
        )
        }
    )


if __name__ == "__main__":
    main()
