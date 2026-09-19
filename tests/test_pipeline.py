"""Integration tests for the basic training pipeline."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

from mapexploc.config import load_config
from mapexploc.data import load_example_dataset
from mapexploc.features import build_feature_matrix
from mapexploc.models.rf import rf_predict, train_random_forest


def test_training_pipeline(tmp_path: Path) -> None:
    cfg = load_config(Path("config/default.yml"))
    df = load_example_dataset(
        Path(str(files("mapexploc").joinpath("examples/smoke.csv")))
    )

    # This tiny fixture tests the no-CV workflow, not predictive performance.

    X = build_feature_matrix(df["sequence"])

    param_grid: dict[str, list[Any]] = {
        "rf__n_estimators": [cfg.model.n_estimators],
        "rf__max_depth": [cfg.model.max_depth],
    }

    model = train_random_forest(X, df["label"], param_grid)
    preds = rf_predict(model["model"], X)
    assert len(preds) == len(df)
