import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mapexploc.artifacts import load_model_artifact
from mapexploc.explainers.shap import ShapExplainer
from mapexploc.features import build_feature_matrix


def test_frozen_baseline_predictions_and_shap() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "examples/models/human-baseline.joblib"
    if not path.exists():
        pytest.skip("Scientific reference assets are repository-only")
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "cd8ecb31c3c177efcf136fa84791839c6ca185ce3525a77aae9155a87b062082"
    )
    data = pd.read_csv(root / "examples/baseline/dataset.csv").set_index("accession")
    expected = pd.read_csv(root / "examples/models/human-baseline.predictions.csv")
    a = load_model_artifact(path)
    model = a.model
    features = build_feature_matrix(data.loc[expected.accession, "sequence"].tolist())
    np.testing.assert_array_equal(model.predict(features), expected.prediction)
    np.testing.assert_allclose(
        model.predict_proba(features),
        expected.filter(like="probability_").to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    expl = ShapExplainer(model).explain_sample(features.iloc[:10])
    np.testing.assert_allclose(
        expl["shap_values"].sum(axis=2) + expl["expected_value"],
        model.predict_proba(features.iloc[:10]),
        atol=1e-7,
    )


def test_frozen_candidate_predictions_and_shap() -> None:
    root = Path(__file__).resolve().parents[1] / "examples/experiments/human-v2"
    path = root / "candidate.joblib"
    if not path.exists():
        pytest.skip("Scientific reference assets are repository-only")
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "28802e97d55d19e9864a2388e6e11e32e9ed46117db5caf8bf26002fbf8ae505"
    )
    data = pd.read_csv(root / "historical.csv").set_index("accession")
    expected = pd.read_csv(root / "predictions.csv")
    artifact = load_model_artifact(path)
    assert artifact.metadata["evaluation_status"] == "development_only"
    features = build_feature_matrix(data.loc[expected.accession, "sequence"].tolist())
    np.testing.assert_array_equal(artifact.model.predict(features), expected.prediction)
    columns = [f"probability_{label}" for label in artifact.classes]
    np.testing.assert_allclose(
        artifact.model.predict_proba(features),
        expected[columns].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    explanation = ShapExplainer(artifact.model).explain_sample(features.iloc[:10])
    np.testing.assert_allclose(
        explanation["shap_values"].sum(axis=2) + explanation["expected_value"],
        artifact.model.predict_proba(features.iloc[:10]),
        atol=1e-7,
    )
