"""End-to-end API contract tests."""

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mapexploc.api import create_app
from mapexploc.artifacts import save_model_artifact
from mapexploc.features import build_feature_matrix
from mapexploc.models.rf import train_random_forest


def fitted_model():  # type: ignore[no-untyped-def]
    sequences = ["AAAAAA", "CCCCCC", "DDDDDD"]
    labels = pd.Series(["alpha", "beta", "gamma"])
    features = build_feature_matrix(sequences)
    return train_random_forest(
        features,
        labels,
        param_grid={"rf__n_estimators": [8], "rf__max_depth": [3]},
        n_jobs=1,
    )["model"]


def test_health_and_prediction_contract() -> None:
    client = TestClient(create_app(model=fitted_model()))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ready",
        "model_available": True,
        "model_loaded": True,
    }

    response = client.post("/predict", json={"sequences": ["aa aa aa"]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["sequence_length"] == 6
    assert len(payload["results"][0]["probabilities"]) == 3
    assert sum(
        item["probability"] for item in payload["results"][0]["probabilities"]
    ) == pytest.approx(1.0)


def test_explanation_contract() -> None:
    client = TestClient(create_app(model=fitted_model()))

    response = client.post("/explain", json={"sequences": ["AAAAAA"], "top_n": 5})

    assert response.status_code == 200, response.text
    report = response.json()["results"][0]
    assert report["prediction"] == "alpha"
    assert len(report["feature_contributions"]) == 5
    assert {"feature", "value", "contribution"} == set(
        report["feature_contributions"][0]
    )


def test_request_cannot_select_server_model_path(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pkl"
    save_model_artifact(fitted_model(), model_path)
    client = TestClient(create_app(model_path=model_path))

    response = client.post(
        "/predict",
        json={"sequences": ["AAAAAA"], "model_path": "/tmp/other.pkl"},
    )

    assert response.status_code == 422


def test_validation_and_missing_model_errors(tmp_path: Path) -> None:
    missing_client = TestClient(create_app(model_path=tmp_path / "missing.pkl"))
    assert missing_client.get("/health").json()["status"] == "model_unavailable"
    assert (
            missing_client.post("/predict", json={"sequences": ["AAAA"]}).status_code == 503
    )

    client = TestClient(create_app(model=fitted_model()))
    assert client.post("/predict", json={"sequences": ["AAX"]}).status_code == 422


def test_features_work_without_a_model(tmp_path: Path) -> None:
    client = TestClient(create_app(model_path=tmp_path / "missing.pkl"))
    response = client.post("/features", json={"sequences": ["AAaa", "ACDE"]})
    assert response.status_code == 200
    rows = response.json()["results"]
    assert rows[0]["composition"]["A"] == 1
    assert rows[1]["index"] == 1
    assert sum(rows[1]["composition"].values()) == pytest.approx(1)
    assert rows[0]["sequence_length"] == 4
    assert client.post("/features", json={"sequences": ["AX"]}).status_code == 422


def test_corrupt_model_is_not_ready(tmp_path: Path) -> None:
    path = tmp_path / "broken.pkl"
    path.write_bytes(b"not a model")
    client = TestClient(create_app(model_path=path))
    assert client.get("/health").json()["status"] == "model_unavailable"
    assert client.get("/model").status_code == 503
    assert client.post("/predict", json={"sequences": ["AAA"]}).status_code == 503


def test_public_metadata_is_allowlisted_and_legacy_is_explicit(tmp_path: Path) -> None:
    import joblib

    model = fitted_model()
    path = tmp_path / "versioned.pkl"
    save_model_artifact(
        model, path, metadata={"name": "Test fixture", "secret_path": "/private/model"}
    )
    client = TestClient(create_app(model_path=path))
    payload = client.get("/model").json()
    assert payload["metadata"] == {"name": "Test fixture"}
    assert payload["metadata_available"] is True
    assert len(payload["model_classes"]) == 3
    legacy = tmp_path / "legacy.pkl"
    joblib.dump(model, legacy)
    payload = TestClient(create_app(model_path=legacy)).get("/model").json()
    assert payload["metadata_available"] is False
    assert payload["metadata"] == {}


def test_shap_full_contributions_reconstruct_probability() -> None:
    from mapexploc.explainers.shap import ShapExplainer

    model = fitted_model()
    features = build_feature_matrix(["AAAAAA", "CCCCCC"])
    reports = ShapExplainer(model).explain_predictions(features, top_n=423)
    probabilities = model.predict_proba(features)
    for index, report in enumerate(reports):
        reconstructed = report["base_value"] + sum(
            item["contribution"] for item in report["feature_contributions"]
        )
        assert reconstructed == pytest.approx(probabilities[index].max(), abs=1e-6)


def test_wrong_feature_count_is_not_a_ready_model(tmp_path: Path) -> None:
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=2, random_state=42).fit(
        [[0, 0], [1, 1]], ["a", "b"]
    )
    path = tmp_path / "incompatible.pkl"
    save_model_artifact(model, path)
    client = TestClient(create_app(model_path=path))
    assert client.get("/health").json()["status"] == "model_unavailable"


def test_batch_and_explanation_limits() -> None:
    client = TestClient(create_app(model=fitted_model()))
    for sequences in [[], ["AAA"] * 101, ["A" * 100001], ["A" * 100000] * 11]:
        assert (
                client.post("/features", json={"sequences": sequences}).status_code == 422
        )
    assert (
            client.post("/explain", json={"sequences": ["AAA"], "top_n": 26}).status_code
            == 422
    )
