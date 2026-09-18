"""Default resolution never trusts a working-directory model implicitly."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mapexploc import default_model
from mapexploc.api import create_app
from mapexploc.artifacts import ModelArtifactError, save_model_artifact
from mapexploc.baseline import checksum
from mapexploc.explainers.shap import ShapExplainer
from mapexploc.features import build_feature_matrix


def fixture_model():  # type: ignore[no-untyped-def]
    features = build_feature_matrix(["AAAAA", "CCCCC", "DDDDD"])
    model = Pipeline(
        [
            ("normalize", StandardScaler()),
            ("trees", ExtraTreesClassifier(n_estimators=4, random_state=42)),
        ]
    )
    return model.fit(features, ["a", "b", "c"])


def repository(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    save_model_artifact(
        fixture_model(), tmp_path / "model.joblib", metadata={"model_id": "fixture"}
    )
    manifest = {
        "schema_version": 1,
        "artifact_path": "model.joblib",
        "model_id": "fixture",
        "sha256": checksum(tmp_path / "model.joblib"),
    }
    (tmp_path / "config/default-model.json").write_text(json.dumps(manifest))
    return tmp_path


def test_manifest_is_cwd_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = repository(tmp_path)
    monkeypatch.delenv("MAPEXPLOC_MODEL_PATH", raising=False)
    monkeypatch.setattr(default_model, "source_root", lambda: root)
    monkeypatch.chdir(tmp_path.parent)
    assert (
        default_model.resolve_default_model().load().metadata["model_id"] == "fixture"
    )
    client = TestClient(create_app(use_default=True))
    assert client.get("/health").json()["status"] == "ready"
    assert (
        client.get("/model").json()["metadata"]["model_family"]
        == "ExtraTreesClassifier"
    )
    assert client.post("/explain", json={"sequences": ["AAAAA"]}).status_code == 200
    # A valid repository fallback must not conceal an invalid explicit override.
    monkeypatch.setenv("MAPEXPLOC_MODEL_PATH", "missing.joblib")
    assert (
        TestClient(create_app(use_default=True)).get("/health").json()["status"]
        == "model_unavailable"
    )
    monkeypatch.setenv("MAPEXPLOC_MODEL_PATH", "")
    with pytest.raises(ModelArtifactError, match="empty"):
        default_model.resolve_default_model()


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("artifact_path", "../elsewhere"),
        ("artifact_path", "nested/../model.joblib"),
        ("artifact_path", ""),
        ("artifact_path", "/tmp/model.joblib"),
        ("sha256", "bad"),
        ("model_id", ""),
        ("model_id", " "),
    ],
)
def test_invalid_manifest(tmp_path: Path, field: str, value: object) -> None:
    root = repository(tmp_path)
    manifest = root / "config/default-model.json"
    payload = json.loads(manifest.read_text())
    payload[field] = value
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ModelArtifactError):
        default_model.manifest_selection(root)


def test_symlink_escape_and_malformed_manifest(tmp_path: Path) -> None:
    root = repository(tmp_path)
    manifest = root / "config/default-model.json"
    payload = json.loads(manifest.read_text())
    (root / "outside.joblib").symlink_to(tmp_path.parent / "outside.joblib")
    payload["artifact_path"] = "outside.joblib"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ModelArtifactError):
        default_model.manifest_selection(root)
    manifest.write_text("invalid json")
    with pytest.raises(ModelArtifactError):
        default_model.manifest_selection(root)


def test_source_default_from_another_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    if not (root / "examples/models/human-baseline.joblib").exists():
        pytest.skip("Scientific reference assets are repository-only")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MAPEXPLOC_MODEL_PATH", raising=False)
    client = TestClient(create_app(use_default=True))
    assert client.get("/health").json()["status"] == "ready"
    metadata = client.get("/model").json()["metadata"]
    assert metadata["model_id"] == "human-2026_03-56fb89f09b33"
    assert metadata["evaluation_status"] == "historical_holdout"


def test_checksum_identity_and_missing_artifact(tmp_path: Path) -> None:
    root = repository(tmp_path)
    selection = default_model.manifest_selection(root)
    with pytest.raises(ModelArtifactError, match="identity"):
        default_model.ModelSelection(selection.path, selection.sha256, "wrong").load()
    selection.path.write_text("corrupt")
    with pytest.raises(ModelArtifactError, match="checksum"):
        selection.load()
    selection.path.unlink()
    with pytest.raises(FileNotFoundError):
        selection.load()


def test_overrides_and_missing_wheel_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(default_model, "source_root", lambda: None)
    monkeypatch.delenv("MAPEXPLOC_MODEL_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    save_model_artifact(fixture_model(), tmp_path / "model.pkl")
    client = TestClient(create_app(use_default=True))
    assert client.get("/health").json()["status"] == "model_unavailable"
    assert "MAPEXPLOC_MODEL_PATH" in client.get("/model").json()["detail"]
    monkeypatch.setenv("MAPEXPLOC_MODEL_PATH", "model.pkl")
    assert default_model.resolve_default_model().path == tmp_path / "model.pkl"
    assert (
        TestClient(create_app(use_default=True)).get("/health").json()["model_loaded"]
    )
    monkeypatch.setenv("MAPEXPLOC_MODEL_PATH", "missing.joblib")
    assert (
        not TestClient(create_app(use_default=True))
        .get("/health")
        .json()["model_loaded"]
    )
    assert (
        TestClient(create_app(model=fixture_model(), use_default=True))
        .get("/health")
        .json()["model_loaded"]
    )
    assert (
        TestClient(create_app(model_path=tmp_path / "model.pkl", use_default=True))
        .get("/health")
        .json()["model_loaded"]
    )


def test_extra_trees_shap_and_class_order() -> None:
    model = fixture_model()
    features = build_feature_matrix(["AAAAA", "CCCCC"])
    explanation = ShapExplainer(model).explain_sample(features)
    import numpy as np

    np.testing.assert_allclose(
        explanation["shap_values"].sum(axis=2) + explanation["expected_value"],
        model.predict_proba(features),
        atol=1e-7,
    )
