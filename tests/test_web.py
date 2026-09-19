"""Production boundary regressions using the actual approved artifact."""

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from time import sleep

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mapexploc import api, default_model
from mapexploc.api import create_app
from mapexploc.artifacts import ModelArtifactError
from mapexploc.explainers.shap import ShapExplainer
from mapexploc.features import build_feature_matrix
from mapexploc.web import MAX_REQUEST_BYTES, create_web_app

ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = "MALWMRLLPLLALLALWGPDPAAA"
MODEL_ID = "human-2026_03-56fb89f09b33"


@pytest.fixture(scope="module")
def client():
    with TestClient(create_web_app(trusted_root=ROOT)) as instance:
        yield instance


def test_environment_is_ignored_even_on_fresh_import(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from fastapi.testclient import TestClient
from mapexploc.web import create_web_app
client = TestClient(create_web_app())
assert client.get('/api/health').json()['model_available']
assert len(client.get('/api/v3/models').json()['models']) == 1
""",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "MAPEXPLOC_ADAPTER_CATALOG": str(tmp_path / "missing-deeploc.json"),
            "MAPEXPLOC_MODEL_PATH": str(tmp_path / "unapproved.joblib"),
        },
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_owned_catalogue_never_configures_external_models(monkeypatch):
    monkeypatch.setenv("MAPEXPLOC_ADAPTER_CATALOG", "/secret/deeploc.json")
    monkeypatch.setenv("MAPEXPLOC_MODEL_PATH", "/secret/other.joblib")

    def forbidden(*args, **kwargs):
        pytest.fail("The public application configured an external adapter")

    monkeypatch.setattr(api, "ConfiguredModel", forbidden)
    c = TestClient(create_web_app())
    assert c.get("/api/model").json()["metadata"]["model_id"] == MODEL_ID
    v2 = c.get("/api/v2/models").json()["adapters"]
    assert list(v2) == ["default"]
    assert v2["default"]["model_id"] == MODEL_ID
    v3 = c.get("/api/v3/models").json()["models"]
    assert [row["adapter_id"] for row in v3] == ["default"]
    assert v3[0]["execution"] == "bounded_live"
    assert "deeploc" not in json.dumps(v3).lower()
    assert (
        c.post(
            "/api/v3/predict",
            json={
                "adapter_id": "deeploc-fast",
                "proteins": [{"protein_id": "p", "sequence": SEQUENCE}],
            },
        ).status_code
        == 404
    )


def test_research_environment_still_registers_deeploc(tmp_path, monkeypatch):
    configuration = json.loads(
        (ROOT / "config/adapters/deeploc.example.json").read_text()
    )
    catalogue = tmp_path / "research-catalogue.json"
    catalogue.write_text(
        json.dumps(
            {"native-fast": {"factory": "deeploc2", "configuration": configuration}}
        )
    )
    monkeypatch.setenv("MAPEXPLOC_ADAPTER_CATALOG", str(catalogue))
    client = TestClient(api.application_from_environment())
    models = client.get("/v3/models").json()["models"]
    native = next(row for row in models if row["adapter_id"] == "native-fast")
    assert native["factory"] == "deeploc2"
    assert native["mode"] == "fast"


def test_browser_routes_and_deterministic_smoke(client):
    assert client.get("/api/health").json()["status"] == "ready"
    model = client.get("/api/model").json()
    assert model["feature_count"] == 423
    assert model["metadata"]["model_id"] == MODEL_ID
    body = {"sequences": [SEQUENCE]}
    response = client.post("/api/predict", json=body)
    prediction = response.json()["results"][0]
    assert prediction["prediction"] == "Secreted"
    assert prediction["confidence"] == pytest.approx(0.517372666396104, abs=1e-8)
    assert client.post("/api/features", json=body).status_code == 200
    explanation = client.post("/api/explain", json=body).json()["results"][0]
    reconstructed = (
        explanation["base_value"]
        + explanation["remainder"]
        + sum(item["contribution"] for item in explanation["feature_contributions"])
    )
    assert reconstructed == pytest.approx(prediction["confidence"], abs=1e-6)
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-request-id"]
    assert "access-control-allow-origin" not in response.headers
    assert client.get("/api/missing").status_code == 404
    assert client.get("/predict").status_code == 404
    assert client.get("/api/openapi.json").status_code == 200


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_versioned_prediction_and_complete_report(client, version):
    body = {"proteins": [{"protein_id": "p", "sequence": SEQUENCE}]}
    prediction = client.post(f"/api/{version}/predict", json=body)
    assert prediction.status_code == 200
    if version == "v3":
        body["configuration"] = {"method_profile": "v2x-legacy"}
    report = client.post(f"/api/{version}/analyze", json=body)
    assert report.status_code == 200, report.text
    assert report.json()["schema_version"] == int(version[-1])
    assert report.json()["model"]["model_id"] == MODEL_ID
    assert len(report.json()["results"][0]["features"]) == 423


@pytest.mark.parametrize(
    "route,count,length",
    [
        ("predict", 26, 3),
        ("predict", 2, 25001),
        ("features", 26, 3),
        ("features", 2, 25001),
        ("explain", 6, 3),
        ("explain", 2, 5001),
        ("v2/predict", 26, 3),
        ("v3/predict", 2, 25001),
        ("v2/analyze", 21, 3),
        ("v3/analyze", 2, 10001),
    ],
)
def test_web_limits_before_inference(client, monkeypatch, route, count, length):
    monkeypatch.setattr(
        api._ModelRuntime,
        "get_adapter",
        lambda _: pytest.fail("Oversized request reached the model"),
    )
    sequences = ["A" * length] * count
    body = (
        {"sequences": sequences}
        if "/" not in route
        else {
            "proteins": [
                {"protein_id": str(i), "sequence": s} for i, s in enumerate(sequences)
            ]
        }
    )
    response = client.post(f"/api/{route}", json=body)
    assert response.status_code == 413
    assert "residues" in response.json()["detail"]


def test_limits_allow_exact_boundary_and_do_not_change_research(client):
    sequences = ["A" * 2000] * 25
    assert client.post("/api/predict", json={"sequences": sequences}).status_code == 200
    research = TestClient(create_app(use_default=True))
    assert (
        research.post("/features", json={"sequences": ["AAA"] * 26}).status_code == 200
    )
    assert (
        client.post("/api/explain", json={"sequences": ["A" * 2000] * 5}).status_code
        == 200
    )


def test_streamed_body_and_optional_report_work_are_bounded(client):
    response = client.post("/api/predict", content=iter([b" " * 1000] * 513))
    assert response.status_code == 413
    assert (
        client.post("/api/predict", content=b" " * (MAX_REQUEST_BYTES + 1)).status_code
        == 413
    )
    for configuration in [
        {"explainer": "region_kernel"},
        {"method_profile": "v2x-legacy", "bootstrap_replicates": 1001},
        {"method_profile": "v2x-legacy", "faithfulness": True},
    ]:
        response = client.post(
            "/api/v3/analyze",
            json={
                "proteins": [{"protein_id": "p", "sequence": SEQUENCE}],
                "configuration": configuration,
            },
        )
        assert 400 <= response.status_code < 500


def test_public_errors_and_logs_do_not_echo_internal_details(
    client, monkeypatch, caplog
):
    def broken(*args):
        raise RuntimeError("/private/operator/model.joblib secret-token")

    monkeypatch.setattr(api, "predict_probabilities", broken)
    response = client.post(
        "/api/v2/predict",
        json={
            "proteins": [{"protein_id": "p", "sequence": SEQUENCE}],
        },
    )
    assert response.status_code == 422
    assert "secret-token" not in response.text
    monkeypatch.setattr(api, "_predict", broken)
    with TestClient(create_web_app(), raise_server_exceptions=False) as c:
        response = c.post("/api/predict", json={"sequences": [SEQUENCE]})
    assert response.status_code == 500
    assert "secret-token" not in response.text
    assert "Public API failed" in caplog.text
    response = client.post("/api/predict", json={"sequences": ["AAX"]})
    assert response.status_code == 422
    assert "input" not in response.json()["detail"][0]


def test_deployment_bundle_without_source_root_preserves_integrity(
    tmp_path, monkeypatch
):
    (tmp_path / "config").mkdir()
    (tmp_path / "examples/models").mkdir(parents=True)
    for path in ["config/default-model.json", "examples/models/human-baseline.joblib"]:
        shutil.copyfile(ROOT / path, tmp_path / path)
    monkeypatch.setattr(default_model, "source_root", lambda: None)
    monkeypatch.chdir(tmp_path.parent)
    monkeypatch.setenv("MAPEXPLOC_MODEL_PATH", "/not/approved.joblib")
    selected = default_model.resolve_default_model(trusted_root=tmp_path)
    assert selected.load().metadata["model_id"] == MODEL_ID
    assert (
        TestClient(create_web_app(trusted_root=tmp_path)).get("/api/model").status_code
        == 200
    )
    manifest = tmp_path / "config/default-model.json"
    contents = json.loads(manifest.read_text())
    contents["model_id"] = "wrong-model"
    manifest.write_text(json.dumps(contents))
    with pytest.raises(ModelArtifactError, match="identity"):
        default_model.resolve_default_model(trusted_root=tmp_path).load()
    selected.path.write_bytes(b"corrupt")
    with pytest.raises(ModelArtifactError, match="checksum"):
        selected.load()
    c = TestClient(create_web_app(trusted_root=tmp_path))
    assert not c.get("/api/health").json()["model_available"]
    unavailable = c.get("/api/model")
    assert unavailable.status_code == 503
    assert str(tmp_path) not in unavailable.text
    assert "MAPEXPLOC_MODEL_PATH" not in unavailable.text


def test_lazy_runtime_reuse_and_concurrent_explanations(monkeypatch):
    loads = []
    original = default_model.ModelSelection.load

    def load(selection):
        loads.append(selection.path)
        return original(selection)

    monkeypatch.setattr(default_model.ModelSelection, "load", load)
    app = create_web_app()
    assert not loads
    with TestClient(app) as c, ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: c.post("/api/explain", json={"sequences": [SEQUENCE]}).json(),
                range(8),
            )
        )
    assert len(loads) == 1
    assert all(result == results[0] for result in results)


def test_shap_calls_and_expected_values_are_serialized():
    model = default_model.resolve_default_model(trusted_root=ROOT).load().model
    explainer = ShapExplainer(model)
    actual = explainer.explainer.shap_values
    lock = Lock()
    active = 0
    peak = 0

    def observed(features):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        sleep(0.01)
        result = actual(features)
        with lock:
            active -= 1
        return result

    explainer.explainer.shap_values = observed
    features = build_feature_matrix([SEQUENCE])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: explainer.explain_sample(features), range(8)))
    assert peak == 1
    for result in results:
        np.testing.assert_allclose(
            result["expected_value"] + result["shap_values"].sum(axis=2),
            model.predict_proba(features),
            atol=1e-6,
        )


def test_public_report_compression_and_uncompressed_size_guard(client, monkeypatch):
    # Model a serialized complete report large enough to trip the platform cap.
    # No scientific fields should be dropped to make a report fit.
    monkeypatch.setattr(api, "_predict", lambda *_: (["A" * 4_100_000], [], None))
    response = client.post("/api/predict", json={"sequences": [SEQUENCE]})
    assert response.status_code == 200
    assert response.headers["content-encoding"] == "gzip"
    assert len(response.json()["model_classes"][0]) == 4_100_000
    assert int(response.headers["content-length"]) < 4_000_000
    response = client.post(
        "/api/predict",
        json={"sequences": [SEQUENCE]},
        headers={"accept-encoding": "identity"},
    )
    assert response.status_code == 413
    assert "Accept gzip" in response.json()["detail"]
    assert "content-encoding" not in response.headers
    assert len(response.content) < 1000


@pytest.mark.parametrize(
    "sizes, accepted",
    [
        ([MAX_REQUEST_BYTES + 1], False),
        ([1000, MAX_REQUEST_BYTES - 999], False),
        ([1000, MAX_REQUEST_BYTES - 1000], True),
    ],
)
def test_boundary_checks_chunks_before_copying(monkeypatch, sizes, accepted):
    import asyncio

    from mapexploc import web

    copied = []

    class BoundedBuffer(bytearray):
        def extend(self, chunk):
            assert len(self) + len(chunk) <= MAX_REQUEST_BYTES
            copied.append(len(chunk))
            super().extend(chunk)

    monkeypatch.setattr(web, "bytearray", BoundedBuffer, raising=False)
    messages = iter(
        {"type": "http.request", "body": b"A" * size, "more_body": i < len(sizes) - 1}
        for i, size in enumerate(sizes)
    )
    sent = []
    received = []

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    async def app(scope, receive, send):
        received.append(await receive())

    asyncio.run(
        web._PublicBoundary(app)({"type": "http", "method": "POST"}, receive, send)
    )
    if accepted:
        assert received[0]["body"] == b"A" * sum(sizes)
        assert copied == sizes
    else:
        assert not received
        assert sent[0]["status"] == 413
        assert copied == sizes[:-1]
