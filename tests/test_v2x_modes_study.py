import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from mapexploc import AdapterDescriptor, create_app
from mapexploc.catalog import ConfiguredModel
from mapexploc.deeploc import DeepLocConfiguration
from mapexploc.study import (
    audit_manifests,
    eligible_features,
    holm,
    planned_power,
    required_groups,
)
from mapexploc.workers import deeploc_worker as worker


def test_native_modes_and_threshold_metadata():
    base = dict(python=sys.executable, package_root="/fixture", torch_home="/fixture")
    assert DeepLocConfiguration(**base).mode == "fast"
    accurate = DeepLocConfiguration(
        **base, mode="accurate", prott5_snapshot="/snapshot"
    )
    assert accurate.batch_size == 1
    assert worker.THRESHOLDS["fast"] != worker.THRESHOLDS["accurate"]
    assert len(worker.THRESHOLDS["accurate"]) == len(worker.CLASSES) == 10
    with pytest.raises(ValueError, match="batch_size=1"):
        DeepLocConfiguration(
            **base, mode="accurate", prott5_snapshot="/snapshot", batch_size=2
        )
    with pytest.raises(ValueError, match="snapshot"):
        DeepLocConfiguration(**base, mode="accurate")


def test_hardware_failures_no_fallback(monkeypatch):
    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(worker, "available_memory", lambda: 1024 ** 3)
    with pytest.raises(RuntimeError, match="MPS device is unavailable"):
        worker.check_resources({"mode": "accurate", "device": "mps"}, loading=True)
    with pytest.raises(MemoryError, match="Insufficient available"):
        worker.check_resources({"mode": "accurate", "device": "cpu"}, loading=True)
    monkeypatch.setattr(worker, "available_memory", lambda: 3 * 1024 ** 3)
    worker.check_resources({"mode": "fast", "device": "cpu"}, length=1022)
    with pytest.raises(MemoryError, match="Insufficient available"):
        worker.check_resources(
            {"mode": "fast", "device": "cpu"}, length=1022, batch_size=8
        )


def test_native_asset_identity_is_mode_specific(tmp_path, monkeypatch):
    import hashlib

    paths = [
        "DeepLoc2/model.py",
        "DeepLoc2/deeploc2.py",
        "DeepLoc2/models/models_esm1b/0.ckpt",
        "DeepLoc2/models/models_prott5/0.ckpt",
        "hub/checkpoints/esm1b_t33_650M_UR50S.pt",
        *["snapshot/" + name for name in worker.PROTT5_FILES],
    ]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)

    def checksum(path):
        known = {
            "model.py": (
                "c24e826542eb2741bb056db23847534354900794672ce390ee383c048846cd40"
            ),
            "deeploc2.py": (
                "026fc1c88e70f86fdf74c1870574ceb74832d8d429daf2e320aeab3365622ef5"
            ),
            "pytorch_model.bin": worker.PROTT5_WEIGHTS_SHA256,
        }
        return known.get(path.name, hashlib.sha256(path.read_bytes()).hexdigest())

    monkeypatch.setattr(worker, "checksum", checksum)
    monkeypatch.setattr(worker.importlib.metadata, "version", lambda _: "fixture")
    config = dict(package_root=str(tmp_path), torch_home=str(tmp_path))
    fast = worker.inspect(config)
    accurate_config = dict(
        **config,
        mode="accurate",
        prott5_snapshot=str(tmp_path / "snapshot"),
        prott5_revision=worker.PROTT5_REVISION,
        batch_size=1,
    )
    accurate = worker.inspect(accurate_config)
    assert fast["checkpoint_sha256"] != accurate["checkpoint_sha256"]
    assert fast["max_length"] == 1022 and accurate["max_length"] == 4000
    assert "ProtT5" in accurate["preprocessing_id"]
    (tmp_path / "snapshot/spiece.model").write_text("changed-tokenizer")
    assert worker.inspect(config)["checkpoint_sha256"] == fast["checkpoint_sha256"]
    assert (
        worker.inspect(accurate_config)["checkpoint_sha256"]
        != accurate["checkpoint_sha256"]
    )
    with pytest.raises(ValueError, match="fingerprint differs"):
        worker.inspect(
            {**accurate_config, "expected_checkpoint_sha256": fast["checkpoint_sha256"]}
        )


def test_catalog_does_not_load_and_reports_missing_assets(monkeypatch):
    monkeypatch.setattr(
        "mapexploc.catalog.registered_adapter",
        lambda *a: pytest.fail("Catalogue must not load"),
    )
    c = ConfiguredModel(
        "deeploc2",
        dict(
            python="/missing",
            package_root="/missing",
            torch_home="/missing",
            mode="accurate",
            prott5_snapshot="/missing",
        ),
    )
    assert c.summary("accurate")["readiness"] == "unavailable"
    assert c.summary("accurate")["mode"] == "accurate"


def test_resident_mode_does_not_require_a_second_cold_load(tmp_path, monkeypatch):
    model = ConfiguredModel(
        "deeploc2",
        {
            "python": sys.executable,
            "package_root": str(tmp_path),
            "torch_home": str(tmp_path),
        },
    )
    model.adapter = SimpleNamespace(
        is_resident=True,
        descriptor=AdapterDescriptor(
            model_id="resident",
            classes=("a", "b"),
            preprocessing_id="test",
        ),
    )
    monkeypatch.setattr(
        "mapexploc.catalog.subprocess.run",
        lambda *a, **k: pytest.fail("Resident worker must not need a second cold load"),
    )
    assert model.summary("fast")["readiness"] == "configured"


def test_api_configured_modes_propagate_and_reject_identity(monkeypatch):
    import numpy as np

    class Model:
        def __init__(self, c):
            self.descriptor = AdapterDescriptor(
                model_id="DeepLoc-2.1-" + c["mode"].title(),
                classes=("a", "b"),
                preprocessing_id=c["mode"],
                checkpoint_sha256=("a" if c["mode"] == "fast" else "b") * 64,
                provenance={"mode": c["mode"]},
            )

        def predict_proba(self, b):
            return np.tile([0.8, 0.2], (len(b), 1))

    monkeypatch.setattr(
        "mapexploc.catalog.registered_adapter", lambda name, c: Model(c)
    )
    base = dict(python=sys.executable, package_root="/tmp", torch_home="/tmp")
    configs = {
        m: dict(
            factory="deeploc2",
            configuration={
                **base,
                "mode": m,
                **({"prott5_snapshot": "/tmp"} if m == "accurate" else {}),
            },
        )
        for m in ("fast", "accurate")
    }
    client = TestClient(create_app(adapter_configurations=configs))
    assert len(client.get("/v3/models").json()["models"]) == 2
    for mode in ("fast", "accurate"):
        body = dict(adapter_id=mode, proteins=[dict(protein_id="p", sequence="A" * 10)])
        r = client.post("/v3/predict", json=body)
        assert r.status_code == 200
        assert r.json()["model"]["provenance"]["mode"] == mode
        assert (
            client.post(
                "/v3/predict", json={**body, "expected_model_id": "wrong"}
            ).status_code
            == 409
        )


def test_feature_specific_evidence_and_power():
    feature = dict(
        type="Signal",
        location={"start": {"value": 1}, "end": {"value": 20}},
        evidences=[{"evidenceCode": "ECO:0000269"}],
    )
    record = dict(
        primaryAccession="P",
        sequence={"value": "A" * 100},
        features=[feature],
        comments=[
            {
                "commentType": "SUBCELLULAR LOCATION",
                "subcellularLocations": [{"location": {"value": "Secreted"}}],
            }
        ],
    )
    assert "secretion_signal" in eligible_features(record)
    feature["evidences"] = []
    assert not eligible_features(record)
    assert required_groups() >= 60 and planned_power(required_groups()) >= 0.8
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    members = [
        dict(protein_id="a", group="same", split="development", sequence_sha256="x"),
        dict(protein_id="b", group="same", split="evaluation", sequence_sha256="y"),
    ]
    with pytest.raises(ValueError, match="Cross-split"):
        audit_manifests([{"members": members}], {})


def test_api_shutdown_closes_owned_native_worker(monkeypatch):
    import numpy as np

    class Model:
        descriptor = AdapterDescriptor(
            model_id="owned", classes=("a", "b"), preprocessing_id="fixture"
        )
        closed = False

        def predict_proba(self, batch):
            return np.tile([0.7, 0.3], (len(batch), 1))

        def close(self):
            self.closed = True

    model = Model()
    monkeypatch.setattr("mapexploc.catalog.registered_adapter", lambda *a: model)
    app = create_app(
        adapter_configurations={"owned": {"factory": "fixture", "configuration": {}}}
    )
    with TestClient(app) as client:
        response = client.post(
            "/v3/predict",
            json={
                "adapter_id": "owned",
                "proteins": [{"protein_id": "p", "sequence": "A" * 10}],
            },
        )
        assert response.status_code == 200
        assert not model.closed
    assert model.closed


def test_failed_mode_switch_does_not_leave_an_unloaded_worker_resident(monkeypatch):
    import threading

    from mapexploc import deeploc

    previous = SimpleNamespace(process=SimpleNamespace(poll=lambda: None))
    unloaded = []
    previous._exchange = unloaded.append
    monkeypatch.setattr(deeploc, "_ACTIVE", lambda: previous)
    target = deeploc.DeepLocAdapter.__new__(deeploc.DeepLocAdapter)
    target.lock = threading.Lock()
    target.descriptor = AdapterDescriptor(
        model_id="failing", classes=("a", "b"), preprocessing_id="fixture"
    )

    def fail(_):
        raise RuntimeError("native memory preflight failed")

    target._exchange = fail
    with pytest.raises(RuntimeError, match="preflight"):
        target.predict_proba(["A" * 10])
    assert unloaded == [{"operation": "unload"}]
    assert deeploc._ACTIVE is None
