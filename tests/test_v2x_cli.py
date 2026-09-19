import json

import numpy as np
from typer.testing import CliRunner

from mapexploc import AdapterDescriptor
from mapexploc.cli import app


def test_cli_mode_and_portable_identity_propagation(tmp_path, monkeypatch):
    seen = []

    class Adapter:
        descriptor = AdapterDescriptor(
            model_id="DeepLoc-2.1-Accurate",
            classes=("a", "b"),
            preprocessing_id="test",
            checkpoint_sha256="a" * 64,
            provenance={"mode": "accurate"},
        )

        def predict_proba(self, b):
            return np.tile([0.6, 0.4], (len(b), 1))

    def factory(name, configuration):
        seen.append((name, configuration))
        return Adapter()

    monkeypatch.setattr("mapexploc.adapter.registered_adapter", factory)
    fasta = tmp_path / "p.fasta"
    fasta.write_text(">p\nACDEFGHIKLMNPQRSTVWY\n")
    native = tmp_path / "native.json"
    native.write_text(json.dumps({"mode": "accurate"}))
    spec = tmp_path / "run.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "adapter": "deeploc2",
                "mode": "accurate",
                "expected_model_id": Adapter.descriptor.model_id,
                "expected_checkpoint_sha256": "a" * 64,
                "configuration": {"method_profile": "v2x-legacy"},
            }
        )
    )
    args = [
        "analyze",
        "--adapter",
        "deeploc2",
        "--adapter-config",
        str(native),
        "--fasta",
        str(fasta),
        "--run-spec",
        str(spec),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert seen[0][1]["mode"] == "accurate"
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["runtime"]["model_mode"] == "accurate"
    assert report["schema_version"] == 3
    altered = json.loads(spec.read_text())
    altered["expected_checkpoint_sha256"] = "b" * 64
    spec.write_text(json.dumps(altered))
    failed = CliRunner().invoke(app, args)
    assert failed.exit_code == 1 and "actual model/checkpoint" in failed.output
