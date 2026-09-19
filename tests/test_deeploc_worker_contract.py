import sys

import numpy as np
import pytest
import subprocess

from mapexploc.deeploc import DeepLocAdapter, DeepLocConfiguration


def test_persistent_worker_contract_and_cleanup(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    descriptor = dict(
        model_id="native-contract-fixture",
        classes=["a", "b"],
        task_type="multilabel",
        thresholds=[0.5, 0.6],
        preprocessing_id="fixture",
        min_length=10,
        max_length=1022,
    )
    worker.write_text(
        "import sys,json\njson.loads(sys.stdin.readline())\n"
        + f'print(json.dumps({{"descriptor":{descriptor!r}}}),flush=True)\n'
        + "for line in sys.stdin:\n r=json.loads(line)\n"
        + ' print(json.dumps({"probabilities":[[.9,.8]]*len(r["sequences"])}),'
        + "flush=True)\n"
    )
    original = subprocess.Popen
    launched = []

    def launch(command, **kwargs):
        process = original([sys.executable, "-u", str(worker)], **kwargs)
        launched.append(process)
        return process

    monkeypatch.setattr("mapexploc.deeploc.subprocess.Popen", launch)
    config = DeepLocConfiguration(
        python=sys.executable, package_root=str(tmp_path), torch_home=str(tmp_path)
    )
    with DeepLocAdapter(config) as adapter:
        np.testing.assert_equal(
            adapter.predict_proba(["A" * 10, "C" * 12]), [[0.9, 0.8], [0.9, 0.8]]
        )
        np.testing.assert_equal(adapter.predict_proba(["L" * 10]), [[0.9, 0.8]])
        with pytest.raises(ValueError, match="never silently truncated"):
            adapter.predict_proba(["A" * 1023])
        assert len(launched) == 1
    assert launched[0].poll() is not None


def test_native_source_version_mismatch_fails_before_loading(tmp_path):
    (tmp_path / "DeepLoc2").mkdir()
    (tmp_path / "DeepLoc2/model.py").write_text("unknown-version")
    with pytest.raises(RuntimeError, match="exited before returning"):
        DeepLocAdapter(
            DeepLocConfiguration(
                python=sys.executable,
                package_root=str(tmp_path),
                torch_home=str(tmp_path),
                timeout_seconds=5,
            )
        )
