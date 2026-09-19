import sqlite3

import numpy as np
import pytest

from mapexploc import AdapterDescriptor, ExecutionOptions, run_analysis
from mapexploc.execution import CachedAdapter


class Model:
    def __init__(self, mode="fast"):
        self.calls = 0
        self.descriptor = AdapterDescriptor(
            model_id="fixture:" + mode,
            classes=("a", "b"),
            preprocessing_id="v1",
            checkpoint_sha256=("a" if mode == "fast" else "b") * 64,
            provenance={"mode": mode},
        )

    def predict_proba(self, batch):
        self.calls += len(batch)
        p = np.array([0.1 + 0.8 * s[:10].count("L") / 10 for s in batch])
        return np.c_[p, 1 - p]


def test_cache_isolation_integrity_bounds(tmp_path):
    a, b = Model(), Model("accurate")
    x = CachedAdapter(a, tmp_path, 2)
    y = CachedAdapter(b, tmp_path, 2)
    np.testing.assert_equal(x.predict_proba(["L" * 10]), x.predict_proba(["L" * 10]))
    assert a.calls == 1 and x.cache_hits == 1
    y.predict_proba(["L" * 10])
    assert b.calls == 1
    x.predict_proba(["A" * 10])
    with sqlite3.connect(x.path) as db:
        assert db.execute("SELECT COUNT(*) FROM probabilities").fetchone()[0] == 2
        db.execute("UPDATE probabilities SET payload='[0.9,0.1]'")
    with pytest.raises(ValueError, match="Corrupted"):
        x.predict_proba(["A" * 10])


def test_resume_rejects_changed_method_and_reassembles(tmp_path, monkeypatch):
    m = Model()
    events = []
    options = ExecutionOptions(
        cache_directory=tmp_path / "cache",
        restart_directory=tmp_path / "restart",
        progress=events.append,
    )
    proteins = [
        dict(protein_id="a", sequence="L" * 10 + "A" * 20),
        dict(protein_id="b", sequence="A" * 10 + "L" * 20),
    ]
    c = dict(method_profile="v2x-legacy", cohort_id="cohort")
    a = run_analysis(m, proteins, c, execution=options)
    calls = m.calls
    b = run_analysis(m, proteins, c, execution=options)
    assert a.results == b.results and m.calls == calls
    assert events[-1]["resumed"]
    with pytest.raises(ValueError, match="Restart manifest"):
        run_analysis(m, proteins, {**c, "seed": 43}, execution=options)
    monkeypatch.setattr("mapexploc.analysis.explanation_implementation", lambda: "new")
    with pytest.raises(ValueError, match="Restart manifest"):
        run_analysis(m, proteins, c, execution=options)
