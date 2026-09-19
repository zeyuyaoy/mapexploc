from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from mapexploc import (
    AdapterDescriptor,
    MethodConfiguration,
    load_report,
    run_analysis,
    write_report,
)
from mapexploc.adapter import adapter_from_artifact
from mapexploc.diagnostics import attribution_agreement
from mapexploc.explainers.references import references
from mapexploc.explainers.regions import RegionGame, _kernel, sequence_regions
from mapexploc.report_v3 import export_original_report


class PositionModel:
    descriptor = AdapterDescriptor(
        model_id="synthetic-position-control",
        classes=("outside", "inside"),
        preprocessing_id="count-first-ten:v1",
    )

    def predict_proba(self, batch):
        p = np.array([0.1 + 0.8 * s[:10].count("L") / 10 for s in batch])
        return np.column_stack([p, 1 - p])


def test_frozen_rf_runtime_measurements_are_not_software_versions():
    model = adapter_from_artifact(
        Path(__file__).resolve().parents[1] / "examples/models/human-baseline.joblib"
    )
    report = run_analysis(
        model,
        [dict(protein_id="p", sequence="ACDEFGHIKLMNPQRSTVWY")],
        {"method_profile": "v2x-legacy"},
    )
    assert "training_seconds" not in report.runtime.software
    assert isinstance(report.runtime.software["numpy"], str)
    assert max(abs(v) for v in report.results[0].residuals) < 1e-6


def test_legacy_and_v3_roundtrip(tmp_path):
    proteins = [dict(protein_id="p", sequence="L" * 10 + "A" * 20)]
    old = run_analysis(PositionModel(), proteins)
    new = run_analysis(PositionModel(), proteins, {"method_profile": "v2x-legacy"})
    assert old.schema_version == 2 and new.schema_version == 3
    assert old.results[0].attributions == new.results[0].attributions
    for report in (old, new):
        paths = write_report(report, tmp_path / str(report.schema_version))
        assert load_report(paths["report.json"]) == report
        if report.schema_version == 3:
            document = paths["report.html"].read_text()
            assert "Methodology and uncertainty" in document
            assert "Attribution stability" in document
        export_original_report(paths["report.json"], tmp_path / "copy.json")
        assert (tmp_path / "copy.json").read_bytes() == paths[
            "report.json"
        ].read_bytes()
    corrupted = new.model_dump()
    corrupted["methodology_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        load_report(corrupted)
    with pytest.raises(ValueError):
        run_analysis(
            PositionModel(), proteins, {"method_profile": "unqualified-default"}
        )


@pytest.mark.parametrize(
    "strategy", ["whole_shuffle", "block_shuffle", "positional_pool"]
)
def test_reference_invariants(strategy):
    s = "ACDEFGHIKLMNPQRSTVWY" * 3
    options = (
        dict(reference_pool=[s, s[::-1]], reference_pool_id="frozen-dev")
        if strategy == "positional_pool"
        else {}
    )
    c = MethodConfiguration(
        method_profile="v2x-development", reference_strategy=strategy, **options
    )
    a = references(s, c)
    assert a == references(s, c)
    assert a == references(s, c.model_copy(update={"references": 16}))[:4]
    assert all(len(r) == len(s) for r in a)
    if strategy == "whole_shuffle":
        assert all(Counter(r) == Counter(s) for r in a)
    if strategy == "block_shuffle":
        assert all(
            Counter(r[i: i + 20]) == Counter(s[i: i + 20])
            for r in a
            for i in range(0, len(s), 20)
        )
    assert references(s, c.model_copy(update={"reference_seed": 43})) != a


def test_development_diagnostics_and_group_stats():
    p = [
        dict(protein_id="a", sequence="L" * 10 + "A" * 20, group="a"),
        dict(protein_id="b", sequence="A" * 10 + "L" * 20, group="b"),
    ]
    c = dict(
        method_profile="v2x-development",
        cohort_id="dev",
        stability_checks=True,
        reference_sensitivity=True,
        faithfulness=True,
        diagnostic_draws=2,
    )
    a = run_analysis(PositionModel(), p, c)
    b = run_analysis(PositionModel(), p[::-1], c)
    assert a.global_statistics == b.global_statistics
    assert a.diagnostics[0].stability
    assert a.diagnostics[0].faithfulness
    assert any(r.get("matched_advantage", 0) > 0 for r in a.diagnostics[0].faithfulness)
    assert max(abs(x) for r in a.results for x in r.residuals) < 1e-5
    assert load_report(a.model_dump_json()) == a
    data = a.model_dump()
    data["global_statistics"][0]["mean"] += 0.1
    with pytest.raises(ValueError, match="Group statistics"):
        load_report(data)


def test_diagnostic_warnings_and_null():
    assert (
        attribution_agreement([[0, 0]], [[0, 0]], ["a"])[0]["status"] == "uninformative"
    )
    a = attribution_agreement([[0.6, 0.1, -0.2]], [[0.1, -0.4, 0.6]], ["a"])[0]
    assert a["status"] == "unstable" and a["sign_agreement"] < 0.9
    assert (
        attribution_agreement(
            [[0.6, 0.1, -0.2]], [[0.1, -0.4, 0.6]], ["a"], methodological=True
        )[0]["status"]
        == "methodological_sensitivity"
    )


def test_resolution_and_rank_guard():
    for strategy in ("legacy", "windows_5", "windows_10"):
        r = sequence_regions(103, strategy)
        assert r[0].start == 0 and r[-1].end == 103
        assert all(a.end == b.start for a, b in zip(r, r[1:]))
    c = MethodConfiguration(
        method_profile="v2x-development",
        region_strategy="windows_5",
        coalition_budget=32,
    )
    game = RegionGame(PositionModel(), "A" * 200, c)
    with pytest.raises(ValueError, match="insufficient"):
        _kernel(game, 42, 32)


def test_exact_interaction_containing_game_and_null_outputs():
    class PairModel(PositionModel):
        def predict_proba(self, batch):
            p = np.asarray([0.1 + 0.8 * (s[0] == "L" and s[10] == "L") for s in batch])
            return np.c_[p, 1 - p]

    game = RegionGame(
        PairModel(),
        "L" * 20 + "A" * 10,
        MethodConfiguration(method_profile="v2x-development"),
    )
    game.references = ["A" * 30] * 4
    phi, base, sampled = _kernel(game, 42, 512)
    np.testing.assert_allclose(base, [0.1, 0.9], atol=1e-12)
    np.testing.assert_allclose(phi, [[0.4, 0.4, 0], [-0.4, -0.4, 0]], atol=1e-12)
    assert sampled == 6
    # The model contains a conjunction. These remain ordinary Shapley values,
    # not an implementation of SHAP interaction values.
    game.sequence = "A" * 30
    game.cache.clear()
    null, _, _ = _kernel(game, 42, 512)
    np.testing.assert_allclose(null, 0, atol=1e-12)
