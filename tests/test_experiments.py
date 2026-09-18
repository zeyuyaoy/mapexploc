"""Offline protocol checks: group leakage, deterministic selection and gates."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mapexploc.experiments import (
    CLASSES,
    _commit_stage,
    _completed,
    assign_partitions,
    bootstrap_comparison,
    candidate_configurations,
    promotion_decision,
    ranking_key,
    sequence_id,
)


def test_changed_sequences_have_distinct_nodes() -> None:
    assert sequence_id("P1", "AAA") != sequence_id("P1", "CCC")
    assert sequence_id("P1", "AAA") == sequence_id("P1", "AAA")


def test_historical_accessions_and_groups_cannot_enter_training() -> None:
    old = pd.DataFrame(
        [
            {"accession": "old", "group": "g1", "split": "test"},
            {"accession": "trained", "group": "g4", "split": "train"},
        ]
    )
    fresh = pd.DataFrame(
        [
            {"accession": "old", "group": "changed", "label": "Nucleus"},
            {"accession": "related", "group": "g1", "label": "Nucleus"},
            {"accession": "new-relative", "group": "changed", "label": "Nucleus"},
            {"accession": "new", "group": "g2", "label": "Nucleus"},
        ]
    )
    split, report = assign_partitions(fresh, old)
    assert list(split.split) == ["excluded_historical"] * 3 + ["development"]
    assert report["status"] == "unavailable"
    assert report["minimum_count_shortfalls"]["Cytoplasm"] == 20


def test_confirmation_is_deterministic_and_isolated() -> None:
    fresh = pd.DataFrame(
        [
            {"accession": f"{label}-{i}", "group": f"{label}-{i}", "label": label}
            for label in CLASSES
            for i in range(150)
        ]
    )
    old = pd.DataFrame([{"accession": "old", "group": "old", "split": "train"}])
    first, report = assign_partitions(fresh, old)
    second, again = assign_partitions(fresh, old)
    pd.testing.assert_frame_equal(first, second)
    assert report == again and report["status"] == "available"
    assert min(report["confirmation_counts"].values()) >= 20
    assert not set(first.loc[first.split == "development", "group"]) & set(
        first.loc[first.split == "confirmation", "group"]
    )


def test_search_design_and_tiebreaks() -> None:
    first = candidate_configurations()
    assert first == candidate_configurations()
    assert len(first) == 49
    assert sum(c["family"] == "extra_trees" for c in first) == 24
    assert sum(c["reference"] for c in first) == 1
    assert len({str(c) for c in first}) == 49
    one = {
        "mean_macro_f1": 0.6,
        "mean_log_loss": 1.0,
        "mean_model_bytes": 100.0,
        "candidate": {"family": "a"},
    }
    two = {**one, "mean_log_loss": 0.9}
    assert ranking_key(two) < ranking_key(one)
    assert ranking_key({**one, "mean_macro_f1": 0.61}) < ranking_key(two)
    assert ranking_key({**one, "mean_model_bytes": 99}) < ranking_key(one)
    assert ranking_key(one) < ranking_key({**one, "candidate": {"family": "b"}})
    from mapexploc.estimators import final_estimator
    from mapexploc.experiments import make_estimator

    for candidate in first:
        estimator = final_estimator(make_estimator(candidate))
        assert estimator.n_jobs == 1
        assert estimator.bootstrap == (candidate["family"] == "random_forest")


def passing_report() -> dict:
    old = {
        "macro_f1": 0.5,
        "log_loss": 1.0,
        "classification_report": {c: {"f1-score": 0.5} for c in CLASSES},
    }
    new = {**deepcopy(old), "macro_f1": 0.55, "log_loss": 0.9}
    return {
        "evaluation_status": "independent_confirmation",
        "bootstrap": {"macro_f1_difference_95_interval": [0.01, 0.09]},
        "evaluation": new,
        "reference_evaluation": old,
        "runtime": {
            m: {
                k: 10
                for k in ("prediction_one_ms", "prediction_batch_ms", "shap_one_ms")
            }
            for m in ("candidate", "reference")
        },
        "validation": {
            k: True for k in ("reload", "schema", "class_order", "shap_additivity")
        },
    }


def test_all_promotion_gates_are_required() -> None:
    report = passing_report()
    assert promotion_decision(report)["eligible"]
    for mutation in (
        lambda r: r.update(evaluation_status="historical_diagnostic"),
        lambda r: r.update(bootstrap={}),
        lambda r: r["bootstrap"].update(macro_f1_difference_95_interval=[-0.03, 0.1]),
        lambda r: r["evaluation"].update(macro_f1=0.51),
        lambda r: r["evaluation"].update(log_loss=1.1),
        lambda r: r["evaluation"]["classification_report"]["Nucleus"].update(
            {"f1-score": 0.4}
        ),
        lambda r: r["runtime"]["candidate"].update(shap_one_ms=21),
        lambda r: r["validation"].update(reload=False),
    ):
        changed = deepcopy(report)
        mutation(changed)
        assert not promotion_decision(changed)["eligible"]


def test_completed_stage_cannot_be_silently_changed(tmp_path: Path) -> None:
    (tmp_path / "output").write_text("original")
    _commit_stage(tmp_path, "test", ["output"])
    assert _completed(tmp_path, "test")
    (tmp_path / "output").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        _completed(tmp_path, "test")


def test_bootstrap_pairs_groups_and_keeps_all_classes() -> None:
    labels = np.array(CLASSES * 2)
    groups = np.array([str(i) for i in range(len(labels))])
    first = bootstrap_comparison(labels, labels, labels, groups, repetitions=50)
    assert first["macro_f1_difference_95_interval"] == [0.0, 0.0]
    assert first == bootstrap_comparison(labels, labels, labels, groups, repetitions=50)


def test_offline_experiment_runs_resumes_and_rejects_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tiny synthetic workflow fixture; its scores are not biological evidence."""
    import json

    from mapexploc import experiments as exp
    from mapexploc.artifacts import save_model_artifact
    from mapexploc.baseline import checksum
    from mapexploc.features import build_feature_matrix

    root, run = tmp_path / "reference", tmp_path / "run"
    (root / "examples/baseline").mkdir(parents=True)
    (root / "examples/models").mkdir()
    run.mkdir()
    entries, rows = [], []
    location = {"Membrane": "Cell membrane"}
    for label_index, label in enumerate(CLASSES):
        for i in range(30):
            accession = f"P{label_index:02d}{i:03d}"
            sequence = "M" + "ACDEK"[label_index] * (i + 2) + "FGH"
            entries.append(
                {
                    "entryType": "UniProtKB reviewed (Swiss-Prot)",
                    "primaryAccession": accession,
                    "organism": {"taxonId": 9606},
                    "sequence": {"value": sequence},
                    "comments": [
                        {
                            "commentType": "SUBCELLULAR LOCATION",
                            "subcellularLocations": [
                                {
                                    "location": {
                                        "value": location.get(label, label),
                                        "evidences": [{"evidenceCode": "ECO:0000269"}],
                                    }
                                }
                            ],
                        }
                    ],
                }
            )
            if i < 20:
                rows.append(
                    {
                        "accession": accession,
                        "sequence": sequence,
                        "label": label,
                        "group": accession,
                        "split": "test" if i < 5 else "train",
                    }
                )
    frame = pd.DataFrame(rows)
    path = root / "examples/baseline/dataset.csv"
    frame.to_csv(path, index=False)
    exp.atomic_json(
        root / "examples/baseline/manifest.json",
        {"dataset_sha256": checksum(path), "attribution": "synthetic offline fixture"},
    )
    tiny = [
        {"family": f, "params": {"n_estimators": 2, "max_depth": 3}, "reference": False}
        for f in ("random_forest", "extra_trees")
    ]
    model = exp.make_estimator(tiny[0]).fit(
        build_feature_matrix(frame.sequence), frame.label
    )
    save_model_artifact(
        model,
        root / "examples/models/human-baseline.joblib",
        metadata={"model_id": "fixture"},
    )
    exp.atomic_json(run / "source.json", {"results": entries})
    (run / "source.headers").write_text("x-uniprot-release: fixture\n")

    def fake_search(query: Path, target: Path, output: Path, threads: int) -> list:
        output.write_text("")
        return []

    monkeypatch.setattr(exp, "search_similar", fake_search)
    monkeypatch.setattr(exp, "candidate_configurations", lambda: tiny)
    prepared = exp.prepare_experiment(run, root)
    assert prepared["confirmation"]["status"] == "unavailable"
    assert exp.prepare_experiment(run, root) == prepared
    trained = exp.train_experiment(run, jobs=1)
    assert trained["fold_count"] == 18
    assert exp.train_experiment(run, jobs=1) == trained
    # An interruption after fold completion reuses those fits, then fits the winner.
    (run / "train.complete.json").unlink()
    original_make = exp.make_estimator
    calls = []

    def count_estimator(candidate: dict, seed: int = 42):
        calls.append(candidate)
        return original_make(candidate, seed)

    monkeypatch.setattr(exp, "make_estimator", count_estimator)
    resumed = exp.train_experiment(run, jobs=1)
    assert len(calls) == 1
    assert resumed["winner"]["candidate"] == trained["winner"]["candidate"]
    evaluated = exp.evaluate_experiment(run)
    assert evaluated["evaluation_status"] == "historical_diagnostic"
    assert exp.evaluate_experiment(run) == evaluated
    assert exp.promote_experiment(run, root)["status"] == "retained_version_1"
    with pytest.raises(ValueError, match="changed"):
        exp.prepare_experiment(run, root, cap=999)
    # Changed dependencies prevent cache reuse before any model work occurs.
    monkeypatch.setattr(exp, "software", lambda: {"python": "changed"})
    with pytest.raises(ValueError, match="software"):
        exp.train_experiment(run)
    assert json.loads((run / "promotion.json").read_text())["eligible"] is False


def test_promotion_publishes_checked_manifest_and_retains_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic passing gates exercise publication, not biological performance."""
    from mapexploc import experiments as exp
    from mapexploc.artifacts import save_model_artifact
    from mapexploc.baseline import checksum
    from mapexploc.default_model import manifest_selection
    from mapexploc.features import build_feature_matrix

    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "examples/models").mkdir(parents=True)
    features = build_feature_matrix(["AAAA", "CCCC", "DDDD", "EEEE", "FFFF"])
    model = exp.make_estimator(
        {"family": "extra_trees", "params": {"n_estimators": 2}}
    ).fit(features, CLASSES)
    old = tmp_path / "examples/models/reference.joblib"
    save_model_artifact(model, old, metadata={"model_id": "old"})
    manifest = {
        "schema_version": 1,
        "artifact_path": str(old.relative_to(tmp_path)),
        "model_id": "old",
        "sha256": checksum(old),
    }
    exp.atomic_json(tmp_path / "config/default-model.json", manifest)
    candidate = run / "candidate.joblib"
    save_model_artifact(model, candidate, metadata={"model_id": "new"})
    report = {
        **passing_report(),
        "model_id": "new",
        "artifact_sha256": checksum(candidate),
    }
    exp.atomic_json(run / "evaluation.json", report)
    # Stage checksum logic is tested independently above.
    monkeypatch.setattr(exp, "_validate_run", lambda _: {})
    monkeypatch.setattr(exp, "_completed", lambda *args: True)
    outcome = exp.promote_experiment(run, tmp_path)
    assert outcome["status"] == "promoted"
    assert manifest_selection(tmp_path).load().metadata["model_id"] == "new"
    backup = tmp_path / f"config/default-model.previous-{manifest['sha256']}.json"
    assert exp.read_json(backup) == manifest
    assert checksum(old) == manifest["sha256"]
    assert exp.promote_experiment(run, tmp_path) == outcome
    assert exp.read_json(backup) == manifest
