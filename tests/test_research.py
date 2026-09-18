"""Scientific safeguards; synthetic fixtures never establish biological accuracy."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mapexploc.baseline import CLASSES
from mapexploc.evaluation import classification_metrics
from mapexploc.preprocessing import _clean_and_primary
from mapexploc.research import (
    candidates,
    choose,
    fit_temperature,
    group_bootstrap,
    load_development,
    make_model,
    read_record,
    research_features,
    score,
    study_groups,
    temperature_scale,
    write_record,
)
from mapexploc.validation import grouped_splits


def test_probability_metrics_have_fixed_labels_and_aligned_columns() -> None:
    p = np.array([[0.9, 0.1], [0.1, 0.9]])
    first = classification_metrics(["a", "b"], ["a", "b"], p, ["a", "b"])
    second = classification_metrics(["a", "b"], ["a", "b"], p[:, ::-1], ["b", "a"])
    assert first["log_loss"] == pytest.approx(-np.log(0.9))
    assert second["log_loss"] == first["log_loss"]
    missing = classification_metrics(["a"], ["a"], p[:1], ["a", "b"])
    assert missing["macro_f1"] == 0.5
    assert missing["balanced_accuracy"] == 0.5
    assert missing["macro_auroc"] is None
    for bad in (
        np.full((2, 2), 0.2),
        p * np.nan,
        p[:, :1],
        np.array([[1.1, -0.1]] * 2),
    ):
        with pytest.raises(ValueError):
            classification_metrics(["a", "b"], ["a", "b"], bad, ["a", "b"])


def test_strict_dat_requires_explicit_experimental_evidence() -> None:
    assert _clean_and_primary(["SUBCELLULAR LOCATION: Nucleus."]) == "Other"
    assert (
        _clean_and_primary(["SUBCELLULAR LOCATION: Nucleus {ECO:0000269}."])
        == "Nucleus"
    )


def test_grouped_nested_folds_isolate_outer_groups() -> None:
    y = np.tile(CLASSES, 30)
    groups = np.repeat(np.arange(75), 2)
    for fit, valid in grouped_splits(y, groups, 3, 42):
        for inner_fit, inner_valid in grouped_splits(y[fit], groups[fit], 3, 43):
            assert not set(groups[fit][inner_fit]) & set(groups[fit][inner_valid])
            assert not set(groups[valid]) & set(groups[fit][inner_valid])
    with pytest.raises(ValueError, match="independent groups"):
        grouped_splits(y, np.zeros(len(y)), 3, 42)


def test_temperature_is_learned_without_changing_decisions() -> None:
    y = np.asarray(CLASSES * 10)
    p = np.tile(np.eye(5) * 0.4 + 0.12, (10, 1))
    t = fit_temperature(y, p)
    calibrated = temperature_scale(p, t)
    assert 0.5 <= t <= 3
    np.testing.assert_allclose(calibrated.sum(axis=1), 1)
    np.testing.assert_array_equal(calibrated.argmax(axis=1), p.argmax(axis=1))
    assert score(y, calibrated)["log_loss"] < score(y, p)["log_loss"]


def test_features_are_position_sensitive_without_changing_service_schema() -> None:
    features = research_features(["A" * 50 + "K" * 50, "K" * 50 + "A" * 50])
    assert features.shape == (2, 463)
    assert features.n50_A.tolist() == [1, 0]
    assert features.c50_A.tolist() == [0, 1]
    model = make_model(candidates()[2], 42)
    x = pd.concat([features] * 10, ignore_index=True)
    model.fit(x, np.tile(["a", "b"], 10))
    assert model.named_steps["classifier"].n_features_in_ == 23


def test_parsimony_rule_uses_only_inner_scores() -> None:
    p = np.eye(5) * 0.8 + 0.04
    metrics = score(CLASSES, p)
    complex_model = {"candidate": candidates()[-1], "metrics": metrics}
    simple_model = {"candidate": candidates()[2], "metrics": metrics}
    assert choose([complex_model, simple_model]) == simple_model


def test_cluster_bootstrap_keeps_repeated_rows_paired() -> None:
    y = np.tile(CLASSES, 2)
    p = np.tile(np.eye(5) * 0.8 + 0.04, (2, 1))
    groups = np.tile(np.arange(5), 2)
    r = group_bootstrap(y, p, p, groups, repetitions=30)
    assert r["groups"] == 5
    assert r["macro_f1_difference_97_5_interval"] == [0, 0]
    assert r["log_loss_difference_97_5_interval"] == [0, 0]


def test_reserved_groups_and_modified_results_rejected(tmp_path: Path) -> None:
    rows = [
        {
            "accession": f"A{i}",
            "sequence": "A" * (i + 1),
            "label": c,
            "group": str(i),
            "split": "development",
            "evidence": "[]",
            "localization_annotations": "[]",
        }
        for i, c in enumerate(CLASSES)
    ]
    rows.append(
        {
            **rows[0],
            "accession": "reserved",
            "sequence": "KKKK",
            "split": "excluded_historical",
        }
    )
    path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="overlap"):
        load_development(path)
    output = tmp_path / "output.json"
    write_record(output, {"metric": 0.5})
    assert read_record(output)["metric"] == 0.5
    output.write_text('{"metric": 1.0}')
    with pytest.raises(ValueError, match="Changed"):
        read_record(output)


def test_publication_groups_include_transitive_sequence_links() -> None:
    frame = pd.DataFrame(
        {
            "group": ["a", "a", "b", "c"],
            "evidence": [
                "[]",
                json.dumps([{"source": "PubMed", "id": "1"}]),
                json.dumps([{"source": "PubMed", "id": "1"}]),
                "[]",
            ],
        }
    )
    groups = study_groups(frame)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_nested_workflow_reloads_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mapexploc import research

    rng = np.random.default_rng(7)
    rows = []
    for i, label in enumerate(CLASSES * 18):
        rows.append(
            {
                "accession": f"fixture{i:03}",
                "sequence": "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 60)),
                "label": label,
                "group": f"g{i}",
                "split": "development",
                "evidence": "[]",
                "localization_annotations": "[]",
            }
        )
    path = tmp_path / "fixture.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    configs = candidates()
    small = [configs[0], configs[2], configs[-2]]
    small[-1]["params"]["n_estimators"] = 2
    monkeypatch.setattr(research, "candidates", lambda: small)
    output = tmp_path / "run"
    result = research.run_research(path, output, jobs=1, seeds=(20260918,))
    assert research.run_research(path, output, jobs=1, seeds=(20260918,)) == result
    assert not result["promotion"]
    predictions = research.predict_research(
        output / "research-model.joblib", [rows[0]["sequence"]]
    )
    np.testing.assert_allclose(predictions.sum(axis=1), 1)
    saved = pd.read_csv(output / "predictions.csv")
    assert saved.groupby("method").accession.nunique().eq(90).all()
    (output / "results.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        research.run_research(path, output, jobs=1, seeds=(20260918,))


def test_generic_training_uses_groups_and_scales_before_optional_smote() -> None:
    from mapexploc.models.knn import train_knn
    from mapexploc.models.rf import train_random_forest

    x = pd.DataFrame({"length": np.arange(60) + 1.0, "value": np.tile([0, 1, 2], 20)})
    y = pd.Series(np.tile(["a", "b", "c"], 20))
    groups = pd.Series(np.repeat(np.arange(30), 2))
    rf = train_random_forest(
        x, y, {"rf__n_estimators": [3]}, groups=groups, n_jobs=1, use_smote=True
    )
    knn = train_knn(
        x, y, param_grid={"knn__n_neighbors": [1]}, groups=groups, n_jobs=1, cv=3
    )
    for search in (rf["search"], knn["grid_search"]):
        for train, valid in search.cv:
            assert not set(groups.iloc[train]) & set(groups.iloc[valid])
    assert [name for name, _ in rf["model"].steps][:2] == ["scaler", "smote"]


def test_artifact_rejects_reordered_class_metadata(tmp_path: Path) -> None:
    import joblib
    from sklearn.dummy import DummyClassifier

    from mapexploc.artifacts import (
        ModelArtifactError,
        load_model_artifact,
        save_model_artifact,
    )
    from mapexploc.features import FEATURE_NAMES

    model = DummyClassifier().fit(np.zeros((2, len(FEATURE_NAMES))), ["a", "b"])
    path = tmp_path / "model.joblib"
    save_model_artifact(model, path)
    payload = joblib.load(path)
    payload["classes"] = ["b", "a"]
    joblib.dump(payload, path)
    with pytest.raises(ModelArtifactError, match="class order"):
        load_model_artifact(path)
