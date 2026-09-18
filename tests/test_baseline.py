"""Offline tests for evidence policy, split integrity and evaluation semantics."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mapexploc.baseline import (
    CLASSES,
    checksum,
    classification_metrics,
    curate_entry,
    curate_snapshot,
    grouped_split,
    similarity_groups,
    train_baseline,
    write_json,
)


def entry(location: str = "Nucleus", evidence: str = "ECO:0000269") -> dict[str, Any]:
    return {
        "entryType": "UniProtKB reviewed (Swiss-Prot)",
        "primaryAccession": "P00001",
        "organism": {"taxonId": 9606},
        "sequence": {"value": "MKKAAAA"},
        "comments": [
            {
                "commentType": "SUBCELLULAR LOCATION",
                "subcellularLocations": [
                    {
                        "location": {
                            "value": location,
                            "evidences": [{"evidenceCode": evidence}],
                        }
                    }
                ],
            }
        ],
    }


def test_evidence_and_conflict_policy() -> None:
    accepted, _ = curate_entry(entry())
    assert accepted is not None and accepted["label"] == "Nucleus"
    assert "ECO:0000269" in accepted["evidence"]
    assert (
        curate_entry(entry(evidence="ECO:0000250"))[1]
        == "no_experimental_location_evidence"
    )
    conflicting = entry()
    conflicting["comments"][0]["subcellularLocations"].append(
        {"location": {"value": "Cytoplasm"}}
    )
    assert curate_entry(conflicting)[1] == "conflicting_locations"
    isoform = entry()
    isoform["comments"][0]["molecule"] = "Isoform 2"
    assert curate_entry(isoform)[1] == "isoform_or_product_specific"
    assert curate_entry(entry("Membrane"))[1] == "unmapped_location"
    assert curate_entry(entry("Cell membrane"))[0]["label"] == "Membrane"  # type: ignore[index]
    invalid = entry()
    invalid["sequence"]["value"] = "MXX"
    assert curate_entry(invalid)[1] == "invalid_sequence"


def test_duplicate_conflicts_are_excluded(tmp_path: Path) -> None:
    first = entry()
    second = entry("Cytoplasm")
    second["primaryAccession"] = "P00002"
    third = entry()
    third["primaryAccession"] = "P00003"
    third["sequence"]["value"] = "MCCC"
    source = tmp_path / "source.json"
    write_json(source, {"results": [first, second, third]})
    frame, reasons = curate_snapshot(source)
    assert list(frame["accession"]) == ["P00003"]
    assert reasons["conflicting_duplicate_sequences"] == 2


def split_frame() -> pd.DataFrame:
    rows = []
    letters = "ACDEFGHIKLMNPQRSTVWY"
    for label_index, label in enumerate(CLASSES):
        for i in range(60):
            accession = f"{label_index}-{i}"
            rows.append(
                {
                    "accession": accession,
                    "label": label,
                    "group": f"{label_index}-{i // 2}",
                    "sequence": "M"
                    + letters[label_index] * 10
                    + letters[i // 20]
                    + letters[i % 20],
                }
            )
    return pd.DataFrame(rows)


def test_grouped_split_is_reproducible_and_disjoint() -> None:
    frame = split_frame()
    train, test = grouped_split(frame)
    other_train, other_test = grouped_split(frame)
    np.testing.assert_array_equal(train, other_train)
    np.testing.assert_array_equal(test, other_test)
    assert not set(frame.iloc[train]["group"]) & set(frame.iloc[test]["group"])
    assert frame.iloc[test]["label"].value_counts().min() >= 10
    groups = similarity_groups(["a", "b", "c", "d"], [("b", "c"), ("a", "b")])
    assert groups[:3] == ["a", "a", "a"]
    assert groups[3] == "d"
    with pytest.raises(ValueError):
        grouped_split(frame.iloc[:20])


def test_probability_metrics_have_known_values() -> None:
    report = classification_metrics(
        ["A", "B"], ["A", "B"], np.array([[1.0, 0.0], [0.0, 1.0]]), ["A", "B"]
    )
    assert report["macro_f1"] == 1
    assert report["multiclass_brier"] == 0
    assert report["top_label_ece_10_bins"] == 0


def test_baseline_trains_only_training_partition(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from sklearn.model_selection import GridSearchCV

    import mapexploc.baseline as baseline

    frame = split_frame()
    train, test = grouped_split(frame)
    frame["split"] = "train"
    frame.loc[test, "split"] = "test"
    frame.to_csv(tmp_path / "dataset.csv", index=False)
    manifest = {
        "seed": 42,
        "release": "test",
        "split": {"cross_partition_hits": 0},
        "dataset_sha256": checksum(tmp_path / "dataset.csv"),
    }
    write_json(tmp_path / "manifest.json", manifest)
    seen = []

    class SmallSearch(GridSearchCV):
        def fit(self, X: Any, y: Any = None, **params: Any) -> Any:
            seen.append(len(X))
            self.param_grid = {"rf__n_estimators": [4], "rf__max_depth": [3]}
            return super().fit(X, y, **params)

    monkeypatch.setattr(baseline, "GridSearchCV", SmallSearch)
    report = train_baseline(tmp_path, tmp_path / "model.joblib", jobs=1)
    assert report["predictions_sha256"] == checksum(tmp_path / "model.predictions.csv")
    assert json.loads((tmp_path / "model.report.json").read_text()) == report
    assert seen == [len(train)]
    assert report["sample_count"] == len(train)
    assert sum(
        report["evaluation"]["classification_report"][label]["support"]
        for label in CLASSES
    ) == len(test)
    assert json.loads((tmp_path / "model.report.json").read_text())[
        "artifact_sha256"
    ] == checksum(tmp_path / "model.joblib")
    with pytest.raises(ValueError, match="exists"):
        train_baseline(tmp_path, tmp_path / "model.joblib")
    with (tmp_path / "dataset.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        train_baseline(tmp_path, tmp_path / "another.joblib")
