"""Tests for evaluation metric validation and weighting."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mapexploc.evaluation import (
    aopc,
    expected_calibration_error,
    insertion_deletion,
    mean_absolute_score_change,
)
from mapexploc.models.knn import evaluate_knn
from mapexploc.models.rf import evaluate_rf


def test_expected_calibration_error_is_sample_weighted() -> None:
    error = expected_calibration_error([0, 0, 0, 1], [0.1, 0.1, 0.1, 0.9], n_bins=2)
    assert error == pytest.approx(0.1)


def test_metric_inputs_must_be_aligned() -> None:
    with pytest.raises(ValueError, match="equally sized"):
        expected_calibration_error([0], [0.1, 0.2])
    with pytest.raises(ValueError, match="aligned"):
        mean_absolute_score_change([0.1], [0.1, 0.2])


def test_perturbation_summaries_have_explicit_semantics() -> None:
    assert aopc([0.2, -0.1]) == pytest.approx(0.05)
    assert mean_absolute_score_change([0.5, 0.3], [0.3, 0.4]) == pytest.approx(0.15)
    with pytest.warns(DeprecationWarning, match="not curve AUC"):
        assert insertion_deletion([0.5], [0.3]) == pytest.approx(0.2)
    for invalid in ([], [np.nan], [np.inf], [[0.5]]):
        with pytest.raises(ValueError):
            aopc(invalid)
        with pytest.raises(ValueError):
            mean_absolute_score_change(invalid, invalid)


class FixedProbabilities:
    classes_ = np.array(["Nucleus", "Cytoplasm", "Secreted"])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.classes_[self.predict_proba(features).argmax(axis=1)]


@pytest.mark.parametrize("evaluate", [evaluate_rf, evaluate_knn])
def test_evaluator_retains_missing_classes_and_probability_order(
    evaluate, tmp_path: Path
) -> None:
    x = pd.DataFrame({"feature": [1, 2]})
    result = evaluate(
        FixedProbabilities(), x, pd.Series(["Nucleus", "Cytoplasm"]), str(tmp_path)
    )
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["log_loss"] == pytest.approx(-np.log([0.8, 0.7]).mean())
    assert result["classification_report"]["Secreted"]["support"] == 0
    assert result["roc_data"]["Secreted"]["auc"] is None
    assert result["pr_data"]["Secreted"]["avg_precision"] is None
    assert result["brier_scores"]["Secreted"] == pytest.approx(0.01)
    assert result["f1_weighted"] == result["weighted_f1"] == 1
    assert list((tmp_path / "csv").glob("*_classification_report.csv"))
    with pytest.raises(ValueError, match="known classes"):
        evaluate(FixedProbabilities(), x, pd.Series(["Nucleus", "Golgi"]), "")
