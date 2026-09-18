"""Tests for evaluation metric validation and weighting."""

import pytest

from mapexploc.evaluation import expected_calibration_error, insertion_deletion


def test_expected_calibration_error_is_sample_weighted() -> None:
    error = expected_calibration_error([0, 0, 0, 1], [0.1, 0.1, 0.1, 0.9], n_bins=2)
    assert error == pytest.approx(0.1)


def test_metric_inputs_must_be_aligned() -> None:
    with pytest.raises(ValueError, match="equally sized"):
        expected_calibration_error([0], [0.1, 0.2])
    with pytest.raises(ValueError, match="aligned"):
        insertion_deletion([0.1], [0.1, 0.2])
