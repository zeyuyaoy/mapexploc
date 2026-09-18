"""Evaluation utilities for predictions and explanations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def expected_calibration_error(
        y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10
) -> float:
    """Compute sample-weighted binary Expected Calibration Error (ECE)."""
    truth = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(y_prob, dtype=float)
    if truth.shape != probabilities.shape or truth.ndim != 1:
        raise ValueError("y_true and y_prob must be one-dimensional and equally sized")
    if len(truth) == 0:
        raise ValueError("At least one prediction is required")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("y_true must contain binary labels")
    if (
            not np.isfinite(probabilities).all()
            or not ((probabilities >= 0) & (probabilities <= 1)).all()
    ):
        raise ValueError("y_prob must contain finite probabilities between 0 and 1")

    bins = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for bin_index in range(n_bins):
        mask = bins == bin_index
        if mask.any():
            error += float(mask.mean()) * abs(
                float(truth[mask].mean()) - float(probabilities[mask].mean())
            )
    return error


def aopc(scores: Sequence[float]) -> float:
    """Area over the Perturbation Curve (AOPC)."""
    scores_array = np.asarray(scores)
    if scores_array.size == 0:
        raise ValueError("At least one perturbation score is required")
    return float(scores_array.mean())


def insertion_deletion(reference: Sequence[float], perturbed: Sequence[float]) -> float:
    """Faithfulness metric comparing reference and perturbed outputs."""
    reference_array = np.asarray(reference)
    perturbed_array = np.asarray(perturbed)
    if reference_array.shape != perturbed_array.shape or reference_array.size == 0:
        raise ValueError(
            "Reference and perturbed outputs must be non-empty and aligned"
        )
    return float(np.abs(reference_array - perturbed_array).mean())


__all__ = ["expected_calibration_error", "aopc", "insertion_deletion"]
