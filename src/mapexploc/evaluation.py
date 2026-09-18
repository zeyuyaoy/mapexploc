"""Evaluation utilities for predictions and explanations."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .estimators import final_estimator


def classification_metrics(
    truth: Any, predicted: Any, probabilities: np.ndarray, classes: list[str]
) -> dict[str, Any]:
    """Score a fixed class universe, with explicitly aligned probability columns.

    Missing-class F1/recall are zero in the fixed macro average. Class AUC/AP
    are undefined (None) without both positive and negative observations.
    """
    actual, guessed = np.asarray(truth), np.asarray(predicted)
    prob = np.asarray(probabilities, dtype=float)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Classes must contain at least two unique labels")
    if actual.ndim != 1 or guessed.shape != actual.shape or len(actual) == 0:
        raise ValueError("Truth and predictions must be nonempty aligned vectors")
    if not np.isin(actual, classes).all() or not np.isin(guessed, classes).all():
        raise ValueError("Truth and predictions must use known classes")
    if prob.shape != (len(actual), len(classes)):
        raise ValueError("Probability shape must match rows and classes")
    if (
        not np.isfinite(prob).all()
        or (prob < 0).any()
        or (prob > 1).any()
        or not np.allclose(prob.sum(axis=1), 1, atol=1e-8, rtol=0)
    ):
        raise ValueError("Probabilities must be finite, nonnegative and sum to one")
    one_hot = actual[:, None] == np.asarray(classes)[None, :]
    report = classification_report(
        actual, guessed, labels=classes, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(actual, guessed, labels=classes)
    per_class = {}
    for i, label in enumerate(classes):
        binary = one_hot[:, i]
        valid = bool(binary.any() and not binary.all())
        negatives = int((~binary).sum())
        fp = int(matrix[:, i].sum() - matrix[i, i])
        per_class[label] = {
            "auroc": float(roc_auc_score(binary, prob[:, i])) if valid else None,
            "average_precision": (
                float(average_precision_score(binary, prob[:, i])) if valid else None
            ),
            "specificity": 1 - fp / negatives if negatives else None,
            "prevalence": float(binary.mean()),
        }
    confidence = prob.max(axis=1)
    correct = np.asarray(classes)[prob.argmax(axis=1)] == actual
    bins = np.minimum((confidence * 10).astype(int), 9)
    reliability = []
    for i in range(10):
        mask = bins == i
        reliability.append(
            {
                "lower": i / 10,
                "upper": (i + 1) / 10,
                "count": int(mask.sum()),
                "accuracy": float(correct[mask].mean()) if mask.any() else None,
                "confidence": float(confidence[mask].mean()) if mask.any() else None,
            }
        )
    return {
        "macro_f1": float(
            f1_score(actual, guessed, labels=classes, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                actual, guessed, labels=classes, average="weighted", zero_division=0
            )
        ),
        "balanced_accuracy": float(
            np.mean([report[str(c)]["recall"] for c in classes])
        ),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "classes": classes,
        # Calculate true-class loss directly: sklearn log_loss sorts labels,
        # irrespective of a caller's probability-column ordering.
        "log_loss": float(-np.log(np.clip(prob[one_hot], 1e-15, 1)).mean()),
        "multiclass_brier": float(np.square(prob - one_hot).sum(axis=1).mean()),
        "top_label_ece_10_bins": expected_calibration_error(
            correct.tolist(), confidence.tolist()
        ),
        "reliability_bins": reliability,
        "per_class_probability": per_class,
        "macro_auroc": (
            float(np.mean([v["auroc"] for v in per_class.values()]))
            if all(v["auroc"] is not None for v in per_class.values())
            else None
        ),
        "macro_average_precision": (
            float(np.mean([v["average_precision"] for v in per_class.values()]))
            if all(v["average_precision"] is not None for v in per_class.values())
            else None
        ),
    }


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
    """Mean of precomputed reference-minus-perturbed score drops.

    This is the discrete AOPC convention for equally weighted perturbation steps.
    Callers must supply signed drops, not raw probabilities or arbitrary outputs.
    This function neither constructs perturbations nor establishes faithfulness.
    """
    scores_array = np.asarray(scores, dtype=float)
    if (
        scores_array.ndim != 1
        or not scores_array.size
        or not np.isfinite(scores_array).all()
    ):
        raise ValueError("Perturbation drops must be a nonempty finite vector")
    return float(scores_array.mean())


def mean_absolute_score_change(
    reference: Sequence[float], perturbed: Sequence[float]
) -> float:
    """Mean absolute output change; not insertion/deletion curve AUC or causality."""
    reference_array = np.asarray(reference, dtype=float)
    perturbed_array = np.asarray(perturbed, dtype=float)
    if (
        reference_array.ndim != 1
        or reference_array.shape != perturbed_array.shape
        or reference_array.size == 0
        or not np.isfinite(reference_array).all()
        or not np.isfinite(perturbed_array).all()
    ):
        raise ValueError(
            "Reference and perturbed outputs must be finite nonempty aligned vectors"
        )
    return float(np.abs(reference_array - perturbed_array).mean())


def insertion_deletion(reference: Sequence[float], perturbed: Sequence[float]) -> float:
    """Deprecated compatibility alias for mean_absolute_score_change."""
    warnings.warn(
        "insertion_deletion computes mean absolute score change, not curve AUC; "
        "use mean_absolute_score_change instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return mean_absolute_score_change(reference, perturbed)


def evaluate_classifier(
    model: Any,
    features: pd.DataFrame,
    truth: pd.Series,
    *,
    output_dir: str | Path | None = None,
    prefix: str = "model",
) -> dict[str, Any]:
    """Shared fixed-class evaluation for generic fitted classifiers.

    No fitting or threshold selection occurs. The caller is responsible for split
    independence. Undefined discrimination curves are explicitly recorded as null.
    Existing RF/k-NN report keys are retained alongside the common probability metrics.
    """
    estimator = final_estimator(model)
    if not prefix.isidentifier():
        raise ValueError("Report prefix must be a simple identifier")
    classes: list[str] = list(estimator.classes_)
    predicted = np.asarray(model.predict(features))
    probabilities = np.asarray(model.predict_proba(features))
    result = classification_metrics(truth, predicted, probabilities, classes)
    roc_data, pr_data, brier_scores = {}, {}, {}
    for i, label in enumerate(classes):
        binary = np.asarray(truth) == label
        brier_scores[label] = float(np.square(probabilities[:, i] - binary).mean())
        if binary.any() and not binary.all():
            fpr, tpr, _ = roc_curve(binary, probabilities[:, i])
            precision, recall, _ = precision_recall_curve(binary, probabilities[:, i])
            roc_data[label] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": result["per_class_probability"][label]["auroc"],
            }
            pr_data[label] = {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "avg_precision": result["per_class_probability"][label][
                    "average_precision"
                ],
            }
        else:
            roc_data[label] = {"fpr": [], "tpr": [], "auc": None}
            pr_data[label] = {"precision": [], "recall": [], "avg_precision": None}
    importance = getattr(estimator, "feature_importances_", None)
    result.update(
        {
            "accuracy": float(np.mean(np.asarray(truth) == predicted)),
            "f1_weighted": result["weighted_f1"],
            "roc_data": roc_data,
            "pr_data": pr_data,
            "brier_scores": brier_scores,
            "feature_importance": (
                np.asarray(importance).tolist() if importance is not None else None
            ),
        }
    )
    if output_dir:
        directory = Path(output_dir) / "csv"
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result["classification_report"]).T.to_csv(
            directory / f"{prefix}_classification_report.csv"
        )
        pd.DataFrame(result["confusion_matrix"], index=classes, columns=classes).to_csv(
            directory / f"{prefix}_confusion_matrix.csv"
        )
        pd.DataFrame(
            [(c, r["auc"]) for c, r in roc_data.items()], columns=["class", "auc"]
        ).to_csv(directory / f"{prefix}_roc_auc_values.csv", index=False)
        pd.DataFrame(
            [(c, r["avg_precision"]) for c, r in pr_data.items()],
            columns=["class", "avg_precision"],
        ).to_csv(directory / f"{prefix}_pr_avg_precision.csv", index=False)
        pd.DataFrame.from_dict(
            brier_scores, orient="index", columns=["brier_score"]
        ).to_csv(directory / f"{prefix}_brier_scores.csv")
        for label, curve in roc_data.items():
            pd.DataFrame({"fpr": curve["fpr"], "tpr": curve["tpr"]}).to_csv(
                directory / f"{prefix}_roc_curve_{quote(str(label), safe='')}.csv",
                index=False,
            )
        if importance is not None and len(importance) == features.shape[1]:
            pd.DataFrame(
                {"feature": features.columns, "importance": importance}
            ).sort_values("importance", ascending=False).to_csv(
                directory / f"{prefix}_feature_importances.csv", index=False
            )
    return result


__all__ = [
    "classification_metrics",
    "evaluate_classifier",
    "expected_calibration_error",
    "aopc",
    "mean_absolute_score_change",
    "insertion_deletion",
]
