"""k-NN classification model for protein subcellular localization."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ..evaluation import evaluate_classifier
from ..validation import grouped_splits

try:
    from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise ImportError("scikit-learn is required for k-NN modeling")

logger = logging.getLogger(__name__)


def validate_features(features: pd.DataFrame) -> None:
    """Simple validation for feature matrix."""
    if features.empty:
        raise ValueError("Features DataFrame is empty")
    if features.isna().all().any():
        raise ValueError("Features DataFrame contains all-NaN columns")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise ValueError("Features DataFrame must contain only numeric columns")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Features DataFrame contains NaN or infinite values")


def train_knn(
    features: pd.DataFrame,
    targets: pd.Series,
    test_features: pd.DataFrame | None = None,
    test_targets: pd.Series | None = None,
    param_grid: dict[str, Any] | None = None,
    cv: int = 5,
    n_jobs: int = -1,
    scoring: str = "f1_macro",
    random_state: int = 42,
    groups: pd.Series | np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Train a k-NN classifier with hyperparameter tuning using grid search.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix with protein features
    targets : pd.Series
        Target localization labels
    test_features : pd.DataFrame, optional
        Deprecated, rejected: evaluate separately after freezing model selection
    test_targets : pd.Series, optional
        Deprecated, rejected: evaluate separately after freezing model selection
    cv : int, default=5
        Number of cross-validation folds
    n_jobs : int, default=-1
        Number of parallel jobs for grid search
    scoring : str, default='f1_macro'
        Scoring metric for grid search
    random_state : int, default=42
        Random state for reproducibility

    Returns
    -------
    dict[str, Any]
        Dictionary containing trained model, best parameters, and evaluation results
    """
    if test_features is not None or test_targets is not None:
        raise ValueError(
            "train_knn does not evaluate test data; call evaluate_knn separately "
            "after model selection is frozen"
        )
    if cv < 2:
        raise ValueError("Cross-validation requires at least two folds")
    if "localization" in features.columns:
        if not np.array_equal(features["localization"].to_numpy(), targets.to_numpy()):
            raise ValueError("Embedded localization conflicts with supplied targets")
        X_train = features.drop(columns="localization")
    else:
        X_train = features
    y_train = targets
    validate_features(X_train)
    if len(X_train) != len(y_train):
        raise ValueError("Training features and labels must have the same row count")
    if y_train.isna().any() or y_train.nunique() < 2:
        raise ValueError(
            "Training requires at least two nonmissing localization classes"
        )
    minimum = int(y_train.value_counts().min())
    if minimum < 2:
        raise ValueError("k-NN selection requires at least two samples in every class")
    cv = min(cv, minimum)
    if groups is not None:
        splits = grouped_splits(y_train, groups, cv, random_state)
    else:
        logger.warning(
            "Ungrouped CV cannot establish generalization to unrelated proteins"
        )
        splits = list(
            StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state).split(
                X_train, y_train
            )
        )
    min_train_size = min(len(fit) for fit, _ in splits)
    if param_grid is None:
        max_neighbors = min(2 if len(X_train) <= 10 else 11, min_train_size)
        param_grid = {
            "knn__n_neighbors": [
                k for k in (1, 2, 3, 5, 7, 9, 11) if k <= max_neighbors
            ],
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["euclidean", "manhattan"],
        }
    grid = list(ParameterGrid(param_grid))
    if not grid or any(p.get("knn__n_neighbors", 5) > min_train_size for p in grid):
        raise ValueError("Neighbor count exceeds the smallest training fold")
    logger.info(
        "k-NN selection: %d samples, %d features, %d configurations",
        len(X_train),
        X_train.shape[1],
        len(grid),
    )
    pipeline = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=splits,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True,
        error_score="raise",
    )

    grid_search.fit(X_train, y_train)

    logger.info("Best k-NN parameters: %s", grid_search.best_params_)
    logger.info("Best CV %s score: %.4f", scoring, grid_search.best_score_)

    # Convert results to list of dictionaries for easier analysis
    cv_results = []
    for i in range(len(grid_search.cv_results_["mean_test_score"])):
        result = {}
        for key, values in grid_search.cv_results_.items():
            result[key] = values[i]
        cv_results.append(result)

    # Return dictionary with all results
    return {
        "model": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "best_cv_score": grid_search.best_score_,
        "cv_results": cv_results,
        "grid_search": grid_search,
    }


def knn_predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Predict using trained k-NN model.

    Args:
        model: Trained k-NN model (pipeline)
        X: Features to predict on

    Returns:
        Predicted labels
    """
    logger.debug("Running k-NN inference on %d samples", len(X))
    return np.asarray(model.predict(X))


def knn_predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Predict class probabilities using trained k-NN model.

    Args:
        model: Trained k-NN model (pipeline)
        X: Features to predict on

    Returns:
        Predicted class probabilities
    """
    logger.debug("Running k-NN probability inference on %d samples", len(X))
    return np.asarray(model.predict_proba(X))


def evaluate_knn(
    model: Any, X_val: pd.DataFrame, y_val: pd.Series, output_dir: str = "results"
) -> dict[str, Any]:
    """Evaluate without fitting using shared, fixed-class metrics and CSV reports.

    The caller must establish evaluation-set independence. Pass an empty
    output_dir to return metrics without writing files.
    """
    return evaluate_classifier(model, X_val, y_val, output_dir=output_dir, prefix="knn")


__all__ = ["train_knn", "knn_predict", "knn_predict_proba", "evaluate_knn"]
