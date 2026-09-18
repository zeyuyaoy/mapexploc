"""k-NN classification model for protein subcellular localization."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import GridSearchCV, KFold
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
        Test set features for final evaluation
    test_targets : pd.Series, optional
        Test set targets for final evaluation
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
    logger.info(
        "Starting k-NN training with %d samples and %d features",
        len(features),
        features.shape[1] - 1,
    )  # -1 for target column

    # Prepare data
    if "localization" in features.columns:
        X_train = features.drop("localization", axis=1)
        y_train = features["localization"]
    else:
        X_train = features
        y_train = targets

    validate_features(X_train)
    if len(X_train) != len(y_train):
        raise ValueError("Training features and labels must have the same row count")
    if len(X_train) < 2:
        raise ValueError("k-NN training requires at least two samples")
    if y_train.isna().any():
        raise ValueError("Training labels must not contain missing values")

    # Adjust CV folds if needed
    n_samples = len(X_train)
    if cv > n_samples:
        cv = max(2, n_samples - 1)
        logger.warning("Reduced CV folds to %d due to small dataset size", cv)

    # Default parameter grid to match notebook behavior
    if param_grid is None:
        # For cross-validation, we need k <= smallest_training_set_size
        # With cv folds, the training size per fold is approximately:
        # n_samples * (cv-1)/cv
        min_train_size_per_fold = int(n_samples * (cv - 1) / cv)

        # For very small datasets, be even more conservative
        if n_samples <= 10:
            max_k = min(2, min_train_size_per_fold)
        else:
            max_k = min(11, min_train_size_per_fold)

        k_values = [k for k in [1, 2, 3, 5, 7, 9, 11] if k <= max_k]
        if not k_values:  # Absolute fallback
            k_values = [1]

        logger.info(
            "Using k values: %s for dataset size %d with %d CV folds",
            k_values,
            n_samples,
            cv,
        )

        param_grid = {
            "knn__n_neighbors": k_values,
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["euclidean", "manhattan"],
        }

    # Create pipeline with scaling (matching notebook)
    pipeline = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])

    # Use regular KFold for small datasets to avoid stratification issues
    if n_samples <= 10:
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        cv_splitter = cv

    # Grid search with cross-validation
    logger.info(
        "Starting k-NN grid search with %d parameter combinations",
        len(param_grid["knn__n_neighbors"])
        * len(param_grid.get("knn__weights", [1]))
        * len(param_grid.get("knn__metric", [1])),
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True,
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
    """Comprehensive k-NN model evaluation.

    Args:
        model: Trained k-NN model
        X_val: Validation features
        y_val: Validation labels
        output_dir: Directory to save results

    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            auc,
            classification_report,
            confusion_matrix,
            f1_score,
            roc_curve,
        )
    except ImportError:
        logger.error("scikit-learn is required for evaluation")
        raise ImportError("scikit-learn is required for evaluation")

    import os

    # Make predictions
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    # Basic metrics
    accuracy = accuracy_score(y_val, y_pred)
    f1_weighted = f1_score(y_val, y_pred, average="weighted")

    logger.info("Validation accuracy: %.4f", accuracy)
    logger.info("Validation F1-weighted: %.4f", f1_weighted)

    # Classification report
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)

    # Confusion matrix
    classes = model.named_steps["knn"].classes_
    cm = confusion_matrix(y_val, y_pred, labels=classes)

    # ROC analysis (for multi-class)
    roc_data = {}
    for i, cls in enumerate(classes):
        y_class = (np.asarray(y_val) == cls).astype(int)
        if np.unique(y_class).size < 2:
            logger.warning("Skipping ROC for class %s absent from one outcome", cls)
            continue
        fpr, tpr, _ = roc_curve(y_class, y_proba[:, i])
        roc_auc = auc(fpr, tpr)

        roc_data[cls] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}

    # Save results if output directory provided
    if output_dir:
        os.makedirs(f"{output_dir}/csv", exist_ok=True)

        # Save classification report
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f"{output_dir}/csv/knn_classification_report.csv")

        # Save confusion matrix
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        cm_df.to_csv(f"{output_dir}/csv/knn_confusion_matrix.csv")

        # Save ROC data for each class
        for cls, data in roc_data.items():
            roc_df = pd.DataFrame({"fpr": data["fpr"], "tpr": data["tpr"]})
            roc_df.to_csv(f"{output_dir}/csv/knn_roc_curve_{cls}.csv", index=False)

    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "classes": classes.tolist(),
        "roc_data": roc_data,
    }


__all__ = ["train_knn", "knn_predict", "knn_predict_proba", "evaluate_knn"]
