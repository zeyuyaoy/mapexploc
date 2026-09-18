"""Random Forest model utilities."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ..evaluation import evaluate_classifier
from ..validation import grouped_splits

logger = logging.getLogger(__name__)


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict[str, list[Any]] | None = None,
    cv: int = 3,
    scoring: str = "f1_macro",
    n_jobs: int = -1,
    use_smote: bool = False,
    random_state: int = 42,
    groups: pd.Series | np.ndarray | None = None,
) -> dict[str, Any]:
    """Train Random Forest classifier with hyperparameter tuning and class balancing.

    Args:
        X_train: Training features
        y_train: Training labels
        param_grid: Parameter grid for randomized search
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        use_smote: Whether to use SMOTE for class balancing
        random_state: Random state for reproducibility
        groups: Homology groups; omission is for exploratory/workflow use only

    Returns:
        Dictionary containing trained model, best parameters, and evaluation results
    """
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import (
            ParameterGrid,
            RandomizedSearchCV,
            StratifiedKFold,
        )
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.error(
            "Required packages not installed. Install with: "
            "pip install scikit-learn imbalanced-learn"
        )
        raise ImportError("scikit-learn and imbalanced-learn are required")

    if X_train.empty:
        raise ValueError("Training features must not be empty")
    if cv < 2:
        raise ValueError("Cross-validation requires at least two folds")
    if len(X_train) != len(y_train):
        raise ValueError("Training features and labels must have the same row count")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_train.dtypes):
        raise ValueError("Training features must all be numeric")
    if not np.isfinite(X_train.to_numpy(dtype=float)).all():
        raise ValueError("Training features must not contain NaN or infinite values")
    if y_train.isna().any():
        raise ValueError("Training labels must not contain missing values")
    if y_train.nunique() < 2:
        raise ValueError("Training data must contain at least two localization classes")

    if param_grid is None:
        param_grid = {
            "rf__n_estimators": [100, 200],
            "rf__max_depth": [None, 20],
            "rf__min_samples_split": [2, 5],
            "rf__min_samples_leaf": [1, 2],
        }

    min_class_size = int(y_train.value_counts().min())
    can_cross_validate = min_class_size >= 2
    if can_cross_validate:
        cv = min(cv, min_class_size)

    splits = (
        grouped_splits(y_train, groups, cv, random_state)
        if groups is not None
        else None
    )
    if groups is None:
        logger.warning(
            "Ungrouped CV cannot establish generalization to unrelated proteins"
        )
    # SMOTE runs inside each fold, so size its neighborhood for the smallest
    # expected training fold rather than for the complete dataset.
    min_fold_class_size = int(min_class_size * (cv - 1) / cv) if cv > 1 else 0
    if splits is not None:
        min_fold_class_size = min(
            int(y_train.iloc[fit].value_counts().min()) for fit, _ in splits
        )
    if use_smote and min_fold_class_size < 2:
        logger.warning(
            "Disabling SMOTE because the smallest training fold has fewer than "
            "two samples for a class"
        )
        use_smote = False

    # Create pipeline with optional SMOTE and scaling
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if use_smote:
        steps.append(
            (
                "smote",
                SMOTE(
                    random_state=random_state,
                    k_neighbors=min(5, min_fold_class_size - 1),
                ),
            )
        )
    steps.extend(
        [
            (
                "rf",
                RandomForestClassifier(
                    class_weight="balanced", random_state=random_state
                ),
            ),
        ]
    )

    pipeline = ImbPipeline(steps)

    if not can_cross_validate:
        params = next(iter(ParameterGrid(param_grid)))
        logger.warning(
            "Fitting without cross-validation because at least one class has only "
            "one sample; use a larger dataset for meaningful model selection"
        )
        pipeline.set_params(**params)
        pipeline.fit(X_train, y_train)
        return {
            "model": pipeline,
            "best_params": params,
            "best_cv_score": None,
            "cv_results": [],
            "search": None,
        }

    # Randomized search with stratified cross-validation
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Adjust n_iter if grid size is smaller to avoid warnings
    grid_size = len(ParameterGrid(param_grid))
    n_iter = min(20, grid_size)

    logger.info("Starting RandomForest randomized search with %d iterations", n_iter)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=splits if splits is not None else cv_splitter,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
        random_state=random_state,
        return_train_score=True,
        error_score="raise",
    )

    search.fit(X_train, y_train)

    logger.info("Best Random Forest parameters: %s", search.best_params_)
    logger.info("Best CV %s score: %.4f", scoring, search.best_score_)

    # Convert results to list of dictionaries
    cv_results = []
    for i in range(len(search.cv_results_["mean_test_score"])):
        result = {}
        for key, values in search.cv_results_.items():
            result[key] = values[i]
        cv_results.append(result)

    # Return dictionary with all results
    return {
        "model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
        "cv_results": cv_results,
        "search": search,
    }


def rf_predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Predict using trained Random Forest model.

    Args:
        model: Trained Random Forest model (pipeline)
        X: Features to predict on

    Returns:
        Predicted labels
    """
    logger.debug("Running Random Forest inference on %d samples", len(X))
    return np.asarray(model.predict(X))


def rf_predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Predict class probabilities using trained Random Forest model.

    Args:
        model: Trained Random Forest model (pipeline)
        X: Features to predict on

    Returns:
        Predicted class probabilities
    """
    logger.debug("Running Random Forest probability inference on %d samples", len(X))
    return np.asarray(model.predict_proba(X))


def evaluate_rf(
    model: Any, X_val: pd.DataFrame, y_val: pd.Series, output_dir: str = "results"
) -> dict[str, Any]:
    """Evaluate without fitting using shared, fixed-class metrics and CSV reports.

    The caller must establish evaluation-set independence. Pass an empty
    output_dir to return metrics without writing files.
    """
    return evaluate_classifier(model, X_val, y_val, output_dir=output_dir, prefix="rf")


__all__ = ["train_random_forest", "rf_predict", "rf_predict_proba", "evaluate_rf"]
