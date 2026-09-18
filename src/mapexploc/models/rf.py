"""Random Forest model utilities."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def train_random_forest(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: dict[str, list[Any]] | None = None,
        cv: int = 3,
        scoring: str = "f1_weighted",
        n_jobs: int = -1,
        use_smote: bool = True,
        random_state: int = 42,
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

    # SMOTE runs inside each fold, so size its neighborhood for the smallest
    # expected training fold rather than for the complete dataset.
    min_fold_class_size = int(min_class_size * (cv - 1) / cv) if cv > 1 else 0
    if use_smote and min_fold_class_size < 2:
        logger.warning(
            "Disabling SMOTE because the smallest training fold has fewer than "
            "two samples for a class"
        )
        use_smote = False

    # Create pipeline with optional SMOTE and scaling
    steps = []
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
            ("scaler", StandardScaler()),
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
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
        random_state=random_state,
        return_train_score=True,
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
    """Comprehensive Random Forest model evaluation.

    Args:
        model: Trained Random Forest model
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
            average_precision_score,
            brier_score_loss,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_recall_curve,
            roc_curve,
        )
    except ImportError:
        logger.error("Required packages not installed")
        raise ImportError(
            "scikit-learn, matplotlib, and seaborn are required for evaluation"
        )

    import os

    # Make predictions
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    # Basic metrics
    accuracy = accuracy_score(y_val, y_pred)
    f1_weighted = f1_score(y_val, y_pred, average="weighted")

    logger.info("Validation accuracy: %.4f", accuracy)
    logger.info("Validation F1-weighted: %.4f", f1_weighted)

    # Get classes
    classes = model.named_steps["rf"].classes_

    # Classification report
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred, labels=classes)

    # ROC analysis
    roc_data = {}
    pr_data = {}
    brier_scores = {}

    for i, cls in enumerate(classes):
        y_class = (np.asarray(y_val) == cls).astype(int)
        if np.unique(y_class).size < 2:
            logger.warning("Skipping curves for class %s absent from one outcome", cls)
            continue
        fpr, tpr, _ = roc_curve(y_class, y_proba[:, i])
        precision, recall, _ = precision_recall_curve(y_class, y_proba[:, i])
        brier_score = brier_score_loss(y_class, y_proba[:, i])

        roc_auc = auc(fpr, tpr)
        avg_precision = average_precision_score(y_class, y_proba[:, i])

        roc_data[cls] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}
        pr_data[cls] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "avg_precision": avg_precision,
        }
        brier_scores[cls] = brier_score

    # Feature importance
    feature_importance = None
    if hasattr(model.named_steps["rf"], "feature_importances_"):
        feature_importance = model.named_steps["rf"].feature_importances_

    # Save results
    if output_dir:
        os.makedirs(f"{output_dir}/csv", exist_ok=True)
        os.makedirs(f"{output_dir}/figures", exist_ok=True)

        # Save metrics
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f"{output_dir}/csv/rf_classification_report.csv")

        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        cm_df.to_csv(f"{output_dir}/csv/rf_confusion_matrix.csv")

        # Save feature importance
        if feature_importance is not None:
            feat_df = pd.DataFrame(
                {
                    "feature": list(X_val.columns),
                    "importance": feature_importance,
                }
            ).sort_values("importance", ascending=False)
            feat_df.to_csv(f"{output_dir}/csv/rf_feature_importances.csv", index=False)

        # Save ROC and PR data
        roc_auc_df = pd.DataFrame(
            [(cls, data["auc"]) for cls, data in roc_data.items()],
            columns=["class", "auc"],
        )
        roc_auc_df.to_csv(f"{output_dir}/csv/rf_roc_auc_values.csv", index=False)

        pr_avg_df = pd.DataFrame(
            [(cls, data["avg_precision"]) for cls, data in pr_data.items()],
            columns=["class", "avg_precision"],
        )
        pr_avg_df.to_csv(f"{output_dir}/csv/rf_pr_avg_precision.csv", index=False)

        brier_df = pd.DataFrame.from_dict(
            brier_scores, orient="index", columns=["brier_score"]
        )
        brier_df.to_csv(f"{output_dir}/csv/rf_brier_scores.csv")

    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "classes": classes.tolist(),
        "roc_data": roc_data,
        "pr_data": pr_data,
        "brier_scores": brier_scores,
        "feature_importance": (
            feature_importance.tolist() if feature_importance is not None else None
        ),
    }


__all__ = ["train_random_forest", "rf_predict", "rf_predict_proba", "evaluate_rf"]
