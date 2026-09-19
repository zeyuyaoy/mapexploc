"""SHAP explanations for MAP-ExPLoc feature-based tree models."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from ..estimators import final_estimator

logger = logging.getLogger(__name__)

try:
    import shap
except ImportError as exc:  # pragma: no cover - dependency guard
    logger.warning("SHAP is unavailable: %s", exc)
    shap = None


@dataclass
class Explanation:
    """Normalized SHAP values with shape ``samples, classes, features``."""

    shap_values: np.ndarray
    interaction_values: np.ndarray | None
    expected_value: float | np.ndarray

    def to_json(self) -> str:
        """Serialize the explanation to JSON."""
        expected = (
            self.expected_value.tolist()
            if isinstance(self.expected_value, np.ndarray)
            else self.expected_value
        )
        return json.dumps(
            {
                "shap_values": self.shap_values.tolist(),
                "interaction_values": None,
                "interaction_status": "unsupported",
                "expected_value": expected,
            }
        )


class ShapExplainer:
    """Explain a fitted tree-ensemble pipeline in engineered-feature space."""

    def __init__(self, model: Any, output_dir: str = "results/figures/shap_analysis"):
        if shap is None:
            raise ImportError(
                "SHAP is required. Install mapexploc with its core dependencies."
            )
        self.model = model
        self._lock = Lock()
        self.output_dir = Path(output_dir)
        self.rf_model = final_estimator(model)
        if not isinstance(
            self.rf_model, (RandomForestClassifier, ExtraTreesClassifier)
        ):
            raise TypeError(
                "Tree probability explanations require a fitted "
                "RandomForestClassifier or ExtraTreesClassifier"
            )
        self.explainer = shap.TreeExplainer(
            self.rf_model,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )

    @staticmethod
    def _transform_step(step: Any, data: pd.DataFrame) -> pd.DataFrame:
        """Allow only transformations with an explicit one-to-one feature mapping."""
        if isinstance(step, str):
            if step == "passthrough":
                return data
            if step == "drop":
                return data.iloc[:, :0]
        if isinstance(step, Pipeline):
            for _, child in step.steps:
                data = ShapExplainer._transform_step(child, data)
            return data
        if isinstance(step, ColumnTransformer):
            parts = []
            # Fitted selections resolve callable selectors and remainder columns.
            for name, child, _ in step.transformers_:
                indices = step._transformer_to_input_indices[name]
                if not indices:
                    continue
                parts.append(
                    ShapExplainer._transform_step(child, data.iloc[:, indices])
                )
            mapped = pd.concat(parts, axis=1) if parts else data.iloc[:, :0]
            if mapped.columns.duplicated().any():
                raise TypeError(
                    "Duplicated transformed feature identities are unsupported"
                )
            actual = step.transform(data)
            if hasattr(actual, "toarray"):
                actual = actual.toarray()
            if np.asarray(actual).shape != mapped.shape:
                raise ValueError("Transformed feature mapping has an invalid shape")
            return pd.DataFrame(actual, index=data.index, columns=mapped.columns)
        if isinstance(step, SelectorMixin):
            names = data.columns[step.get_support()]
        elif isinstance(
            step, (StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler)
        ):
            names = data.columns
        elif isinstance(step, FunctionTransformer) and step.func is None:
            names = data.columns
        else:
            raise TypeError(
                f"No biological feature mapping for {type(step).__name__}; "
                "use sequence-region explanations for the full predictor"
            )
        transformed = step.transform(data)
        return pd.DataFrame(np.asarray(transformed), columns=names, index=data.index)

    def _transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if not features.columns.is_unique:
            raise ValueError("Feature identities must be unique")
        transformed = features
        for _, step in getattr(self.model, "steps", ()):
            if step is self.rf_model:
                break
            # Skip resamplers used only during training.
            from imblearn.base import BaseSampler

            if isinstance(step, BaseSampler):
                continue
            transformed = self._transform_step(step, transformed)
        return transformed

    @staticmethod
    def _normalise_values(
        values: Any, n_samples: int, n_features: int, n_classes: int
    ) -> np.ndarray:
        # Accept output-last arrays or class lists; equal axis sizes are ambiguous.
        if isinstance(values, list):
            if len(values) != n_classes or any(
                np.asarray(v).shape != (n_samples, n_features) for v in values
            ):
                raise ValueError("Invalid legacy SHAP class shapes")
            array = np.stack(values, axis=1)
        else:
            raw = np.asarray(values)
            if raw.shape == (n_samples, n_features) and n_classes == 1:
                array = raw[:, None, :]
            elif raw.shape == (n_samples, n_features, n_classes):
                array = raw.transpose(0, 2, 1)
            else:
                raise ValueError(f"Unsupported SHAP value shape: {raw.shape}")
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite SHAP contributions")
        return np.asarray(array, dtype=float)

    def explain_sample(
        self, X_sample: pd.DataFrame, sample_size: int = 200, random_state: int = 42
    ) -> dict[str, Any]:
        """Compute normalized SHAP values for a deterministic sample."""
        if sample_size < 1:
            raise ValueError("sample_size must be positive")
        if X_sample.empty:
            raise ValueError("At least one feature row is required for explanation")
        sampled = (
            X_sample.sample(sample_size, random_state=random_state)
            if len(X_sample) > sample_size
            else X_sample.copy()
        )
        transformed = self._transform(sampled)
        # TreeExplainer updates internal state during shap_values. Snapshot its
        # base values under the same lock before another request can use it.
        with self._lock:
            raw_values = self.explainer.shap_values(transformed)
            expected = (
                np.asarray(self.explainer.expected_value, dtype=float)
                .reshape(-1)
                .copy()
            )
        classes = np.asarray(getattr(self.rf_model, "classes_", ()))
        class_count = len(classes)
        if class_count < 1 or len(set(classes)) != class_count:
            raise ValueError("Fitted class identities are invalid")
        if not np.array_equal(classes, self.model.classes_):
            raise ValueError("Pipeline and estimator class order differ")
        values = self._normalise_values(
            raw_values, len(transformed), transformed.shape[1], class_count
        )
        # Preserve feature identity after selection or reordering.
        mapped_values = np.zeros((len(sampled), class_count, sampled.shape[1]))
        for i, name in enumerate(transformed.columns):
            mapped_values[:, :, sampled.columns.get_loc(name)] = values[:, :, i]
        if expected.shape != (class_count,) or not np.isfinite(expected).all():
            raise ValueError("Invalid SHAP base-value shape")
        probabilities = np.asarray(self.model.predict_proba(sampled), dtype=float)
        reconstructed = expected + mapped_values.sum(axis=2)
        if probabilities.shape != reconstructed.shape or not np.allclose(
            reconstructed, probabilities, atol=1e-6, rtol=0
        ):
            raise ValueError(
                "SHAP does not reconstruct the complete model probabilities"
            )
        return {
            "shap_values": mapped_values,
            "output_space": "probability",
            "feature_perturbation": "tree_path_dependent",
            "background": "fitted weighted tree-path training counts",
            "residuals": reconstructed - probabilities,
            "transformed_feature_names": list(transformed.columns),
            "expected_value": expected,
            "X_sample": sampled,
            "X_transformed": transformed,
            "classes": classes,
        }

    def explain_predictions(
        self, features: pd.DataFrame, top_n: int = 12
    ) -> list[dict[str, Any]]:
        """Return top feature contributions for each predicted class."""
        explanation = self.explain_sample(features, sample_size=len(features))
        values = explanation["shap_values"]
        classes = [str(label) for label in explanation["classes"]]
        predictions = [str(label) for label in self.model.predict(features)]
        expected = np.atleast_1d(explanation["expected_value"])
        reports = []
        for row_index, prediction in enumerate(predictions):
            if prediction not in classes:
                raise ValueError("Prediction uses an undeclared class")
            class_index = classes.index(prediction)
            row_values = values[row_index, class_index]
            ranked = np.argsort(np.abs(row_values))[::-1][:top_n]
            reports.append(
                {
                    "prediction": prediction,
                    "base_value": float(expected[class_index]),
                    "remainder": float(row_values.sum() - row_values[ranked].sum()),
                    "output_space": "probability",
                    "feature_contributions": [
                        {
                            "feature": str(features.columns[index]),
                            "value": float(features.iloc[row_index, index]),
                            "contribution": float(row_values[index]),
                        }
                        for index in ranked
                    ],
                }
            )
        return reports

    def plot_summary(self, explanation: dict[str, Any], save: bool = True) -> None:
        """Render a global summary plot when matplotlib is installed."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "Plotting requires: pip install 'mapexploc[plots]'"
            ) from exc
        values = explanation["shap_values"]
        transformed = explanation["X_sample"]
        classes = [str(label) for label in explanation["classes"]]
        plot_values = [values[:, index, :] for index in range(values.shape[1])]
        shap.summary_plot(
            plot_values,
            transformed,
            feature_names=transformed.columns,
            class_names=classes or None,
            plot_type="bar",
            show=False,
        )
        if save:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(
                self.output_dir / "shap_summary_all_classes.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

    def generate_all_plots(
        self,
        X_data: pd.DataFrame,
        sample_size: int = 200,
        max_individual: int = 10,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Generate the supported summary plot and return its SHAP values."""
        del max_individual  # Retain the legacy public signature.
        explanation = self.explain_sample(X_data, sample_size, random_state)
        self.plot_summary(explanation)
        return explanation


def explain(model: Any, features: np.ndarray) -> np.ndarray:
    """Compatibility helper returning raw TreeExplainer values."""
    if shap is None:
        raise RuntimeError("SHAP is not installed")
    safe = ShapExplainer(model)
    names = getattr(
        model, "feature_names_in_", [f"feature_{i}" for i in range(features.shape[1])]
    )
    values = safe.explain_sample(
        pd.DataFrame(features, columns=names), sample_size=len(features)
    )["shap_values"]
    return np.asarray(values.transpose(0, 2, 1))


__all__ = ["ShapExplainer", "Explanation", "explain"]
