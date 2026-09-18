"""SHAP explanations for MAP-ExPLoc feature-based tree models."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    interaction_values: np.ndarray
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
                "interaction_values": self.interaction_values.tolist(),
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
        self.output_dir = Path(output_dir)
        self.rf_model = final_estimator(model)
        if not hasattr(self.rf_model, "estimators_"):
            raise TypeError(
                "SHAP explanations currently require a fitted tree ensemble"
            )
        self.explainer = shap.TreeExplainer(self.rf_model)

    def _transform(self, features: pd.DataFrame) -> pd.DataFrame:
        transformed: Any = features
        for name, step in getattr(self.model, "steps", ()):
            if step is self.rf_model:
                break
            if hasattr(step, "transform"):
                transformed = step.transform(transformed)
        return pd.DataFrame(
            np.asarray(transformed), columns=features.columns, index=features.index
        )

    @staticmethod
    def _normalise_values(
        values: Any, n_samples: int, n_features: int, n_classes: int
    ) -> np.ndarray:
        if isinstance(values, list):
            return np.stack([np.asarray(value) for value in values], axis=1)
        array = np.asarray(values)
        if array.ndim == 2:
            return array.reshape(n_samples, 1, n_features)
        if array.ndim == 3 and array.shape == (n_samples, n_features, n_classes):
            return np.transpose(array, (0, 2, 1))
        if array.ndim == 3 and array.shape == (n_samples, n_classes, n_features):
            return array
        raise ValueError(f"Unsupported SHAP value shape: {array.shape}")

    def explain_sample(
        self, X_sample: pd.DataFrame, sample_size: int = 200, random_state: int = 42
    ) -> dict[str, Any]:
        """Compute normalized SHAP values for a deterministic sample."""
        if X_sample.empty:
            raise ValueError("At least one feature row is required for explanation")
        sampled = (
            X_sample.sample(sample_size, random_state=random_state)
            if len(X_sample) > sample_size
            else X_sample.copy()
        )
        transformed = self._transform(sampled)
        raw_values = self.explainer.shap_values(transformed)
        classes = np.asarray(getattr(self.rf_model, "classes_", ()))
        class_count = max(len(classes), 1)
        values = self._normalise_values(
            raw_values, len(transformed), transformed.shape[1], class_count
        )
        return {
            "shap_values": values,
            "expected_value": np.asarray(self.explainer.expected_value),
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
            class_index = classes.index(prediction) if prediction in classes else 0
            row_values = values[row_index, class_index]
            ranked = np.argsort(np.abs(row_values))[::-1][:top_n]
            reports.append(
                {
                    "prediction": prediction,
                    "base_value": float(expected[min(class_index, len(expected) - 1)]),
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
        transformed = explanation["X_transformed"]
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
        del max_individual  # Kept for compatibility with the earlier public signature.
        explanation = self.explain_sample(X_data, sample_size, random_state)
        self.plot_summary(explanation)
        return explanation


def explain(model: Any, features: np.ndarray) -> np.ndarray:
    """Compatibility helper returning raw TreeExplainer values."""
    if shap is None:
        raise RuntimeError("SHAP is not installed")
    return np.asarray(shap.TreeExplainer(model).shap_values(features))


__all__ = ["ShapExplainer", "Explanation", "explain"]
