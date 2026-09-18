"""Model adapter interface for MAP-ExPLoc."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from .features import build_feature_matrix


@runtime_checkable
class BaseModelAdapter(Protocol):
    """Minimal interface expected from user models.

    Models must implement ``predict`` and ``predict_proba`` operating on batches
    of sequences. Optionally ``embed`` may be provided to expose intermediate
    representations that can accelerate explainers such as Deep SHAP.
    """

    def predict(self, batch: Sequence[str]) -> np.ndarray:
        """Return predicted class labels for ``batch``."""

    def predict_proba(self, batch: Sequence[str]) -> np.ndarray:
        """Return class probabilities for ``batch``."""

    def embed(
            self, batch: Sequence[str]
    ) -> np.ndarray | None:  # pragma: no cover - optional
        """Return embeddings for ``batch`` if available."""
        raise NotImplementedError


class _SimpleAdapter:
    """Wrap objects implementing the required methods into ``BaseModelAdapter``."""

    def __init__(self, model: Any):
        """Initialize the adapter with the given model."""
        self.model = model

    def predict(self, batch: Sequence[str]) -> np.ndarray:
        """Return predicted class labels for batch by delegating to wrapped model."""
        return np.asarray(self.model.predict(batch))

    def predict_proba(self, batch: Sequence[str]) -> np.ndarray:
        """Return class probabilities for batch by delegating to wrapped model."""
        return np.asarray(self.model.predict_proba(batch))

    def embed(self, batch: Sequence[str]) -> np.ndarray | None:
        """Return embeddings for batch if the wrapped model supports it."""
        if hasattr(self.model, "embed"):
            return np.asarray(self.model.embed(batch))
        return None


class FeatureModelAdapter:
    """Adapt a fitted feature-based classifier to accept protein sequences."""

    def __init__(self, model: Any):
        if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
            raise TypeError("Feature model does not expose predict/predict_proba")
        self.model = model

    def prepare(self, batch: Sequence[str]) -> pd.DataFrame:
        """Build one reusable feature matrix for a sequence batch."""
        return build_feature_matrix(list(batch))

    def predict(self, batch: Sequence[str]) -> np.ndarray:
        """Featureize and predict class labels for a sequence batch."""
        return np.asarray(self.model.predict(self.prepare(batch)))

    def predict_proba(self, batch: Sequence[str]) -> np.ndarray:
        """Featureize and predict class probabilities for a sequence batch."""
        return np.asarray(self.model.predict_proba(self.prepare(batch)))

    def embed(self, batch: Sequence[str]) -> np.ndarray | None:
        """Return the engineered feature representation."""
        return np.asarray(self.prepare(batch).to_numpy(), dtype=float)

    @property
    def classes(self) -> tuple[str, ...]:
        """Return fitted class labels in probability-column order."""
        estimator = getattr(self.model, "named_steps", {}).get("rf", self.model)
        labels = getattr(estimator, "classes_", getattr(self.model, "classes_", ()))
        return tuple(str(label) for label in labels)


def load_adapter(model: Any) -> BaseModelAdapter:
    """Return a ``BaseModelAdapter`` for ``model``.

    ``model`` may already satisfy the protocol or provide ``predict`` and
    ``predict_proba`` methods. Otherwise a ``TypeError`` is raised.
    """

    if isinstance(model, BaseModelAdapter):
        return model
    if hasattr(model, "predict") and hasattr(model, "predict_proba"):
        return _SimpleAdapter(model)
    raise TypeError("Model does not expose predict/predict_proba")


__all__ = ["BaseModelAdapter", "FeatureModelAdapter", "load_adapter"]
