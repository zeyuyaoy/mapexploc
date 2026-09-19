"""Versioned model artifacts shared by the CLI and API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

from .estimators import final_estimator
from .features import FEATURE_NAMES

ARTIFACT_KIND = "mapexploc-model"
ARTIFACT_VERSION = 1


class ModelArtifactError(ValueError):
    """Raised when a model artifact is incompatible with this feature pipeline."""


@dataclass(frozen=True)
class ModelArtifact:
    """Loaded estimator and the metadata needed for safe inference."""

    model: Any
    feature_names: tuple[str, ...]
    classes: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def _model_classes(model: Any) -> tuple[str, ...]:
    estimator = final_estimator(model)
    classes = getattr(estimator, "classes_", getattr(model, "classes_", ()))
    return tuple(str(label) for label in classes)


def save_model_artifact(
    model: Any,
    path: Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save a pickle-based model with its feature-schema version.

    Exchange artifacts only with trusted parties. HTTP requests cannot
    supply artifact paths."""

    payload = {
        "kind": ARTIFACT_KIND,
        "version": ARTIFACT_VERSION,
        "model": model,
        "feature_names": list(FEATURE_NAMES),
        "classes": list(_model_classes(model)),
        "metadata": dict(metadata or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def _validate_feature_names(names: Sequence[object]) -> tuple[str, ...]:
    feature_names = tuple(str(name) for name in names)
    if feature_names != FEATURE_NAMES:
        raise ModelArtifactError(
            "Model feature schema does not match this MAP-ExPLoc version; retrain "
            "the model with the current feature extractor"
        )
    return feature_names


def _validate_estimator(model: Any) -> None:
    if not callable(getattr(model, "predict", None)) or not callable(
        getattr(model, "predict_proba", None)
    ):
        raise ModelArtifactError("Artifact has no compatible classifier")
    if getattr(model, "n_features_in_", None) != len(FEATURE_NAMES):
        raise ModelArtifactError("Estimator feature count does not match the schema")
    if len(_model_classes(model)) < 2:
        raise ModelArtifactError("Artifact has no fitted localization classes")


def load_model_artifact(path: Path) -> ModelArtifact:
    """Load a trusted model artifact, including legacy bare estimators."""

    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    try:
        payload = joblib.load(path)
    except Exception as exc:
        raise ModelArtifactError(
            "Could not deserialize the trusted model artifact"
        ) from exc
    if isinstance(payload, Mapping) and payload.get("kind") == ARTIFACT_KIND:
        if payload.get("version") != ARTIFACT_VERSION:
            raise ModelArtifactError(
                f"Unsupported model artifact version: {payload.get('version')}"
            )
        model = payload.get("model")
        if model is None:
            raise ModelArtifactError("Model artifact does not contain an estimator")
        _validate_estimator(model)
        feature_names = _validate_feature_names(payload.get("feature_names", ()))
        classes = tuple(str(label) for label in payload.get("classes", ()))
        if classes and classes != _model_classes(model):
            raise ModelArtifactError(
                "Artifact classes do not match estimator class order"
            )
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ModelArtifactError("Artifact metadata must be a mapping")
        return ModelArtifact(
            model, feature_names, classes or _model_classes(model), dict(metadata)
        )

    _validate_estimator(payload)
    names = getattr(payload, "feature_names_in_", FEATURE_NAMES)
    feature_names = _validate_feature_names(names)
    return ModelArtifact(payload, feature_names, _model_classes(payload))


__all__ = [
    "ARTIFACT_VERSION",
    "ModelArtifact",
    "ModelArtifactError",
    "load_model_artifact",
    "save_model_artifact",
]
