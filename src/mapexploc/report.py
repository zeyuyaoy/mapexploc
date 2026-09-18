"""Validated response models shared by the REST API and its documentation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ClassProbability(BaseModel):
    """Probability assigned to one model class."""

    label: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictionReport(BaseModel):
    """Prediction and uncalibrated model probabilities for one input sequence."""

    index: int = Field(ge=0)
    sequence_length: int = Field(gt=0)
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: list[ClassProbability]


class FeatureContribution(BaseModel):
    """One engineered feature's SHAP contribution to the predicted class."""

    feature: str
    value: float
    contribution: float


class ExplanationReport(PredictionReport):
    """Prediction plus local feature contributions for its predicted class."""

    base_value: float
    feature_contributions: list[FeatureContribution]


class PredictResponse(BaseModel):
    """Batch prediction response."""

    model_classes: list[str]
    results: list[PredictionReport]


class ExplainResponse(BaseModel):
    """Batch explanation response."""

    model_classes: list[str]
    results: list[ExplanationReport]


class HealthResponse(BaseModel):
    """Service readiness without exposing the configured filesystem path."""

    status: Literal["ready", "model_unavailable"]
    model_available: bool
    model_loaded: bool


__all__ = [
    "ClassProbability",
    "ExplainResponse",
    "ExplanationReport",
    "FeatureContribution",
    "HealthResponse",
    "SequenceFeatures",
    "FeaturesResponse",
    "ModelResponse",
    "PredictResponse",
    "PredictionReport",
]


class SequenceFeatures(BaseModel):
    """Measured sequence descriptors, independent of the configured model."""

    index: int
    sequence_length: int
    composition: dict[str, float]
    gravy: float
    isoelectric_point: float


class FeaturesResponse(BaseModel):
    """Descriptors in the same order as the request."""

    results: list[SequenceFeatures]


class ModelResponse(BaseModel):
    """Public model provenance; never includes server filesystem paths."""

    model_classes: list[str]
    feature_count: int = 423
    metadata_available: bool
    metadata: dict[str, Any]
