"""Research inference API with input validation and model-level explanations."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .adapter import BaseModelAdapter, FeatureModelAdapter, load_adapter
from .artifacts import ModelArtifactError, load_model_artifact
from .default_model import resolve_default_model
from .estimators import final_estimator
from .explainers.shap import ShapExplainer
from .features import (
    AMINO_ACIDS,
    FEATURE_NAMES,
    build_feature_matrix,
    normalize_protein_sequence,
)
from .report import (
    ClassProbability,
    ExplainResponse,
    ExplanationReport,
    FeatureContribution,
    FeaturesResponse,
    HealthResponse,
    ModelResponse,
    PredictionReport,
    PredictResponse,
    SequenceFeatures,
)

MAX_BATCH_SIZE = 100
MAX_BATCH_RESIDUES = 1_000_000


class PredictRequest(BaseModel):
    """A bounded batch of unambiguous protein sequences."""

    model_config = ConfigDict(extra="forbid")
    sequences: list[str] = Field(min_length=1, max_length=MAX_BATCH_SIZE)

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, sequences: list[str]) -> list[str]:
        """Normalize sequence case and whitespace before inference."""
        return [normalize_protein_sequence(sequence) for sequence in sequences]

    @model_validator(mode="after")
    def validate_batch_size(self) -> PredictRequest:
        """Bound total work independently of the item-count limit."""
        if sum(map(len, self.sequences)) > MAX_BATCH_RESIDUES:
            raise ValueError(f"Batch exceeds the {MAX_BATCH_RESIDUES:,}-residue limit")
        return self


class ExplainRequest(PredictRequest):
    """Prediction request with a bounded number of returned contributions."""

    top_n: int = Field(default=12, ge=1, le=25)


class _ModelRuntime:
    def __init__(
        self, model: Any | None, model_path: Path | None, use_default: bool = False
    ):
        self.use_default = use_default
        self.model_path = model_path
        self.model: Any | None = None
        self.adapter: BaseModelAdapter | None = None
        self.explainer: ShapExplainer | None = None
        self.metadata: dict[str, Any] = {}
        self.lock = Lock()
        if model is not None:
            self._set_model(model)

    def _set_model(self, model: Any) -> None:
        if isinstance(model, FeatureModelAdapter):
            self.adapter = model
            self.model = model.model
        elif hasattr(model, "named_steps") or hasattr(model, "n_features_in_"):
            self.model = model
            self.adapter = FeatureModelAdapter(model)
        else:
            self.adapter = load_adapter(model)

    def get_adapter(self) -> BaseModelAdapter:
        if self.adapter is not None:
            return self.adapter
        with self.lock:
            if self.adapter is not None:
                return self.adapter
            if self.model_path is not None:
                artifact = load_model_artifact(self.model_path)
            elif self.use_default:
                artifact = resolve_default_model().load()
            else:
                raise RuntimeError("No model is configured")
            self._set_model(artifact.model)
            self.metadata = dict(artifact.metadata)
            if self.metadata.get("model_id"):
                self.metadata.setdefault(
                    "model_family", type(final_estimator(artifact.model)).__name__
                )
            if self.metadata.get("evaluation"):
                self.metadata.setdefault("evaluation_status", "historical_holdout")
        assert self.adapter is not None
        return self.adapter

    def model_available(self) -> bool:
        try:
            self.get_adapter()
        except (OSError, ValueError, TypeError, RuntimeError):
            return False
        return True

    def get_explainer(self) -> ShapExplainer:
        """Create the tree explainer once and reuse it across requests."""
        self.get_adapter()
        if self.model is None:
            raise TypeError("Explanations require a feature-based tree model")
        if self.explainer is None:
            with self.lock:
                if self.explainer is None:
                    self.explainer = ShapExplainer(self.model)
        return self.explainer


def _class_labels(adapter: BaseModelAdapter, probability_count: int) -> list[str]:
    labels = [str(label) for label in getattr(adapter, "classes", ())]
    if not labels and hasattr(adapter, "model"):
        model = getattr(adapter, "model")
        estimator = final_estimator(model)
        labels = [str(label) for label in getattr(estimator, "classes_", ())]
    if len(labels) != probability_count:
        labels = [f"class_{index}" for index in range(probability_count)]
    return labels


def _predict(
    adapter: BaseModelAdapter, sequences: list[str]
) -> tuple[list[str], list[PredictionReport], Any | None]:
    features = None
    if isinstance(adapter, FeatureModelAdapter):
        features = adapter.prepare(sequences)
        predictions = np.asarray(adapter.model.predict(features))
        probabilities = np.asarray(adapter.model.predict_proba(features), dtype=float)
    else:
        predictions = np.asarray(adapter.predict(sequences))
        probabilities = np.asarray(adapter.predict_proba(sequences), dtype=float)
    if predictions.shape != (len(sequences),):
        raise ValueError("Model returned an invalid prediction shape")
    if probabilities.ndim != 2 or probabilities.shape[0] != len(sequences):
        raise ValueError("Model returned an invalid probability shape")
    if not np.isfinite(probabilities).all():
        raise ValueError("Model returned non-finite probabilities")
    labels = _class_labels(adapter, probabilities.shape[1])
    reports = []
    for index, (sequence, prediction, row) in enumerate(
        zip(sequences, predictions, probabilities)
    ):
        reports.append(
            PredictionReport(
                index=index,
                sequence_length=len(sequence),
                prediction=str(prediction),
                confidence=float(np.max(row)),
                probabilities=[
                    ClassProbability(label=label, probability=float(value))
                    for label, value in zip(labels, row)
                ],
            )
        )
    return labels, reports, features


def create_app(
    model: Any | None = None,
    model_path: Path | None = None,
    *,
    use_default: bool = False,
) -> FastAPI:
    """Create the service using a model object or trusted server-side artifact path."""

    service = FastAPI(
        title="MAP-ExPLoc",
        version="0.1.0",
        description=(
            "Research-use protein subcellular localization predictions with "
            "feature-level SHAP explanations."
        ),
    )
    runtime = _ModelRuntime(model, model_path, use_default)

    def require_adapter() -> BaseModelAdapter:
        try:
            return runtime.get_adapter()
        except (OSError, ModelArtifactError, TypeError, RuntimeError) as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Prediction model is unavailable or incompatible. "
                    "Set MAPEXPLOC_MODEL_PATH to a trusted compatible artifact "
                    "or repair the repository default manifest."
                ),
            ) from exc

    @service.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        available = runtime.model_available()
        return HealthResponse(
            status="ready" if available else "model_unavailable",
            model_available=available,
            model_loaded=runtime.adapter is not None,
        )

    @service.get("/model", response_model=ModelResponse)
    def model_info() -> ModelResponse:
        adapter = require_adapter()
        labels = list(getattr(adapter, "classes", ()))
        if not labels and hasattr(adapter, "model"):
            labels = [str(item) for item in getattr(adapter.model, "classes_", ())]
        public_keys = {
            "name",
            "model_family",
            "evaluation_status",
            "experiment_id",
            "scope",
            "source",
            "release",
            "retrieved_at",
            "source_sha256",
            "dataset_sha256",
            "sample_count",
            "class_counts",
            "seed",
            "split",
            "evaluation",
            "runtime",
            "best_params",
            "best_cv_score",
            "limitations",
            "attribution",
            "software_versions",
            "model_id",
        }
        metadata = {
            key: value for key, value in runtime.metadata.items() if key in public_keys
        }
        return ModelResponse(
            model_classes=labels,
            feature_count=(
                len(FEATURE_NAMES) if isinstance(adapter, FeatureModelAdapter) else None
            ),
            metadata_available=bool(metadata),
            metadata=metadata,
        )

    @service.post("/features", response_model=FeaturesResponse)
    def sequence_features(request: PredictRequest) -> FeaturesResponse:
        matrix = build_feature_matrix(request.sequences)
        return FeaturesResponse(
            results=[
                SequenceFeatures(
                    index=index,
                    sequence_length=int(row["length"]),
                    composition={aa: float(row[f"aa_{aa}"]) for aa in AMINO_ACIDS},
                    gravy=float(row["gravy"]),
                    isoelectric_point=float(row["isoelectric_point"]),
                )
                for index, (_, row) in enumerate(matrix.iterrows())
            ]
        )

    @service.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        try:
            labels, reports, _ = _predict(require_adapter(), request.sequences)
        except ValueError as exc:
            raise HTTPException(
                status_code=500, detail="Model inference failed"
            ) from exc
        return PredictResponse(model_classes=labels, results=reports)

    @service.post("/explain", response_model=ExplainResponse)
    def explain(request: ExplainRequest) -> ExplainResponse:
        adapter = require_adapter()
        if runtime.model is None:
            raise HTTPException(
                status_code=501,
                detail="Explanations require a feature-based tree model",
            )
        try:
            labels, predictions, features = _predict(adapter, request.sequences)
            if features is None:
                raise TypeError("Explanations require a feature-based tree model")
            explanations = runtime.get_explainer().explain_predictions(
                features, top_n=request.top_n
            )
        except (ImportError, TypeError) as exc:
            raise HTTPException(
                status_code=501,
                detail="Explanations are unavailable for the configured model",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=500, detail="Model explanation failed"
            ) from exc

        reports = [
            ExplanationReport(
                **prediction.model_dump(),
                base_value=float(explanation["base_value"]),
                feature_contributions=[
                    FeatureContribution(**contribution)
                    for contribution in explanation["feature_contributions"]
                ],
            )
            for prediction, explanation in zip(predictions, explanations)
        ]
        return ExplainResponse(model_classes=labels, results=reports)

    return service


app = create_app(use_default=True)

__all__ = ["ExplainRequest", "PredictRequest", "app", "create_app"]
