"""Research inference API with input validation and model-level explanations."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .adapter import (
    BaseModelAdapter,
    FeatureModelAdapter,
    load_adapter,
    predict_probabilities,
)
from .analysis import run_analysis
from .annotations import Annotation
from .artifacts import ModelArtifactError, load_model_artifact
from .catalog import ConfiguredModel
from .contracts import class_decisions, validate_batch, validate_probabilities
from .default_model import ModelSelection, resolve_default_model
from .estimators import final_estimator
from .explainers.shap import ShapExplainer
from .features import (
    AMINO_ACIDS,
    FEATURE_NAMES,
    build_feature_matrix,
    normalize_protein_sequence,
)
from .methods import MethodConfiguration
from .provenance import public_provenance
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
from .report_v2 import AnalysisConfiguration, AnalysisReport, Protein
from .report_v3 import AnalysisReportV3

MAX_BATCH_SIZE = 100
MAX_BATCH_RESIDUES = 1_000_000
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RequestLimits:
    """Per-application limits; offline analysis and CLI contracts are unchanged."""

    prediction_proteins: int = MAX_BATCH_SIZE
    prediction_residues: int = MAX_BATCH_RESIDUES
    explanation_proteins: int = MAX_BATCH_SIZE
    explanation_residues: int = MAX_BATCH_RESIDUES
    analysis_proteins: int = 20
    analysis_residues: int = 20_000

    def check(self, sequences: list[str], operation: str) -> None:
        proteins = getattr(self, f"{operation}_proteins")
        residues = getattr(self, f"{operation}_residues")
        if len(sequences) > proteins or sum(map(len, sequences)) > residues:
            raise HTTPException(
                413,
                f"Live {operation} requests accept at most {proteins} proteins "
                f"and {residues:,} residues; use the CLI for larger cohorts",
            )


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


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    adapter_id: str = "default"
    proteins: list[Protein] = Field(min_length=1, max_length=100)
    configuration: AnalysisConfiguration | MethodConfiguration = Field(
        default_factory=AnalysisConfiguration
    )
    expected_model_id: str | None = None
    expected_checkpoint_sha256: str | None = None
    annotations: list[Annotation] = Field(default_factory=list, max_length=10000)

    @model_validator(mode="after")
    def bounded(self) -> AnalyzeRequest:
        if sum(len(p.sequence) for p in self.proteins) > MAX_BATCH_RESIDUES:
            raise ValueError("Batch exceeds the residue limit")
        if len({p.protein_id for p in self.proteins}) != len(self.proteins):
            raise ValueError("Protein identifiers must be unique")
        return self


class _ModelRuntime:
    def __init__(
        self,
        model: Any | None,
        model_path: Path | None,
        use_default: bool = False,
        model_selection: ModelSelection | None = None,
    ):
        self.use_default = use_default
        self.model_path = model_path
        self.model_selection = model_selection
        self.model: Any | None = None
        self.adapter: BaseModelAdapter | None = None
        self.explainer: ShapExplainer | None = None
        self.metadata: dict[str, Any] = {}
        self.lock = Lock()
        if model is not None:
            self._set_model(model)

    def _set_model(self, model: Any) -> None:
        if isinstance(model, FeatureModelAdapter):
            self.model = model.model
            self.metadata = dict(model.descriptor.provenance)
            self.adapter = model
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
            selection = self.model_selection
            if selection is not None:
                artifact = selection.load()
            elif self.model_path is not None:
                artifact = load_model_artifact(self.model_path)
            elif self.use_default:
                selection = resolve_default_model()
                artifact = selection.load()
            else:
                raise RuntimeError("No model is configured")
            self.metadata = public_provenance(artifact.metadata)
            if self.metadata.get("model_id"):
                self.metadata.setdefault(
                    "model_family", type(final_estimator(artifact.model)).__name__
                )
            if self.metadata.get("evaluation"):
                self.metadata.setdefault("evaluation_status", "historical_holdout")
            from .provenance import file_sha256

            artifact_path = selection.path if selection is not None else self.model_path
            assert artifact_path is not None
            self._set_model(
                FeatureModelAdapter(
                    artifact.model,
                    metadata=self.metadata,
                    checkpoint_sha256=file_sha256(artifact_path),
                )
            )
        assert self.adapter is not None
        return self.adapter

    def model_available(self) -> bool:
        try:
            self.get_adapter()
        except (OSError, ValueError, TypeError, RuntimeError):
            logger.exception("Model unavailable", extra={"event": "model_unavailable"})
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
    labels = list(adapter.descriptor.classes)
    if len(labels) != probability_count:
        raise ValueError("Probability columns do not match declared classes")
    return labels


def _predict(
    adapter: BaseModelAdapter, sequences: list[str]
) -> tuple[list[str], list[PredictionReport], Any | None]:
    if adapter.descriptor.task_type != "multiclass":
        raise ValueError("Multilabel models require the v2 prediction contract")
    sequences = validate_batch(adapter.descriptor, sequences)
    features = None
    predictions: np.ndarray | None
    if isinstance(adapter, FeatureModelAdapter):
        features = adapter.prepare(sequences)
        predictions = np.asarray(adapter.model.predict(features))
        probabilities = np.asarray(adapter.model.predict_proba(features), dtype=float)
    else:
        native_predict = getattr(adapter, "predict", None)
        predictions = (
            np.asarray(native_predict(sequences)) if callable(native_predict) else None
        )
        probabilities = np.asarray(adapter.predict_proba(sequences), dtype=float)
    probabilities = validate_probabilities(
        adapter.descriptor, probabilities, len(sequences)
    )
    labels = _class_labels(adapter, probabilities.shape[1])
    if predictions is None:
        predictions = np.asarray(labels)[probabilities.argmax(axis=1)]
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
        if str(prediction) not in labels:
            raise ValueError("Model predicted an undeclared class")
        if str(prediction) != labels[int(np.argmax(row))]:
            raise ValueError(
                "Multiclass prediction disagrees with declared argmax decision"
            )
        reports.append(
            PredictionReport(
                index=index,
                sequence_length=len(sequence),
                prediction=str(prediction),
                confidence=float(row[labels.index(str(prediction))]),
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
    adapters: dict[str, BaseModelAdapter] | None = None,
    adapter_configurations: dict[str, dict[str, Any]] | None = None,
    model_selection: ModelSelection | None = None,
    limits: RequestLimits = RequestLimits(),
    analysis_policy: Callable[[AnalyzeRequest], None] | None = None,
) -> FastAPI:
    """Create the service using a model object or trusted server-side artifact path."""

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            for registration in registered.values():
                registration.close()

    service = FastAPI(
        lifespan=lifespan,
        title="MAP-ExPLoc",
        version="1.0.0",
        description=(
            "Research-use protein subcellular localization predictions with "
            "feature-level SHAP explanations."
        ),
    )
    runtime = _ModelRuntime(model, model_path, use_default, model_selection)

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
        labels = list(adapter.descriptor.classes)
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
            "interpretation_revision",
            "original_limitations",
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
        limits.check(request.sequences, "prediction")
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
        limits.check(request.sequences, "prediction")
        try:
            labels, reports, _ = _predict(require_adapter(), request.sequences)
        except ValueError as exc:
            raise HTTPException(
                status_code=500, detail="Model inference failed"
            ) from exc
        return PredictResponse(model_classes=labels, results=reports)

    @service.post("/explain", response_model=ExplainResponse)
    def explain(request: ExplainRequest) -> ExplainResponse:
        limits.check(request.sequences, "explanation")
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
                remainder=float(explanation["remainder"]),
                feature_contributions=[
                    FeatureContribution(**contribution)
                    for contribution in explanation["feature_contributions"]
                ],
            )
            for prediction, explanation in zip(predictions, explanations)
        ]
        return ExplainResponse(model_classes=labels, results=reports)

    configured_adapters = {
        name: load_adapter(value) for name, value in (adapters or {}).items()
    }
    registered = {
        name: ConfiguredModel(value["factory"], value["configuration"])
        for name, value in (adapter_configurations or {}).items()
    }
    if set(registered) & (set(configured_adapters) | {"default"}):
        raise ValueError("Configured model identifiers must be unique")

    def v2_adapter(identifier: str) -> BaseModelAdapter:
        if identifier == "default":
            return require_adapter()
        if identifier in registered:
            try:
                return registered[identifier].get()
            except (ValueError, RuntimeError, OSError) as exc:
                raise HTTPException(503, str(exc)) from exc
        if identifier not in configured_adapters:
            raise HTTPException(404, "Unknown configured adapter identifier")
        return configured_adapters[identifier]

    @service.get("/v2/models")
    def v2_models() -> dict[str, Any]:
        available = {
            name: value.descriptor.model_dump(mode="json")
            for name, value in configured_adapters.items()
        }
        if runtime.model_available():
            available["default"] = runtime.get_adapter().descriptor.model_dump(
                mode="json"
            )
        return {"schema_version": 2, "adapters": available}

    @service.get("/v3/models")
    def model_catalogue() -> dict[str, Any]:
        models = [
            registration.summary(name) for name, registration in registered.items()
        ]
        for name, descriptor in v2_models()["adapters"].items():
            models.append(
                dict(
                    adapter_id=name,
                    model_id=descriptor["model_id"],
                    descriptor=descriptor,
                    mode=descriptor.get("provenance", {}).get("mode"),
                    readiness="configured",
                    issues=[],
                    execution=(
                        "bounded_live"
                        if "tree" in descriptor["capabilities"]
                        else "cli_explanations"
                    ),
                    expected_checkpoint_sha256=descriptor.get("checkpoint_sha256"),
                )
            )
        return {
            "schema_version": 3,
            "models": models,
            "default_method_profile": "v2x-legacy",
            "replacement_default_qualified": False,
        }

    @service.post("/v3/predict")
    @service.post("/v2/predict")
    def v2_predict(request: AnalyzeRequest) -> dict[str, Any]:
        limits.check([p.sequence for p in request.proteins], "prediction")
        adapter = v2_adapter(request.adapter_id)
        if (
            request.expected_model_id
            and request.expected_model_id != adapter.descriptor.model_id
        ) or (
            request.expected_checkpoint_sha256
            and request.expected_checkpoint_sha256
            != adapter.descriptor.checkpoint_sha256
        ):
            raise HTTPException(
                409, "Selected model identity changed; refresh the model catalogue"
            )
        try:
            probabilities = predict_probabilities(
                adapter, [p.sequence for p in request.proteins]
            )
        except (ValueError, TypeError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return {
            "schema_version": 2,
            "model": adapter.descriptor.model_dump(mode="json"),
            "results": [
                {
                    "protein": protein.model_dump(),
                    "probabilities": row.tolist(),
                    "decisions": decisions,
                }
                for protein, row, decisions in zip(
                    request.proteins,
                    probabilities,
                    class_decisions(adapter.descriptor, probabilities),
                )
            ],
        }

    @service.post("/v3/analyze", response_model=AnalysisReportV3 | AnalysisReport)
    @service.post("/v2/analyze", response_model=AnalysisReportV3 | AnalysisReport)
    def v2_analyze(request: AnalyzeRequest) -> AnalysisReport | AnalysisReportV3:
        limits.check([p.sequence for p in request.proteins], "analysis")
        if analysis_policy is not None:
            analysis_policy(request)
        adapter = v2_adapter(request.adapter_id)
        method = request.configuration.explainer
        if method == "region_kernel" or "tree" not in adapter.descriptor.capabilities:
            raise HTTPException(
                409,
                "Sequence-region analyses run through the analyze CLI; "
                "import the completed report in the viewer",
            )
        try:
            return run_analysis(
                adapter,
                request.proteins,
                request.configuration,
                request.annotations,
                tree_explainer=(
                    runtime.get_explainer() if adapter is runtime.adapter else None
                ),
            )
        except (ValueError, TypeError, RuntimeError) as exc:
            raise HTTPException(
                422, "Analysis does not satisfy the scientific report contract"
            ) from exc

    return service


def application_from_environment() -> FastAPI:
    """Catalogue path is operator environment configuration, never HTTP input."""
    path = os.environ.get("MAPEXPLOC_ADAPTER_CATALOG")
    configuration = json.loads(Path(path).read_text()) if path else None
    return create_app(use_default=True, adapter_configurations=configuration)


@lru_cache(maxsize=1)
def _environment_app() -> FastAPI:
    return application_from_environment()


app: FastAPI  # Resolved lazily by __getattr__; annotation does not initialize it.


def __getattr__(name: str) -> Any:
    # Preserve `uvicorn mapexploc.api:app` and `from mapexploc.api import app`
    # without configuring research adapters merely by importing shared code.
    if name == "app":
        return _environment_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ExplainRequest", "PredictRequest", "app", "create_app"]
