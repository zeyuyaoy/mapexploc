"""Public sequence adapter contract and installed adapter discovery."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from .contracts import AdapterDescriptor, validate_batch, validate_probabilities
from .estimators import final_estimator
from .features import build_feature_matrix


@runtime_checkable
class BaseModelAdapter(Protocol):
    """Only a descriptor and batch probability inference are required.

    Native tokenization, transformations and calibration belong inside inference.
    Embeddings are optional implementation details, not biological attributions.
    """

    @property
    def descriptor(self) -> AdapterDescriptor: ...

    def predict_proba(self, batch: Sequence[str]) -> np.ndarray: ...


class _SimpleAdapter:
    """Compatibility wrapper for legacy sequence-in classifiers."""

    def __init__(self, model: Any):
        self.model = model
        labels = getattr(model, "classes", getattr(model, "classes_", ()))
        self.descriptor = AdapterDescriptor(
            model_id=f"unversioned:{type(model).__module__}.{type(model).__name__}",
            classes=tuple(str(label) for label in labels),
            preprocessing_id="legacy-sequence-adapter:unspecified",
            task_type=getattr(model, "task_type", "multiclass"),
            thresholds=getattr(model, "thresholds", None),
            provenance={
                "warning": "Legacy model has no checkpoint/preprocessing provenance"
            },
        )

    @property
    def classes(self) -> tuple[str, ...]:
        return self.descriptor.classes

    def predict(self, batch: Sequence[str]) -> np.ndarray:
        if callable(getattr(self.model, "predict", None)):
            return np.asarray(self.model.predict(batch))
        probabilities = predict_probabilities(self, batch)
        return np.asarray(self.classes)[probabilities.argmax(axis=1)]

    def predict_proba(self, batch: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.predict_proba(batch))

    def embed(self, batch: Sequence[str]) -> np.ndarray | None:
        value = self.model.embed(batch) if hasattr(self.model, "embed") else None
        return None if value is None else np.asarray(value)


class FeatureModelAdapter:
    """The original 423-feature classifier, behind the same public contract."""

    def __init__(
        self,
        model: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
        checkpoint_sha256: str | None = None,
    ):
        if not callable(getattr(model, "predict_proba", None)):
            raise TypeError("Feature model does not expose predict_proba")
        self.model = model
        estimator = final_estimator(model)
        self.descriptor = AdapterDescriptor(
            model_id=str(
                (metadata or {}).get(
                    "model_id", f"unversioned:{type(estimator).__name__}"
                )
            ),
            classes=tuple(str(label) for label in getattr(model, "classes_", ())),
            preprocessing_id="mapexploc:engineered-423:v1",
            capabilities=(
                ("tree", "region_kernel")
                if isinstance(estimator, (RandomForestClassifier, ExtraTreesClassifier))
                else ("region_kernel",)
            ),
            checkpoint_sha256=checkpoint_sha256,
            provenance=dict(metadata or {}),
        )

    @property
    def classes(self) -> tuple[str, ...]:
        return self.descriptor.classes

    def prepare(self, batch: Sequence[str]) -> pd.DataFrame:
        return build_feature_matrix(validate_batch(self.descriptor, batch))

    def predict(self, batch: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.predict(self.prepare(batch)))

    def predict_proba(self, batch: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.predict_proba(self.prepare(batch)))

    def embed(self, batch: Sequence[str]) -> np.ndarray:
        return np.asarray(self.prepare(batch).to_numpy(), dtype=float)


def load_adapter(model: Any) -> BaseModelAdapter:
    """Normalize a public adapter or an explicitly labeled legacy sequence model."""
    if isinstance(model, BaseModelAdapter):
        if not isinstance(model.descriptor, AdapterDescriptor):
            raise TypeError("Adapter descriptor must be an AdapterDescriptor")
        return model
    if callable(getattr(model, "predict_proba", None)):
        return _SimpleAdapter(model)
    raise TypeError("Model does not expose predict_proba")


def predict_probabilities(
    adapter: BaseModelAdapter, batch: Sequence[str]
) -> np.ndarray:
    sequences = validate_batch(adapter.descriptor, batch)
    return validate_probabilities(
        adapter.descriptor, adapter.predict_proba(sequences), len(sequences)
    )


def adapter_from_artifact(path: Path) -> FeatureModelAdapter:
    """Load a trusted original artifact without changing its bytes or identity."""
    from .artifacts import load_model_artifact
    from .provenance import file_sha256, public_provenance

    artifact = load_model_artifact(path)
    return FeatureModelAdapter(
        artifact.model,
        metadata=public_provenance(artifact.metadata),
        checkpoint_sha256=file_sha256(path),
    )


def registered_adapter(
    name: str, configuration: Mapping[str, Any] | None = None
) -> BaseModelAdapter:
    """Operator-only registry. Factories receive one JSON configuration mapping.

    Third-party packages publish the ``mapexploc.adapters`` entry-point group.
    Arbitrary Python imports and remote factory URLs are not supported.
    """
    config = dict(configuration or {})
    if name == "feature_artifact":
        if set(config) != {"path"}:
            raise ValueError("feature_artifact requires exactly one trusted path")
        return adapter_from_artifact(Path(config["path"]))
    if name == "deeploc2":
        from .deeploc import DeepLocAdapter

        return DeepLocAdapter.from_config(config)
    candidates = list(entry_points(group="mapexploc.adapters", name=name))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one installed adapter named {name!r}; found {len(candidates)}"
        )
    return load_adapter(candidates[0].load()(config))


__all__ = [
    "AdapterDescriptor",
    "BaseModelAdapter",
    "FeatureModelAdapter",
    "load_adapter",
    "predict_probabilities",
    "registered_adapter",
    "adapter_from_artifact",
]
