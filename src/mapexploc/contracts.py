"""Public, versioned contracts for sequence-in localization predictors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .features import MAX_SEQUENCE_LENGTH, normalize_protein_sequence


class AdapterDescriptor(BaseModel):
    """Class columns and native decisions are part of the model, never guessed."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    contract_version: Literal[1] = 1
    model_id: str = Field(min_length=1)
    classes: tuple[str, ...] = Field(min_length=1)
    task_type: Literal["multiclass", "multilabel"] = "multiclass"
    preprocessing_id: str = Field(min_length=1)
    min_length: int = Field(default=1, ge=1)
    max_length: int = Field(default=MAX_SEQUENCE_LENGTH, ge=1)
    thresholds: tuple[float, ...] | None = None
    threshold_comparison: Literal["greater", "greater_equal"] = "greater_equal"
    empty_decision_policy: Literal["empty", "nearest_threshold"] = "empty"
    fallback_round_decimals: int | None = Field(default=None, ge=0, le=15)
    capabilities: tuple[Literal["tree", "region_kernel"], ...] = ("region_kernel",)
    checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_contract(self) -> AdapterDescriptor:
        if len(set(self.classes)) != len(self.classes) or any(
            not label.strip() for label in self.classes
        ):
            raise ValueError("Class identifiers must be nonempty and unique")
        if self.max_length < self.min_length:
            raise ValueError("Invalid sequence length interval")
        if self.task_type == "multilabel":
            if self.thresholds is None or len(self.thresholds) != len(self.classes):
                raise ValueError("Multilabel models must declare native thresholds")
            if not all(np.isfinite(t) and 0 <= t <= 1 for t in self.thresholds):
                raise ValueError("Thresholds must be finite probabilities")
        elif self.thresholds is not None:
            raise ValueError("Multiclass decisions use argmax, not thresholds")
        return self


def validate_batch(descriptor: AdapterDescriptor, batch: Sequence[str]) -> list[str]:
    if not batch:
        raise ValueError("At least one protein is required")
    sequences = [normalize_protein_sequence(s) for s in batch]
    if any(
        not descriptor.min_length <= len(s) <= descriptor.max_length for s in sequences
    ):
        raise ValueError(
            f"{descriptor.model_id} accepts {descriptor.min_length}–"
            f"{descriptor.max_length} residues; sequences are never silently truncated"
        )
    return sequences


def validate_probabilities(
    descriptor: AdapterDescriptor, values: Any, count: int
) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.shape != (count, len(descriptor.classes)):
        raise ValueError(
            "Probability shape does not match proteins and declared classes"
        )
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Model probabilities must be finite and between zero and one")
    if descriptor.task_type == "multiclass" and not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0
    ):
        raise ValueError("Multiclass probabilities must sum to one")
    return probabilities


def class_decisions(
    descriptor: AdapterDescriptor, probabilities: np.ndarray
) -> list[list[str]]:
    if descriptor.task_type == "multiclass":
        return [[descriptor.classes[i]] for i in probabilities.argmax(axis=1)]
    thresholds = np.asarray(descriptor.thresholds)
    selected = (
        probabilities > thresholds
        if descriptor.threshold_comparison == "greater"
        else probabilities >= thresholds
    )
    decisions = [
        [label for label, keep in zip(descriptor.classes, row) if keep]
        for row in selected
    ]
    if descriptor.empty_decision_policy == "nearest_threshold":
        rounded = (
            probabilities
            if descriptor.fallback_round_decimals is None
            else np.around(probabilities, descriptor.fallback_round_decimals)
        )
        for index, labels in enumerate(decisions):
            if not labels:
                labels.append(
                    descriptor.classes[np.argsort(rounded[index] - thresholds)[-1]]
                )
    return decisions
