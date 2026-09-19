"""Public interpretation corrections, separate from immutable scientific artifacts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def public_provenance(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    if str(result.get("model_id", "")).startswith("human-"):
        result["interpretation_revision"] = "structured-annotation-policy:1"
        result["original_limitations"] = result.get("limitations")
        result["limitations"] = (
            "Human canonical proteins with one mapped structured localization and "
            "experimental localization evidence. Free-text notes may describe "
            "additional localizations; a selected label does not establish exclusive "
            "localization. Localization evidence does not establish experimental "
            "evidence for positional sequence features. Probabilities are "
            "uncalibrated; class-balanced curation is not natural prevalence. "
            "Grouping at 30% identity and 80% bidirectional coverage does not "
            "exclude all remote or domain-level homology. SHAP describes model "
            "behavior, not biological causality."
        )
    if result.get("evaluation"):
        result.setdefault("evaluation_status", "historical_holdout")
    return result
