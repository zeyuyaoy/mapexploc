"""Independent adapter package. Synthetic software control, not a pretrained model."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from mapexploc import AdapterDescriptor


class SyntheticLocalizationAdapter:
    descriptor = AdapterDescriptor(
        model_id="synthetic-position-control:v1",
        classes=("outside", "inside"),
        preprocessing_id="first-ten-leucine-frequency:v1",
        provenance={"purpose": "Software control only; no biological evidence"},
    )

    def predict_proba(self, batch: Sequence[str]) -> np.ndarray:
        probabilities = np.array([0.1 + 0.8 * s[:10].count("L") / 10 for s in batch])
        return np.column_stack([probabilities, 1 - probabilities])


def factory(configuration: dict[str, Any]) -> SyntheticLocalizationAdapter:
    if configuration:
        raise ValueError("The deterministic fixture takes no configuration")
    return SyntheticLocalizationAdapter()
