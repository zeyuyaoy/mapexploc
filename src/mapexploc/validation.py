"""Explicit grouped cross-validation shared by scientific training entry points."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def grouped_splits(
    labels: Any, groups: Any, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fail closed when independent, all-class validation folds are impossible."""
    y, g = np.asarray(labels), np.asarray(groups)
    if y.ndim != 1 or g.shape != y.shape or pd.isna(g).any():
        raise ValueError("Groups must be a nonmissing vector aligned to labels")
    if n_splits < 2 or len(np.unique(g)) < n_splits:
        raise ValueError(
            "At least two folds and enough independent groups are required"
        )
    expected = set(y)
    result = list(
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y, g
        )
    )
    for fit, valid in result:
        if set(g[fit]) & set(g[valid]):
            raise ValueError("Cross-validation group leakage")
        if set(y[fit]) != expected or set(y[valid]) != expected:
            raise ValueError("Every grouped fold must contain all classes")
    return result
