"""Utilities for loading example datasets."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_example_dataset(path: Path) -> pd.DataFrame:
    """Load the example dataset shipped with the package.

    Parameters
    ----------
    path:
        Path to the CSV file containing sequences and labels.
    """
    logger.info("Loading dataset from %s", path)
    frame = pd.read_csv(path)
    required = {"sequence", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(sorted(missing))}"
        )
    if frame.empty:
        raise ValueError("Dataset must contain at least one protein sequence")
    if frame[list(required)].isna().any().any():
        raise ValueError("Dataset sequence and label values must not be missing")
    return frame


def iter_sequences(df: pd.DataFrame) -> Iterable[str]:
    """Yield sequences from a dataframe one by one."""
    if "sequence" not in df.columns:
        raise ValueError("Dataset is missing required column: sequence")
    for seq in df["sequence"]:
        yield str(seq)


__all__ = ["load_example_dataset", "iter_sequences"]
