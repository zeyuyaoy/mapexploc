"""Configuration utilities using pydantic and YAML."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Hyperparameters for the RandomForest model."""

    n_estimators: int = Field(default=100, gt=0)
    max_depth: int | None = Field(default=None, gt=0)


class Settings(BaseModel):
    """Application settings."""

    seed: int = 42
    model: ModelConfig = Field(default_factory=ModelConfig)


def set_seed(seed: int) -> None:
    """Seed ``random`` and ``numpy``; estimator and runtime controls remain separate.

    Parameters
    ----------
    seed:
        The random seed to use.
    """
    logger.debug("Setting random seed to %s", seed)
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: Path) -> Settings:
    """Load a :class:`Settings` object from a YAML file.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.
    """
    logger.info("Loading configuration from %s", path)
    with path.open("r", encoding="utf8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = Settings.model_validate(data)
    set_seed(cfg.seed)
    return cfg


__all__ = ["Settings", "ModelConfig", "load_config", "set_seed"]
