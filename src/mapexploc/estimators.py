"""Shared access to fitted feature classifiers, including legacy pipelines."""

from typing import Any


def final_estimator(model: Any) -> Any:
    """Return a pipeline's final estimator without relying on its step name."""
    steps = getattr(model, "steps", ())
    return steps[-1][1] if steps else model
