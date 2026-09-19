"""Persistent isolated adapter for both licensed native DeepLoc 2.1 modes."""

from __future__ import annotations

import selectors
import threading
import weakref
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import json
import numpy as np
import subprocess
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import AdapterDescriptor, validate_batch, validate_probabilities


class DeepLocConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    python: str
    package_root: str
    torch_home: str
    mode: Literal["fast", "accurate"] = "fast"
    prott5_snapshot: str | None = None
    prott5_revision: str = "973be27c52ee6474de9c945952a8008aeb2a1a73"
    device: str = "cpu"
    batch_size: int = Field(default=8, ge=1, le=128)
    threads: int = Field(default=4, ge=1, le=32)
    timeout_seconds: float = Field(default=1800, gt=0)
    expected_checkpoint_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

    @model_validator(mode="after")
    def native_mode(self) -> "DeepLocConfiguration":
        if self.mode == "accurate":
            if not self.prott5_snapshot:
                raise ValueError("Accurate requires an offline pinned prott5_snapshot")
            if "batch_size" not in self.model_fields_set:
                self.batch_size = 1
            if self.batch_size != 1:
                raise ValueError(
                    "Accurate requires batch_size=1 because native pooling includes"
                    " padding"
                )
        return self


_RESIDENCY_LOCK = threading.RLock()
_ACTIVE: weakref.ReferenceType[DeepLocAdapter] | None = None


class DeepLocAdapter:
    """Run native preprocessing and inference in a persistent worker.

    Configuration is trusted operator input, never HTTP input.
    Use a context manager or call close() after analysis."""

    def __init__(self, configuration: DeepLocConfiguration):
        self.configuration = configuration
        self.lock = threading.Lock()
        self.measurements: list[dict[str, Any]] = []
        self.process = subprocess.Popen(
            [
                configuration.python,
                "-u",
                str(Path(__file__).parent / "workers/deeploc_worker.py"),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            response = self._exchange(configuration.model_dump())
            self.descriptor = AdapterDescriptor.model_validate(response["descriptor"])
            expected = configuration.expected_checkpoint_sha256
            if expected is not None and self.descriptor.checkpoint_sha256 != expected:
                raise ValueError(
                    "DeepLoc asset fingerprint differs from the configured checkpoint"
                )
        except Exception:
            self.close()
            raise

    @classmethod
    def from_config(cls, configuration: Mapping[str, Any]) -> DeepLocAdapter:
        return cls(DeepLocConfiguration.model_validate(configuration))

    def _exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        if (
            self.process.poll() is not None
            or self.process.stdin is None
            or self.process.stdout is None
        ):
            raise RuntimeError("DeepLoc worker is unavailable")
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(self.configuration.timeout_seconds):
                self.close()
                raise TimeoutError(
                    "DeepLoc worker timed out; no partial report was emitted"
                )
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(
                "DeepLoc worker exited before returning a result; inspect native stderr"
            )
        result = json.loads(line)
        if "error" in result:
            self.close()
            raise RuntimeError(f"DeepLoc inference failed: {result['error']}")
        return dict(result)

    def predict_proba(self, batch: Sequence[str]) -> np.ndarray:
        global _ACTIVE
        sequences = validate_batch(self.descriptor, batch)
        with _RESIDENCY_LOCK, self.lock:
            previous = _ACTIVE() if _ACTIVE else None
            if (
                previous is not None
                and previous is not self
                and previous.process.poll() is None
            ):
                previous._exchange({"operation": "unload"})
                _ACTIVE = None
            response = self._exchange({"sequences": sequences})
            _ACTIVE = weakref.ref(self)
            if "measurement" in response:
                self.measurements.append(response["measurement"])
        return validate_probabilities(
            self.descriptor, response["probabilities"], len(sequences)
        )

    def close(self) -> None:
        if self.process.poll() is None:
            if self.process.stdin:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        if self.process.stdout:
            self.process.stdout.close()

    @property
    def is_resident(self) -> bool:
        """Whether this process owns the currently loaded native backbone."""
        with _RESIDENCY_LOCK:
            return bool(_ACTIVE and _ACTIVE() is self and self.process.poll() is None)

    def __enter__(self) -> DeepLocAdapter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
