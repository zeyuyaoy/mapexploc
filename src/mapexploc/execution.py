"""Operator-owned bounded inference caches and atomic restart checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .adapter import BaseModelAdapter, predict_probabilities
from .contracts import validate_probabilities


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def explanation_implementation() -> str:
    """Invalidate restart outputs across numerical code or dependency changes."""
    from importlib.metadata import version

    root = Path(__file__).parent
    files = (
        "analysis.py",
        "diagnostics.py",
        "explainers/regions.py",
        "explainers/references.py",
        "explainers/shap.py",
    )
    return fingerprint(
        {
            "code": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in files
            },
            "dependencies": {
                name: version(name)
                for name in ("numpy", "scipy", "shap", "scikit-learn")
            },
        }
    )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, allow_nan=False)
    fd, temporary = tempfile.mkstemp(prefix=".mapexploc-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class ExecutionOptions:
    cache_directory: Path | None = None
    restart_directory: Path | None = None
    max_cache_entries: int = 100000
    progress: Callable[[dict[str, Any]], None] | None = None


class CachedAdapter:
    """Cache exact complete probability vectors, with all declared runtime inputs."""

    def __init__(
        self, adapter: BaseModelAdapter, directory: Path, max_entries: int = 100000
    ):
        if max_entries < 1:
            raise ValueError("Cache capacity must be positive")
        if adapter.descriptor.checkpoint_sha256 is None:
            raise ValueError(
                "Persistent inference caching requires an immutable checkpoint"
                " fingerprint"
            )
        self.adapter = adapter
        self.descriptor = adapter.descriptor
        self.identity = fingerprint(self.descriptor.model_dump(mode="json"))
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "probabilities.sqlite3"
        self.max_entries = max_entries
        self.cache_hits = 0
        self.native_sequences = 0
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS probabilities (key TEXT PRIMARY KEY,"
                " payload TEXT NOT NULL, checksum TEXT NOT NULL, accessed REAL NOT"
                " NULL)"
            )

    def predict_proba(self, batch: Sequence[str]) -> np.ndarray:
        keys = [fingerprint([self.identity, s]) for s in batch]
        values = {}
        missing = {}
        with sqlite3.connect(self.path) as db:
            for key, s in zip(keys, batch):
                row = db.execute(
                    "SELECT payload,checksum FROM probabilities WHERE key=?", (key,)
                ).fetchone()
                if row:
                    if hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
                        raise ValueError(
                            "Corrupted probability cache entry; remove the cache before"
                            " retrying"
                        )
                    vector = validate_probabilities(
                        self.descriptor, [json.loads(row[0])], 1
                    )[0]
                    values[key] = vector
                    self.cache_hits += 1
                    db.execute(
                        "UPDATE probabilities SET accessed=? WHERE key=?",
                        (time.time(), key),
                    )
                else:
                    missing[key] = s
        if missing:
            predicted = predict_probabilities(self.adapter, list(missing.values()))
            self.native_sequences += len(missing)
            with sqlite3.connect(self.path) as db:
                for key, vector in zip(missing, predicted):
                    payload = json.dumps(vector.tolist(), allow_nan=False)
                    db.execute(
                        "INSERT OR REPLACE INTO probabilities VALUES (?,?,?,?)",
                        (
                            key,
                            payload,
                            hashlib.sha256(payload.encode()).hexdigest(),
                            time.time(),
                        ),
                    )
                    values[key] = vector
                excess = (
                    db.execute("SELECT COUNT(*) FROM probabilities").fetchone()[0]
                    - self.max_entries
                )
                if excess > 0:
                    db.execute(
                        "DELETE FROM probabilities WHERE key IN (SELECT key FROM"
                        " probabilities ORDER BY accessed,key LIMIT ?)",
                        (excess,),
                    )
        return np.stack([values[key] for key in keys])


class RestartStore:
    def __init__(self, directory: Path, identity: dict[str, Any]) -> None:
        self.directory = directory
        manifest = directory / "manifest.json"
        expected = {"schema": 1, "identity": identity, "sha256": fingerprint(identity)}
        if manifest.exists():
            if json.loads(manifest.read_text()) != expected:
                raise ValueError(
                    "Restart manifest does not match the model, proteins or full"
                    " analysis configuration; use a new directory"
                )
        else:
            atomic_json(manifest, expected)
        self.identity = expected["sha256"]

    def load(self, protein_id: str) -> dict[str, Any] | None:
        path = self.directory / (fingerprint(protein_id) + ".json")
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if data.get("identity") != self.identity or data.get("checksum") != fingerprint(
            data.get("explanation")
        ):
            raise ValueError("Corrupted explanation restart checkpoint")
        return dict(data["explanation"])

    def save(self, protein_id: str, explanation: dict[str, Any]) -> None:
        atomic_json(
            self.directory / (fingerprint(protein_id) + ".json"),
            dict(
                identity=self.identity,
                checksum=fingerprint(explanation),
                explanation=explanation,
            ),
        )


def estimate_work(
    region_count: int, references: int, budget: int, diagnostics: bool = False
) -> dict[str, Any]:
    nontrivial = min(budget, 2**region_count - 2)
    evaluations = (nontrivial + 1) * references + 1
    return {
        "region_count": region_count,
        "nontrivial_coalitions": nontrivial,
        "reference_sequences": references,
        "base_native_evaluations_upper_bound": evaluations,
        "diagnostic_work_included": False,
        "diagnostics_requested": diagnostics,
        "estimated_seconds": None,
        "status": (
            "No calibrated mode/device timing model; sequence-evaluation count only"
        ),
    }
