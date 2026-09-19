"""Trusted operator model configurations; listing never constructs a backbone."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import json
import subprocess

from .adapter import BaseModelAdapter, registered_adapter
from .deeploc import DeepLocConfiguration


class ConfiguredModel:
    def __init__(self, factory: str, configuration: dict[str, Any]):
        self.factory = factory
        self.configuration = configuration
        self.adapter: BaseModelAdapter | None = None
        self.lock = threading.Lock()
        if factory == "deeploc2":
            DeepLocConfiguration.model_validate(configuration)

    def summary(self, identifier: str) -> dict[str, Any]:
        mode = (
            self.configuration.get("mode", "fast")
            if self.factory == "deeploc2"
            else None
        )
        readiness = "configured"
        issues = []
        resources: dict[str, Any] = {"ready": None, "qualification": "Not inspected"}
        if self.factory == "deeploc2":
            for name in ("python", "package_root", "torch_home"):
                if not Path(self.configuration[name]).exists():
                    issues.append(f"Missing operator-configured {name}")
            if (
                mode == "accurate"
                and not (
                Path(self.configuration["prott5_snapshot"]) / "pytorch_model.bin"
            ).is_file()
            ):
                issues.append("Pinned ProtT5 checkpoint is not installed")
            if issues:
                readiness = "unavailable"
            elif getattr(self.adapter, "is_resident", False):
                resources = {
                    "ready": True,
                    "qualification": (
                        "Backbone already resident; per-batch resource checks still"
                        " apply. No second cold-load allocation is required."
                    ),
                }
            else:
                try:
                    checked = subprocess.run(
                        [
                            self.configuration["python"],
                            str(Path(__file__).parent / "workers/deeploc_worker.py"),
                            "--resources",
                        ],
                        input=json.dumps(self.configuration),
                        capture_output=True,
                        text=True,
                        timeout=20,
                        check=True,
                    )
                    resources = json.loads(checked.stdout)
                    if not resources["ready"]:
                        readiness = "unavailable"
                        issues.append(
                            resources.get("reason", "Native resource preflight failed")
                        )
                except (OSError, ValueError, subprocess.SubprocessError):
                    readiness = "unavailable"
                    issues.append(
                        "Could not inspect the configured native runtime; check its"
                        " Python and dependencies"
                    )
        descriptor = (
            self.adapter.descriptor.model_dump(mode="json") if self.adapter else None
        )
        return {
            "adapter_id": identifier,
            "factory": self.factory,
            "mode": mode,
            "model_id": (
                f"DeepLoc-2.1-{mode.title()}"
                if mode
                else (descriptor or {}).get("model_id", identifier)
            ),
            "expected_checkpoint_sha256": self.configuration.get(
                "expected_checkpoint_sha256"
            ),
            "readiness": readiness,
            "resources": resources,
            "issues": issues,
            "descriptor": descriptor,
            "execution": "cli_explanations" if mode else "bounded_live",
            "device": self.configuration.get("device", "adapter-defined"),
            "tradeoff": (
                "ProtT5; substantially higher compute; potentially better predictive"
                " performance"
                if mode == "accurate"
                else (
                    "Lower compute; higher throughput"
                    if mode == "fast"
                    else "Configured external model"
                )
            ),
            "qualification": "not_established_by_catalogue",
            "estimate": {
                "seconds": None,
                "status": "Timing requires measurements on the selected device",
            },
        }

    def get(self) -> BaseModelAdapter:
        with self.lock:
            process = getattr(self.adapter, "process", None)
            if process is not None and process.poll() is not None:
                self.adapter = None
            if self.adapter is None:
                self.adapter = registered_adapter(self.factory, self.configuration)
            return self.adapter

    def close(self) -> None:
        """Release only the worker owned by this configured registration."""
        with self.lock:
            close = getattr(self.adapter, "close", None)
            if callable(close):
                close()
            self.adapter = None
