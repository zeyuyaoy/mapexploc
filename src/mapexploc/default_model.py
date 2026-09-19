"""Resolve an operator override or the trusted source-checkout model manifest."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .artifacts import ModelArtifact, ModelArtifactError, load_model_artifact


@dataclass(frozen=True)
class ModelSelection:
    path: Path
    sha256: str | None = None
    model_id: str | None = None

    def load(self) -> ModelArtifact:
        if self.sha256 is not None:
            actual = hashlib.sha256(self.path.read_bytes()).hexdigest()
            if actual != self.sha256:
                raise ModelArtifactError(
                    "Default model checksum does not match manifest"
                )
        artifact = load_model_artifact(self.path)
        if (
            self.model_id is not None
            and artifact.metadata.get("model_id") != self.model_id
        ):
            raise ModelArtifactError("Default model identity does not match manifest")
        return artifact


def source_root() -> Path | None:
    """Recognize this package's source checkout, never the caller's directory."""
    package = Path(__file__).resolve().parent
    root = package.parent.parent
    if package.parent.name == "src" and (root / "pyproject.toml").is_file():
        return root
    return None


def manifest_selection(root: Path) -> ModelSelection:
    root = root.resolve()
    try:
        manifest = json.loads((root / "config/default-model.json").read_text())
        if (
            type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 1
        ):
            raise ValueError("Unsupported default-model manifest version")
        artifact_path = manifest["artifact_path"]
        if not isinstance(artifact_path, str) or not artifact_path.strip():
            raise ValueError("Missing artifact path")
        relative = Path(artifact_path)
        path = (root / relative).resolve()
        digest = manifest["sha256"]
        identity = manifest["model_id"]
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_relative_to(root)
        ):
            raise ValueError("Default artifact must be inside the repository")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Invalid artifact checksum")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("Missing model identity")
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelArtifactError("Invalid default-model manifest") from exc
    return ModelSelection(path, digest, identity)


def resolve_default_model(*, trusted_root: Path | None = None) -> ModelSelection:
    """Resolve the research override, or an explicitly trusted deployment bundle.

    A deployment root is supplied by application code, never by an HTTP request
    or the working directory. It always uses the checked manifest and ignores
    the unrestricted research artifact override.
    """
    if trusted_root is not None:
        return manifest_selection(trusted_root)
    override = os.environ.get("MAPEXPLOC_MODEL_PATH")
    if override is not None:
        if not override.strip():
            raise ModelArtifactError("MAPEXPLOC_MODEL_PATH is empty")
        return ModelSelection(Path(override).expanduser().resolve())
    root = source_root()
    if root is None:
        raise ModelArtifactError(
            "No packaged model. Set MAPEXPLOC_MODEL_PATH to a trusted artifact."
        )
    return manifest_selection(root)
