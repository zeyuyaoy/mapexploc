"""Fail CI when required reference assets are absent or their hashes differ."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import json


def required_file(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe asset path: {name}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or escaping asset: {root / name}")
    return path


def verify_hash(path: Path, expected: str) -> None:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Invalid SHA-256 for {path}")
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Checksum mismatch: {path}")


def verify_manifest(root: Path, name: str) -> None:
    manifest = json.loads(required_file(root, name).read_text())
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError(f"Empty or invalid manifest: {root / name}")
    for relative, expected in manifest.items():
        verify_hash(required_file(root, relative), expected)


def verify_assets(root: Path) -> None:
    default = json.loads(required_file(root, "config/default-model.json").read_text())
    verify_hash(required_file(root, default["artifact_path"]), default["sha256"])
    baseline = json.loads(
        required_file(root, "examples/baseline/manifest.json").read_text()
    )
    report = json.loads(
        required_file(root, "examples/models/human-baseline.report.json").read_text()
    )
    dataset = required_file(root, "examples/baseline/dataset.csv")
    verify_hash(dataset, baseline["dataset_sha256"])
    verify_hash(dataset, report["dataset_sha256"])
    verify_hash(
        required_file(root, "examples/models/human-baseline.joblib"),
        report["artifact_sha256"],
    )
    verify_hash(
        required_file(root, "examples/models/human-baseline.predictions.csv"),
        report["predictions_sha256"],
    )
    study = required_file(
        root, "examples/experiments/research-revision/checksums.json"
    ).parent
    # Require these files even if their manifest entries are removed.
    required_file(study, "dataset.csv")
    required_file(study, "final/model.joblib")
    verify_manifest(study, "checksums.json")
    for run in ("primary", "study-groups", "note-free"):
        directory = required_file(study, f"{run}/complete.json").parent
        complete = required_file(directory, "complete.json")
        sidecar = required_file(directory, "complete.sha256").read_text().strip()
        verify_hash(complete, sidecar)
        verify_manifest(directory, "complete.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    verify_assets(args.root.resolve())
    print("Required assets and scientific result checksums verified.")


if __name__ == "__main__":
    main()
