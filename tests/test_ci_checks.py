"""CI gates must reject missing, tampered, or escaping release assets."""

import hashlib
import json
import runpy
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
verify_assets = runpy.run_path(str(SCRIPTS / "ci_verify_assets.py"))["verify_assets"]
distributions = runpy.run_path(str(SCRIPTS / "ci_package_smoke.py"))["distributions"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def assets(tmp_path: Path) -> Path:
    for name in (
        "examples/models/human-baseline.joblib",
        "examples/models/human-baseline.predictions.csv",
        "examples/baseline/dataset.csv",
        "examples/experiments/research-revision/dataset.csv",
        "examples/experiments/research-revision/final/model.joblib",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    model = "examples/models/human-baseline.joblib"
    (tmp_path / "config").mkdir()
    (tmp_path / "config/default-model.json").write_text(
        json.dumps({"artifact_path": model, "sha256": digest(tmp_path / model)})
    )
    dataset_hash = digest(tmp_path / "examples/baseline/dataset.csv")
    (tmp_path / "examples/baseline/manifest.json").write_text(
        json.dumps({"dataset_sha256": dataset_hash})
    )
    (tmp_path / "examples/models/human-baseline.report.json").write_text(
        json.dumps(
            {
                "dataset_sha256": dataset_hash,
                "artifact_sha256": digest(tmp_path / model),
                "predictions_sha256": digest(
                    tmp_path / "examples/models/human-baseline.predictions.csv"
                ),
            }
        )
    )
    study = tmp_path / "examples/experiments/research-revision"
    for run in ("primary", "study-groups", "note-free"):
        directory = study / run
        directory.mkdir()
        (directory / "result.json").write_text("{}")
        complete = directory / "complete.json"
        complete.write_text(
            json.dumps({"result.json": digest(directory / "result.json")})
        )
        (directory / "complete.sha256").write_text(digest(complete))
    (study / "checksums.json").write_text(
        json.dumps({"dataset.csv": digest(study / "dataset.csv")})
    )
    return tmp_path


def test_valid_assets(assets: Path) -> None:
    verify_assets(assets)


@pytest.mark.parametrize(
    "name",
    [
        "examples/models/human-baseline.joblib",
        "examples/models/human-baseline.predictions.csv",
        "examples/baseline/dataset.csv",
        "examples/experiments/research-revision/final/model.joblib",
        "examples/experiments/research-revision/primary/complete.json",
        "examples/experiments/research-revision/note-free/complete.sha256",
    ],
)
def test_missing_required_asset_fails(assets: Path, name: str) -> None:
    (assets / name).unlink()
    with pytest.raises(ValueError):
        verify_assets(assets)


@pytest.mark.parametrize(
    "name",
    [
        "examples/models/human-baseline.joblib",
        "examples/baseline/dataset.csv",
        "examples/models/human-baseline.predictions.csv",
        "examples/experiments/research-revision/dataset.csv",
        "examples/experiments/research-revision/primary/result.json",
        "examples/experiments/research-revision/study-groups/complete.json",
    ],
)
def test_tampered_asset_fails(assets: Path, name: str) -> None:
    (assets / name).write_text("tampered")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_assets(assets)


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside"])
def test_escaping_manifest_path_fails(assets: Path, name: str) -> None:
    study = assets / "examples/experiments/research-revision"
    (study / "checksums.json").write_text(json.dumps({name: "0" * 64}))
    with pytest.raises(ValueError, match="Unsafe asset path"):
        verify_assets(assets)


def test_symlink_escape_fails(assets: Path, tmp_path_factory) -> None:
    outside = tmp_path_factory.mktemp("outside") / "model"
    outside.write_text("fixture")
    model = assets / "examples/models/human-baseline.joblib"
    model.unlink()
    model.symlink_to(outside)
    with pytest.raises(ValueError, match="escaping asset"):
        verify_assets(assets)


def test_empty_manifest_fails(assets: Path) -> None:
    (assets / "examples/experiments/research-revision/checksums.json").write_text("{}")
    with pytest.raises(ValueError, match="Empty or invalid manifest"):
        verify_assets(assets)


def test_distribution_selection_rejects_stale_outputs(tmp_path: Path) -> None:
    wheel = tmp_path / "mapexploc-1.0-py3-none-any.whl"
    wheel.touch()
    with pytest.raises(ValueError):
        distributions(tmp_path, "1.0")
    (tmp_path / "mapexploc-1.0.tar.gz").touch()
    assert distributions(tmp_path, "1.0") == wheel
    (tmp_path / "mapexploc-0.9-py3-none-any.whl").touch()
    with pytest.raises(ValueError):
        distributions(tmp_path, "1.0")
