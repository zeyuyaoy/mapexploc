"""Tests for the command line interface."""

from pathlib import Path
from typing import Any

from mapexploc.cli import app
from typer.testing import CliRunner


def test_cli_train_predict(tmp_path: Path, monkeypatch: Any) -> None:
    runner = CliRunner()
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config" / "default.yml"

    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        result = runner.invoke(app, ["train", "--config", str(cfg_path)])
        assert result.exit_code == 0
        result = runner.invoke(
            app, ["predict", "MKTIIALSYIFCLVFADYKDDDDK", "--model-path", "model.pkl"]
        )
        assert result.exit_code == 0
        result = runner.invoke(
            app, ["explain", "MKTIIALSYIFCLVFADYKDDDDK", "--model-path", "model.pkl"]
        )
        assert result.exit_code == 0
        assert (tmp_path / "results" / "shap" / "explanation.json").is_file()
