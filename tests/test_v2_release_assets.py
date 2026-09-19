"""Offline integrity of real-model evidence; no native checkpoint is needed."""

import hashlib
import json
from pathlib import Path

from mapexploc import AnalysisReport
from mapexploc.biological_validation import validate_signal_concordance


def test_frozen_v2_release_reproduces_its_scientific_result():
    root = Path(__file__).resolve().parents[1] / "examples/validation/v2"
    checksums = json.loads((root / "checksums.json").read_text())
    for name, checksum in checksums.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert actual == checksum, (
            f"Frozen evidence changed: {root / name}; "
            f"expected {checksum}, got {actual}"
        )
    report = AnalysisReport.model_validate_json((root / "report.json").read_text())
    manifest = json.loads((root / "cohort-manifest.json").read_text())
    assert [r.protein.protein_id for r in report.results] == [
        r["protein_id"] for r in manifest["members"]
    ]
    assert len(report.results) == len({r.protein.group for r in report.results}) == 30
    expected = json.loads((root / "biological-validation.json").read_text())
    assert validate_signal_concordance(report) == expected
    assert (
        expected["conclusion"] == "No support for concordance beyond the matched null"
    )
    parity = json.loads((root / "native-parity.json").read_text())
    assert parity["native_decisions_agree"]
    assert parity["native_cli_max_absolute_probability_error"] < parity["tolerance"]
    clean = json.loads((root / "clean-qualification.json").read_text())
    assert clean["status"] == "passed"
    assert clean["checkpoint_sha256"] == report.model.checkpoint_sha256
    assert clean["probability_max_absolute_delta"] <= clean["tolerance"]
    assert clean["attribution_max_absolute_delta"] <= clean["tolerance"]
