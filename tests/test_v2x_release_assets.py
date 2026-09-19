"""Offline integrity and reproducibility of the explicitly partial v2.x bundle."""

import hashlib
from pathlib import Path

import json

from mapexploc import load_report
from mapexploc.study import audit_manifests, eligible_features

ROOT = Path(__file__).resolve().parents[1] / "examples/validation/v2x"


def test_v2x_frozen_cohorts_and_feature_evidence():
    root = ROOT / "cohorts"
    frozen = json.loads((root / "freeze.json").read_text())
    for name, expected in frozen["checksums"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    manifests = [
        json.loads((root / f"v2x-{split}.json").read_text())
        for split in ("development", "reference_pool", "evaluation")
    ]
    groups = json.loads((root / "group-membership.json").read_text())
    historical = load_report(ROOT.parent / "v2/report.json")
    audit_manifests(
        manifests,
        {
            r.protein.protein_id: groups[r.protein.protein_id]
            for r in historical.results
        },
    )
    records = {
        r["primaryAccession"]: r
        for r in json.loads((root / "eligible-records.json").read_text())
    }
    for manifest in manifests:
        for member in manifest["members"]:
            assert (
                eligible_features(records[member["protein_id"]]) == member["families"]
            )
            assert (
                hashlib.sha256(member["sequence"].encode()).hexdigest()
                == member["sequence_sha256"]
            )


def test_partial_native_evidence_never_claims_accurate_passed():
    for name, expected in json.loads((ROOT / "checksums.json").read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    assert not json.loads((ROOT / "qualification.json").read_text())[
        "release_qualified"
    ]
    parity = json.loads((ROOT / "fast-parity.json").read_text())
    assert parity["status"] == "passed" and parity["native_decisions_agree"]
    assert parity["errors"]["full_precision"] <= 1e-5
    assert parity["errors"]["native_csv"] <= 5.1e-5
    failure = json.loads((ROOT / "accurate-qualification-failure.json").read_text())
    assert failure["status"] == "failed" and "MemoryError" in failure["error"]
    native = load_report(ROOT / "fast-native-run/report.json")
    cached = load_report(ROOT / "fast-report/report.json")
    assert native.results == cached.results
    assert native.global_statistics == cached.global_statistics
    assert cached.measurements.native_calls == 0
