"""Synthetic safeguards only; none of these tests estimate biological performance."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import mapexploc.external_validation as external


def protocol() -> dict:
    return {
        "support": {"length_range_inclusive": [51, 5654]},
        "study_id": "mapexploc-external-v1",
        "prospective_cutoff": "2026-09-18T09:27:29+00:00",
        "cohort_gates": {
            "min_joint_groups": 5,
            "min_proteins_per_class": 1,
            "min_joint_groups_per_class": 1,
            "max_group_fraction_per_class": 1,
        },
        "primary_estimand": "synthetic fixture",
        "practical_margin": 0.03,
        "bootstrap": {
            "replications": 20,
            "seed": 42,
            "max_missing_class_fraction": 0.01,
        },
    }


def review(label: str, identity: str) -> dict:
    return {
        "reviewer_id": identity,
        "labels": [label],
        "other_locations": [],
        "status": "resolved",
        "evidence_complete": True,
        "model_blind": True,
        "independent_submission": True,
        "notes_and_ambiguity_reviewed": True,
        "evidence_rationale": "synthetic test record",
        "submitted_at": "2027-01-02T00:00:00+00:00",
    }


def cohort() -> dict:
    rows = []
    for i, label in enumerate(external.CLASSES):
        rows.append(
            {
                "case_id": f"case{i}",
                "accession": f"new{i}",
                "sequence": "A" * 51 + "K" * i,
                "taxon_id": 9606,
                "canonical": True,
                "full_native_sequence": True,
                "assay_sequence_verified": True,
                "detection_breadth_adequate": True,
                "context": "synthetic",
                "assay_evidence_refs": [f"test:{i}"],
                "independence_route": "prospective_temporal",
                "first_assay_at": "2027-01-01T00:00:00+00:00",
                "independence": dict.fromkeys(external.AUDITS, "clear"),
                "reviews": [review(label, "A"), review(label, "B")],
                **{k: [f"{k}:{i}"] for k in external.DEPENDENCIES},
            }
        )
    return {
        "study_id": "mapexploc-external-v1",
        "records": rows,
        "sampling": "consecutive_complete_frame",
        "ascertainment_description": "synthetic fixture only",
    }


def history() -> dict:
    return {"accessions": [], "sequence_sha256": [], "study_ids": []}


def test_multilocalization_and_partial_observation_are_not_single_labels() -> None:
    data = cohort()
    for r in data["records"][0]["reviews"]:
        r["labels"] = ["Cytoplasm", "Nucleus"]
    data["records"][1]["detection_breadth_adequate"] = False
    data["records"][2]["assay_sequence_verified"] = False
    result = external.audit_cohort(data, protocol(), history())
    assert result["records"][0]["stratum"] == "multilocalized"
    assert result["records"][0]["labels"] == ["Cytoplasm", "Nucleus"]
    assert "partially_observed_localization" in result["records"][1]["exclusions"]
    assert "assay_isoform_attribution_uncertain" in result["records"][2]["exclusions"]
    assert result["gate_failures"]


def test_disagreement_requires_independent_blind_third_reviewer() -> None:
    row = cohort()["records"][0]
    row["reviews"][1]["labels"] = ["Nucleus"]
    with pytest.raises(ValueError, match="third adjudicator"):
        external.adjudicated(row)
    row["adjudicator"] = review("Cytoplasm", "A")
    with pytest.raises(ValueError, match="distinct"):
        external.adjudicated(row)
    row["adjudicator"] = review("Cytoplasm", "C")
    row["adjudicator"]["labels"] = ["Cytoplasm", "Nucleus"]
    assert external.adjudicated(row)[0] == ["Cytoplasm", "Nucleus"]
    row["reviews"][0]["model_blind"] = False
    with pytest.raises(ValueError, match="attestation"):
        external.adjudicated(row)


def test_history_provenance_and_temporal_exclusions_cannot_be_overridden() -> None:
    data = cohort()
    old = history()
    old["accessions"] = ["new0"]
    old["study_ids"] = ["study_ids:1"]
    data["records"][2]["independence"]["remote_domain"] = "unresolved"
    data["records"][3]["first_assay_at"] = "2020-01-01T00:00:00+00:00"
    result = external.audit_cohort(data, protocol(), old)
    assert all(r["stratum"] == "excluded" for r in result["records"][:4])
    data["records"][4]["sequence"] = data["records"][0]["sequence"]
    with pytest.raises(ValueError, match="Duplicate sequences"):
        external.audit_cohort(data, protocol(), old)


def test_dependence_components_are_transitive_across_metadata_types() -> None:
    rows = cohort()["records"]
    rows[0]["study_ids"] = rows[1]["study_ids"]
    rows[1]["gene_ids"] = rows[2]["gene_ids"]
    groups = external.dependence_groups(rows)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]
    rows[3]["assay_ids"] = []
    assert external.dependence_groups(rows)[0] == groups[0]
    assert (
        "incomplete_dependence_metadata"
        in external.audit_cohort({**cohort(), "records": rows}, protocol(), history())[
            "records"
        ][3]["exclusions"]
    )


def test_bootstrap_is_paired_and_missing_classes_are_counted_not_redrawn() -> None:
    y = np.asarray(external.CLASSES * 20)
    p = np.tile(np.eye(5), (20, 1))
    q = np.roll(p, 1, axis=1)
    a = external.paired_bootstrap(y, p, q, np.arange(100), 100, 42)
    b = external.paired_bootstrap(y, q, p, np.arange(100), 100, 42)
    assert a["difference_interval"] == [1, 1]
    assert b["difference_interval"] == [-1, -1]
    sparse = external.paired_bootstrap(
        np.asarray(external.CLASSES), np.eye(5), np.eye(5), np.arange(5), 100, 42
    )
    assert sparse["missing_class_replicate_fraction"] > 0.5
    assert sparse["replications"] == 100
    assert sparse["difference_interval"] == [0, 0]


def test_decision_gates_do_not_conflate_inconclusive_with_equivalent() -> None:
    assert "overturns" in external.interpret([-0.1, -0.01], 0.03, True)
    assert "ruled_out" in external.interpret([-0.01, 0.02], 0.03, True)
    assert "material_gain_supported" in external.interpret([0.04, 0.08], 0.03, True)
    assert "inconclusive" in external.interpret([-0.01, 0.08], 0.03, True)
    assert "uncertainty_inadequate" in external.interpret([0.04, 0.08], 0.03, False)


def test_one_use_receipt_and_tamper_guards_before_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    bundle = root / "bundle"
    bundle.mkdir()
    external.write_new(root / "protocol.json", protocol())
    external.write_new(root / "history.json", history())
    freeze = {
        "protocol": "protocol.json",
        "history": "history.json",
        "logistic_model": "fake-lr",
        "reference_model": "fake-rf",
    }
    freeze_path = root / "freeze.json"
    external.write_new(freeze_path, freeze)
    monkeypatch.setattr(external, "verify_freeze", lambda *args: freeze)
    external.write_new(bundle / "cohort.json", cohort())
    external.write_new(
        bundle / "cohort-audit.json",
        external.audit_cohort(cohort(), protocol(), history()),
    )
    seal = {
        "freeze_sha256": external.digest(freeze_path),
        "files": {
            name: external.digest(bundle / name)
            for name in ("cohort.json", "cohort-audit.json")
        },
    }
    external.write_new(bundle / "seal.json", seal)
    calls = []
    monkeypatch.setattr(
        external, "predict_research", lambda *args: calls.append("predict") or np.eye(5)
    )

    class FixedModel:
        classes_ = np.asarray(external.CLASSES)

        def predict_proba(self, _values: object) -> np.ndarray:
            return np.eye(5)

    monkeypatch.setattr(external.joblib, "load", lambda *args: {"model": FixedModel()})
    external.evaluate_once(root, freeze_path, bundle)
    assert calls == ["predict"]
    with pytest.raises(FileExistsError):
        external.evaluate_once(root, freeze_path, bundle)
    assert calls == ["predict"]
    # A different bundle cannot evade the study-level use receipt.
    second = root / "second"
    import shutil

    shutil.copytree(bundle, second)
    (second / "evaluation-started.json").unlink()
    with pytest.raises(FileExistsError):
        external.evaluate_once(root, freeze_path, second)
    alternative_freeze = root / "elsewhere" / "freeze.json"
    external.write_new(alternative_freeze, freeze)
    with pytest.raises(FileExistsError):
        external.evaluate_once(root, alternative_freeze, second)
    (bundle / "cohort.json").write_text(json.dumps(copy.deepcopy(cohort())) + " ")
    with pytest.raises(ValueError, match="Sealed cohort file changed"):
        external.evaluate_once(root, freeze_path, bundle)


def test_probability_class_universe_is_five_and_immutable() -> None:
    with pytest.raises(ValueError, match="Unknown mapped label"):
        external.label_set(["ER"])
    with pytest.raises(ValueError, match="unique list"):
        external.label_set(["Nucleus", "Nucleus"])


def test_multiple_unsupported_records_remain_in_complete_frame() -> None:
    data = cohort()
    data["records"][0]["sequence"] = "AXXX"
    data["records"][1]["sequence"] = "MMMMU"
    data["records"][0]["first_assay_at"] = ""
    result = external.audit_cohort(data, protocol(), history())
    assert len(result["records"]) == 5
    assert result["records"][0]["stratum"] == "excluded"
    assert result["records"][1]["stratum"] == "excluded"


def test_agreement_compares_localization_sets_not_order() -> None:
    row = cohort()["records"][0]
    row["reviews"][0]["labels"] = ["Cytoplasm", "Nucleus"]
    row["reviews"][1]["labels"] = ["Nucleus", "Cytoplasm"]
    labels, other, resolved = external.adjudicated(row)
    assert labels == ["Cytoplasm", "Nucleus"]
    assert not other
    assert resolved


def make_sealable_fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Invented evidence is used ONLY to test file/logic checks in a temp folder."""
    bundle = root / "bundle"
    bundle.mkdir()
    external.write_new(root / "protocol.json", protocol())
    external.write_new(root / "history.json", history())
    (root / "history-reference.fasta").write_text("synthetic")
    (root / "bridge-reference.fasta.gz").write_bytes(b"synthetic")
    freeze = {"protocol": "protocol.json", "history": "history.json"}
    external.write_new(root / "freeze.json", freeze)
    monkeypatch.setattr(external, "verify_freeze", lambda *args: freeze)
    data = cohort()
    for i, row in enumerate(data["records"]):
        row["family_ids"].append(f"sequence:E_{i}")
    external.write_new(bundle / "cohort.json", data)
    recruitment = {
        "study_id": "mapexploc-external-v1",
        "predictions_accessed": False,
        "source_frames": ["synthetic"],
        "committed_at": "2026-12-31T00:00:00+00:00",
        "annotation_started_at": "2027-01-02T00:00:00+00:00",
        "assay_acquisition_end": "2027-01-01T23:00:00+00:00",
        "independent_registration_receipt": "synthetic",
        **dict.fromkeys(
            (
                "precision_design_justification",
                "laboratories_and_contexts",
                "fixed_acquisition_schedule",
                "profile_database_version_and_sha256",
                "sequence_and_profile_tool_versions",
            ),
            "synthetic",
        ),
    }
    external.write_new(bundle / "recruitment-plan.json", recruitment)
    reports = {}
    for name in (
        *external.AUDITS,
        "within_cohort",
        "bridge_search",
        "source_frame",
        "precision_design",
    ):
        contents = {"synthetic": True}
        if name == "sequence":
            contents = {
                "sequence_input_sha256": external.sequence_input_digest(
                    data["records"]
                ),
                "reference_sha256": external.digest(root / "history-reference.fasta"),
                "bridge_sha256": external.digest(root / "bridge-reference.fasta.gz"),
                "records": [
                    {
                        "case_id": r["case_id"],
                        "sequence_status": "screened",
                        "connected_to_history": False,
                        "sequence_component": f"E_{i}",
                    }
                    for i, r in enumerate(data["records"])
                ],
            }
        path = bundle / "reports" / f"{name}.json"
        external.write_new(path, contents)
        reports[name] = {
            "path": str(path.relative_to(bundle)),
            "sha256": external.digest(path),
            "reviewer": "synthetic",
            "methods": "synthetic",
        }
    submissions = {}
    for row in data["records"]:
        submissions[row["case_id"]] = {}
        for review_row in row["reviews"]:
            path = (
                bundle
                / "reviews"
                / f"{row['case_id']}-{review_row['reviewer_id']}.json"
            )
            external.write_new(path, {"case_id": row["case_id"], **review_row})
            submissions[row["case_id"]][review_row["reviewer_id"]] = {
                "path": str(path.relative_to(bundle)),
                "sha256": external.digest(path),
            }
    provenance = {
        "model_predictions_accessed": False,
        "independent_custodian": "synthetic",
        "cohort_sha256": external.digest(bundle / "cohort.json"),
        "history_sha256": external.digest(root / "history.json"),
        "status": "complete",
        "reports": reports,
        "review_submissions": submissions,
    }
    external.write_new(bundle / "provenance.json", provenance)
    return bundle, provenance


def test_seal_checks_screen_inputs_and_original_submissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, provenance = make_sealable_fixture(tmp_path, monkeypatch)
    screen_path = bundle / provenance["reports"]["sequence"]["path"]
    screen = external.read(screen_path)
    screen["records"][0]["connected_to_history"] = True
    screen_path.write_text(json.dumps(screen))
    provenance["reports"]["sequence"]["sha256"] = external.digest(screen_path)
    (bundle / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="contradicts"):
        external.seal_cohort(tmp_path, tmp_path / "freeze.json", bundle)
    screen["records"][0]["connected_to_history"] = False
    screen_path.write_text(json.dumps(screen))
    provenance["reports"]["sequence"]["sha256"] = external.digest(screen_path)
    (bundle / "provenance.json").write_text(json.dumps(provenance))
    original = bundle / provenance["review_submissions"]["case0"]["A"]["path"]
    original.write_text(original.read_text() + " ")
    with pytest.raises(ValueError, match="Original review submission changed"):
        external.seal_cohort(tmp_path, tmp_path / "freeze.json", bundle)
    original.write_text(original.read_text().rstrip() + "\n")
    external.seal_cohort(tmp_path, tmp_path / "freeze.json", bundle)
    assert external.verify_bundle(bundle)["files"]["cohort.json"] == external.digest(
        bundle / "cohort.json"
    )


def test_stale_sequence_screen_and_blank_commitment_block_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _ = make_sealable_fixture(tmp_path, monkeypatch)
    (tmp_path / "bridge-reference.fasta.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="stale"):
        external.seal_cohort(tmp_path, tmp_path / "freeze.json", bundle)
    (tmp_path / "bridge-reference.fasta.gz").write_bytes(b"synthetic")
    recruitment = external.read(bundle / "recruitment-plan.json")
    recruitment["profile_database_version_and_sha256"] = ""
    (bundle / "recruitment-plan.json").write_text(json.dumps(recruitment))
    with pytest.raises(ValueError, match="Missing recruitment commitment"):
        external.seal_cohort(tmp_path, tmp_path / "freeze.json", bundle)
