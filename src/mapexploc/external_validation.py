"""Locked external validation. No estimator fitting or label repair is allowed here.

Human attestations and provenance audits are prerequisites, not facts this software
can establish. The local one-use receipt is an audit control, not tamper-proof escrow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .evaluation import classification_metrics
from .features import build_feature_matrix, normalize_protein_sequence
from .research import predict_research

CLASSES = ["Cytoplasm", "Membrane", "Mitochondrion", "Nucleus", "Secreted"]
AUDITS = ("sequence", "remote_domain", "gene", "publication", "assay")
DEPENDENCIES = ("gene_ids", "study_ids", "assay_ids", "batch_ids", "family_ids")
STUDY_STATE = Path("examples/validation/external-v1")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sequence_input_digest(records: list[dict[str, Any]]) -> str:
    payload = sorted(
        [{"case_id": r["case_id"], "sequence": r["sequence"]} for r in records],
        key=lambda r: r["case_id"],
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write_new(path: Path, payload: Any) -> None:
    """Exclusive creation prevents accidental replacement of frozen evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        handle.write("\n")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Timestamps require a timezone")
    return parsed


def verify_freeze(root: Path, freeze_path: Path) -> dict[str, Any]:
    freeze: dict[str, Any] = read(freeze_path)
    for path, expected in freeze["files"].items():
        if digest(root / path) != expected:
            raise ValueError(f"Frozen file changed: {path}")
    actual = {package: version(package) for package in freeze["dependencies"]}
    if actual != freeze["dependencies"]:
        raise ValueError("Frozen inference dependency versions changed")
    if platform.python_version() != freeze["python"]:
        raise ValueError("Frozen Python version changed")
    return freeze


def label_set(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) != len(set(value)):
        raise ValueError("Labels must be a unique list; preserve multilocalization")
    if not set(value) <= set(CLASSES):
        raise ValueError("Unknown mapped label; put other locations in other_locations")
    return sorted(value)


def adjudicated(record: dict[str, Any]) -> tuple[list[str], list[str], bool]:
    """Validate completed independent submissions; never invent a consensus."""
    reviews = record["reviews"]
    if len(reviews) != 2 or len({r["reviewer_id"] for r in reviews}) != 2:
        raise ValueError("Two distinct independent reviewers are required")
    fields = ("labels", "other_locations", "status", "evidence_complete")
    for review in reviews:
        label_set(review["labels"])
        timestamp(review["submitted_at"])
        if (
            not str(review["reviewer_id"]).strip()
            or review["model_blind"] is not True
            or review["independent_submission"] is not True
            or review["notes_and_ambiguity_reviewed"] is not True
            or not review["evidence_rationale"].strip()
        ):
            raise ValueError("Missing blind independent annotation attestation")
    agree = all(
        (
            (sorted(reviews[0][k]) == sorted(reviews[1][k]))
            if k in {"labels", "other_locations"}
            else reviews[0][k] == reviews[1][k]
        )
        for k in fields
    )
    final = reviews[0] if agree else record.get("adjudicator")
    if final is None:
        raise ValueError("Reviewer disagreement requires a third adjudicator")
    if not agree and (
        not str(final["reviewer_id"]).strip()
        or final["reviewer_id"] in {r["reviewer_id"] for r in reviews}
        or final["model_blind"] is not True
        or not final["evidence_rationale"].strip()
    ):
        raise ValueError("Third adjudicator must be distinct and model-blind")
    if not agree and timestamp(final["submitted_at"]) < max(
        timestamp(r["submitted_at"]) for r in reviews
    ):
        raise ValueError("Adjudication must follow both independent submissions")
    labels = label_set(final["labels"])
    other = final["other_locations"]
    if not isinstance(other, list) or any(not str(x).strip() for x in other):
        raise ValueError("Other locations must be an explicit list")
    if final["status"] not in {"resolved", "unresolved"}:
        raise ValueError("Unknown adjudication status")
    resolved = final["status"] == "resolved" and final["evidence_complete"] is True
    return labels, sorted(set(other)), resolved


def dependence_groups(records: list[dict[str, Any]]) -> list[str]:
    """Union all declared dependencies, including across different evidence sources."""
    parent = list(range(len(records)))

    def find(i: int) -> int:
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners: dict[tuple[str, str], int] = {}
    for i, record in enumerate(records):
        for field in DEPENDENCIES:
            values = record[field]
            if not isinstance(values, list):
                raise ValueError(f"Dependence metadata must be a list: {field}")
            for value in values:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Blank dependence identifier: {field}")
                key = field, value
                if key in owners:
                    parent[find(i)] = find(owners[key])
                owners[key] = i
    return [records[find(i)]["case_id"] for i in range(len(records))]


def audit_cohort(
    cohort: dict[str, Any], protocol: dict[str, Any], history: dict[str, Any]
) -> dict[str, Any]:
    records = cohort["records"]
    if not records:
        raise ValueError("An empty cohort cannot be evaluated")
    if not cohort["ascertainment_description"].strip():
        raise ValueError("Declare the source frame and complete ascertainment")
    if cohort["sampling"] != "consecutive_complete_frame":
        raise ValueError("Outcome-dependent class balancing is outside this protocol")
    ids = [r["case_id"] for r in records]
    if len(ids) != len(set(ids)) or any(not str(i).strip() for i in ids):
        raise ValueError("Case identifiers must be nonempty and unique")
    groups = dependence_groups(records)
    accessions = set(history["accessions"])
    sequences = set(history["sequence_sha256"])
    publications = set(history["study_ids"])
    seen_sequences: set[str] = set()
    audited = []
    for record, group in zip(records, groups):
        reasons = []
        if any(not record[field] for field in DEPENDENCIES):
            reasons.append("incomplete_dependence_metadata")
        labels, other, resolved = adjudicated(record)
        try:
            sequence = normalize_protein_sequence(record["sequence"])
        except (ValueError, TypeError):
            sequence = ""
            reasons.append("unsupported_sequence")
        seq_hash = hashlib.sha256(
            (sequence if sequence else str(record["sequence"])).encode()
        ).hexdigest()
        if sequence and seq_hash in seen_sequences:
            raise ValueError(
                "Duplicate sequences: resolve the case unit before sealing"
            )
        if sequence:
            seen_sequences.add(seq_hash)
        if record["accession"] in accessions or seq_hash in sequences:
            reasons.append("development_or_historical_overlap")
        if publications.intersection(record["study_ids"]):
            reasons.append("known_publication_overlap")
        if record["taxon_id"] != 9606:
            reasons.append("nonhuman")
        if (
            record["canonical"] is not True
            or record["full_native_sequence"] is not True
        ):
            reasons.append("isoform_fragment_or_modified_sequence")
        if record["assay_sequence_verified"] is not True:
            reasons.append("assay_isoform_attribution_uncertain")
        if record["detection_breadth_adequate"] is not True:
            reasons.append("partially_observed_localization")
        lower, upper = protocol["support"]["length_range_inclusive"]
        if not lower <= len(sequence) <= upper:
            reasons.append("outside_length_support")
        if not record["context"].strip() or not record["assay_evidence_refs"]:
            reasons.append("missing_context_or_primary_assay_evidence")
        if record["independence_route"] != "prospective_temporal":
            reasons.append("not_primary_temporal_route")
        try:
            assay_time = timestamp(record["first_assay_at"])
            cutoff = timestamp(protocol["prospective_cutoff"])
            if assay_time <= cutoff:
                reasons.append("assay_not_after_freeze")
        except ValueError:
            reasons.append("missing_or_invalid_assay_acquisition_time")
        for name in AUDITS:
            status = record["independence"][name]
            if status not in {"clear", "overlap", "unresolved"}:
                raise ValueError(f"Unknown independence status: {name}")
            if status != "clear":
                reasons.append(f"{name}_{status}")
        if not resolved:
            reasons.append("annotation_unresolved_or_incomplete")
        if not labels:
            reasons.append("no_supported_target_label")
        stratum = (
            "excluded"
            if reasons
            else "primary" if len(labels) == 1 and not other else "multilocalized"
        )
        audited.append(
            {
                "case_id": record["case_id"],
                "labels": labels,
                "other_locations": other,
                "group": group,
                "stratum": stratum,
                "exclusions": reasons,
                "sequence_sha256": seq_hash,
            }
        )
    primary = [r for r in audited if r["stratum"] == "primary"]
    counts = Counter(r["labels"][0] for r in primary)
    failures = []
    gates = protocol["cohort_gates"]
    if len({r["group"] for r in primary}) < gates["min_joint_groups"]:
        failures.append("too_few_joint_groups")
    for label in CLASSES:
        by_group = Counter(r["group"] for r in primary if r["labels"] == [label])
        if counts[label] < gates["min_proteins_per_class"]:
            failures.append(f"too_few_proteins:{label}")
        if len(by_group) < gates["min_joint_groups_per_class"]:
            failures.append(f"too_few_groups:{label}")
        if (
            by_group
            and max(by_group.values()) / counts[label]
            > gates["max_group_fraction_per_class"]
        ):
            failures.append(f"dominant_group:{label}")
    return {
        "records": audited,
        "primary_counts": dict(counts),
        "strata": dict(Counter(r["stratum"] for r in audited)),
        "gate_failures": failures,
        "prediction_accessed": False,
    }


def verify_bundle(bundle: Path) -> dict[str, Any]:
    seal: dict[str, Any] = read(bundle / "seal.json")
    for name, expected in seal["files"].items():
        if digest(bundle / name) != expected:
            raise ValueError(f"Sealed cohort file changed: {name}")
    return seal


def seal_cohort(root: Path, freeze_path: Path, bundle: Path) -> dict[str, Any]:
    freeze = verify_freeze(root, freeze_path)
    protocol = read(root / freeze["protocol"])
    cohort = read(bundle / "cohort.json")
    history = read(root / freeze["history"])
    provenance = read(bundle / "provenance.json")
    if provenance["model_predictions_accessed"] is not False:
        raise ValueError("Cohort construction must remain prediction-blind")
    if not provenance["independent_custodian"].strip():
        raise ValueError("Independent custodian attestation missing")
    if provenance["cohort_sha256"] != digest(bundle / "cohort.json"):
        raise ValueError("Provenance refers to a different cohort")
    if provenance["history_sha256"] != digest(root / freeze["history"]):
        raise ValueError("Wrong development/history exclusion universe")
    if provenance["status"] != "complete":
        raise ValueError("Dependence audit is incomplete")
    recruitment = read(bundle / "recruitment-plan.json")
    if any(item["study_id"] != protocol["study_id"] for item in (cohort, recruitment)):
        raise ValueError("Mismatched study identifiers")
    for name in (
        "precision_design_justification",
        "laboratories_and_contexts",
        "fixed_acquisition_schedule",
        "profile_database_version_and_sha256",
        "sequence_and_profile_tool_versions",
    ):
        if not recruitment.get(name):
            raise ValueError(f"Missing recruitment commitment: {name}")
    if (
        recruitment["predictions_accessed"] is not False
        or not recruitment["source_frames"]
    ):
        raise ValueError("Recruitment needs a prediction-blind source-frame commitment")
    for key in ("committed_at", "assay_acquisition_end", "annotation_started_at"):
        timestamp(recruitment[key])
    if datetime.fromisoformat(recruitment["committed_at"]) >= datetime.fromisoformat(
        recruitment["annotation_started_at"]
    ):
        raise ValueError("Source frame must be committed before annotation starts")
    if not recruitment["independent_registration_receipt"].strip():
        raise ValueError("Recruitment commitment needs an independent dated receipt")
    end = datetime.fromisoformat(recruitment["assay_acquisition_end"])
    observed_dates = []
    for record in cohort["records"]:
        try:
            observed_dates.append(timestamp(record["first_assay_at"]))
        except ValueError:
            pass  # Retain as excluded; do not invent a date for a failed assay.
    if any(t > end for t in observed_dates):
        raise ValueError(
            "Cohort contains assays after the committed acquisition cutoff"
        )
    expected = set(AUDITS) | {
        "within_cohort",
        "bridge_search",
        "source_frame",
        "precision_design",
    }
    if any(t <= timestamp(recruitment["committed_at"]) for t in observed_dates):
        raise ValueError("Prospective source frame must precede assay acquisition")
    reports = provenance["reports"]
    if not expected <= set(reports):
        raise ValueError("Missing sequence, domain, provenance or ascertainment report")
    files: dict[str, str] = {
        name: digest(bundle / name)
        for name in ("cohort.json", "provenance.json", "recruitment-plan.json")
    }
    for name, report in reports.items():
        path = (bundle / report["path"]).resolve()
        if not path.is_relative_to(bundle.resolve()) or not path.is_file():
            raise ValueError("Audit report must be a real file inside the bundle")
        if digest(path) != report["sha256"]:
            raise ValueError(f"Audit report checksum mismatch: {name}")
        if not report["reviewer"].strip() or not report["methods"].strip():
            raise ValueError("Audit report needs accountable reviewer and methods")
        files[str(path.relative_to(bundle.resolve()))] = report["sha256"]
    screen = read(bundle / reports["sequence"]["path"])
    reference_directory = (root / freeze["history"]).parent
    if (
        screen["sequence_input_sha256"] != sequence_input_digest(cohort["records"])
        or screen["reference_sha256"]
        != digest(reference_directory / "history-reference.fasta")
        or screen["bridge_sha256"]
        != digest(reference_directory / "bridge-reference.fasta.gz")
    ):
        raise ValueError("Sequence screen uses stale cohort or reference inputs")
    screened = {r["case_id"]: r for r in screen["records"]}
    if len(screened) != len(screen["records"]) or set(screened) != {
        r["case_id"] for r in cohort["records"]
    }:
        raise ValueError("Sequence screen does not cover every candidate exactly once")
    for record in cohort["records"]:
        result = screened[record["case_id"]]
        if result["sequence_status"] == "unsupported":
            if record["independence"]["sequence"] == "clear":
                raise ValueError("Unscreenable sequence cannot be declared clear")
        else:
            expected_status = "overlap" if result["connected_to_history"] else "clear"
            if record["independence"]["sequence"] != expected_status:
                raise ValueError("Sequence screening contradicts independence claim")
            if "sequence:" + result["sequence_component"] not in record["family_ids"]:
                raise ValueError("Screened sequence components missing from grouping")
    # Initial reviews are separate immutable records; comparing their contents here
    # binds the merged adjudication table to those original submissions.
    submissions = provenance["review_submissions"]
    for record in cohort["records"]:
        reviews = record["reviews"] + (
            [record["adjudicator"]] if record.get("adjudicator") else []
        )
        for review in reviews:
            item = submissions[record["case_id"]][review["reviewer_id"]]
            path = (bundle / item["path"]).resolve()
            if (
                not path.is_relative_to(bundle.resolve())
                or digest(path) != item["sha256"]
            ):
                raise ValueError(
                    "Original review submission changed or is outside bundle"
                )
            if read(path) != {"case_id": record["case_id"], **review}:
                raise ValueError(
                    "Merged review does not match the independent submission"
                )
            if timestamp(review["submitted_at"]) < timestamp(
                recruitment["annotation_started_at"]
            ):
                raise ValueError("Review predates committed annotation start")
            files[str(path.relative_to(bundle.resolve()))] = item["sha256"]
    audit = audit_cohort(cohort, protocol, history)
    if audit["gate_failures"]:
        raise ValueError("Cohort not ready: " + ", ".join(audit["gate_failures"]))
    write_new(bundle / "cohort-audit.json", audit)
    files["cohort-audit.json"] = digest(bundle / "cohort-audit.json")
    seal = {
        "created_at": utc_now(),
        "freeze_sha256": digest(freeze_path),
        "files": files,
    }
    write_new(bundle / "seal.json", seal)
    return seal


def macro_f1(matrix: np.ndarray) -> float:
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    values = np.divide(
        2 * matrix.diagonal(), denominator, out=np.zeros(5), where=denominator != 0
    )
    return float(values.mean())


def paired_bootstrap(
    truth: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    groups: np.ndarray,
    repetitions: int = 10000,
    seed: int = 20260919,
) -> dict[str, Any]:
    """Resample the same joint clusters for both fixed models, without refitting."""
    y = np.array([CLASSES.index(str(label)) for label in truth])
    a, b = first.argmax(axis=1), second.argmax(axis=1)
    unique = np.unique(groups)
    matrices = np.zeros((len(unique), 2, 5, 5), dtype=int)
    for i, group in enumerate(unique):
        members = groups == group
        for j, predictions in enumerate((a, b)):
            np.add.at(matrices[i, j], (y[members], predictions[members]), 1)
    rng = np.random.default_rng(seed)
    values = []
    missing = 0
    for _ in range(repetitions):
        summed = matrices[rng.integers(len(unique), size=len(unique))].sum(axis=0)
        missing += int((summed[0].sum(axis=1) == 0).any())
        # Never redraw a replicate until it looks acceptable.
        x, z = macro_f1(summed[0]), macro_f1(summed[1])
        values.append([x, z, x - z])
    interval = np.quantile(values, [0.025, 0.975], axis=0)
    return {
        "replications": repetitions,
        "seed": seed,
        "joint_groups": len(unique),
        "missing_class_replicate_fraction": missing / repetitions,
        "lr_macro_f1_interval": interval[:, 0].tolist(),
        "rf_macro_f1_interval": interval[:, 1].tolist(),
        "difference_interval": interval[:, 2].tolist(),
    }


def interpret(interval: list[float], margin: float, uncertainty_valid: bool) -> str:
    low, high = interval
    if not uncertainty_valid:
        return "uncertainty_inadequate_no_confirmatory_conclusion"
    if high <= 0:
        return "direction_of_gain_not_supported_external_result_overturns_gain"
    if high < margin:
        return "prespecified_material_gain_ruled_out"
    if low > margin:
        return "material_gain_supported_in_primary_target_population"
    if low > 0:
        return "positive_gain_supported_material_gain_uncertain"
    return "inconclusive_gain_not_independently_confirmed"


def evaluate_once(root: Path, freeze_path: Path, bundle: Path) -> dict[str, Any]:
    freeze = verify_freeze(root, freeze_path)
    protocol = read(root / freeze["protocol"])
    seal = verify_bundle(bundle)
    if seal["freeze_sha256"] != digest(freeze_path):
        raise ValueError("Cohort was sealed under a different model/protocol freeze")
    cohort = read(bundle / "cohort.json")
    audit = audit_cohort(cohort, protocol, read(root / freeze["history"]))
    if audit["gate_failures"] or audit != read(bundle / "cohort-audit.json"):
        raise ValueError("Cohort audit does not reproduce")
    # Create before deserialization or prediction. Failure consumes the attempt;
    # a documented independent technical recovery is required, never delete it.
    receipt = {
        "started_at": utc_now(),
        "seal_sha256": digest(bundle / "seal.json"),
        "freeze_sha256": digest(freeze_path),
        "purpose": "single_primary_evaluation",
    }
    write_new(root / STUDY_STATE / "primary-evaluation-started.json", receipt)
    write_new(bundle / "evaluation-started.json", receipt)
    indices = [i for i, r in enumerate(audit["records"]) if r["stratum"] != "excluded"]
    sequences = [cohort["records"][i]["sequence"] for i in indices]
    with threadpool_limits(limits=1):
        p = predict_research(root / freeze["logistic_model"], sequences)
        reference = joblib.load(root / freeze["reference_model"])
        if list(reference["model"].classes_) != CLASSES:
            raise ValueError("Reference probability class order changed")
        q = reference["model"].predict_proba(build_feature_matrix(sequences))
    rows = [audit["records"][i] for i in indices]
    primary = np.array([r["stratum"] == "primary" for r in rows])
    truth = np.array([r["labels"][0] for r in rows])[primary]
    groups = np.array([r["group"] for r in rows])[primary]
    metrics = []
    for probabilities in (p[primary], q[primary]):
        metrics.append(
            classification_metrics(
                truth,
                np.asarray(CLASSES)[probabilities.argmax(axis=1)],
                probabilities,
                CLASSES,
            )
        )
    uncertainty = paired_bootstrap(
        truth,
        p[primary],
        q[primary],
        groups,
        protocol["bootstrap"]["replications"],
        protocol["bootstrap"]["seed"],
    )
    influence: list[dict[str, Any]] = []
    encoded = np.array([CLASSES.index(str(label)) for label in truth])
    primary_predictions = [prob[primary].argmax(axis=1) for prob in (p, q)]
    for group in np.unique(groups):
        keep = groups != group
        if set(truth[keep]) != set(CLASSES):
            influence.append(
                {"omitted_group": str(group), "difference": None, "missing_class": True}
            )
            continue
        values = []
        for predicted in primary_predictions:
            matrix = np.zeros((5, 5), dtype=int)
            np.add.at(matrix, (encoded[keep], predicted[keep]), 1)
            values.append(macro_f1(matrix))
        influence.append(
            {
                "omitted_group": str(group),
                "difference": values[0] - values[1],
                "missing_class": False,
            }
        )
    predictions = [
        {
            **row,
            "logistic_probabilities": a.tolist(),
            "reference_probabilities": b.tolist(),
        }
        for row, a, b in zip(rows, p, q)
    ]
    multi = [i for i, r in enumerate(rows) if r["stratum"] == "multilocalized"]
    set_hits = {
        name: (
            float(
                np.mean([CLASSES[prob[i].argmax()] in rows[i]["labels"] for i in multi])
            )
            if multi
            else None
        )
        for name, prob in (("logistic", p), ("reference", q))
    }
    result = {
        "completed_at": utc_now(),
        "external_test": True,
        "primary_estimand": protocol["primary_estimand"],
        "logistic": metrics[0],
        "reference": metrics[1],
        "difference": metrics[0]["macro_f1"] - metrics[1]["macro_f1"],
        "uncertainty": uncertainty,
        "leave_one_group_out_secondary": influence,
        "decision": interpret(
            uncertainty["difference_interval"],
            protocol["practical_margin"],
            uncertainty["missing_class_replicate_fraction"]
            <= protocol["bootstrap"]["max_missing_class_fraction"],
        ),
        "multilocalized_secondary": {
            "n": len(multi),
            "top1_in_supported_set": set_hits,
        },
        "coverage": audit["strata"],
        "wet_lab_confirmation": False,
        "limitations": [
            "Conditional on fixed models, this cohort frame and adjudication",
            "No architecture-causal or state-of-the-art claim",
            "Multilabel set hits do not estimate complete localization accuracy",
        ],
    }
    write_new(bundle / "predictions.json", predictions)
    write_new(bundle / "results.json", result)
    write_new(
        bundle / "evaluation-complete.json",
        {
            p.name: digest(p)
            for p in (
                bundle / "results.json",
                bundle / "predictions.json",
                bundle / "evaluation-started.json",
            )
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "seal", "evaluate"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--freeze",
        type=Path,
        default=Path("examples/validation/external-v1/freeze-v1.1.json"),
    )
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    freeze_path = args.root / args.freeze
    if args.action == "verify":
        verify_freeze(args.root, freeze_path)
        print("Frozen files and inference environment verified; no predictions made.")
    else:
        if args.bundle is None:
            parser.error("--bundle is required for seal/evaluate")
        result = (seal_cohort if args.action == "seal" else evaluate_once)(
            args.root, freeze_path, args.bundle
        )
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
