"""Versioned, resumable CPU experiments; scientific runs are always explicit."""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .artifacts import load_model_artifact, save_model_artifact
from .baseline import (
    CLASSES,
    LIMITATIONS,
    SOURCE_URL,
    checksum,
    classification_metrics,
    curate_snapshot,
    search_similar,
    similarity_groups,
    write_fasta,
)
from .default_model import manifest_selection
from .explainers.shap import ShapExplainer
from .features import build_feature_matrix

PROTOCOL = 1
SEEDS = (42, 43, 44)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def software() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        **{
            name: version(name)
            for name in (
                "numpy",
                "pandas",
                "scikit-learn",
                "shap",
                "biopython",
                "joblib",
            )
        },
    }


def implementation_hash() -> str:
    package = Path(__file__).parent
    names = (
        "experiments.py",
        "baseline.py",
        "features.py",
        "artifacts.py",
        "explainers/shap.py",
    )
    return hashlib.sha256(
        "".join(checksum(package / n) for n in names).encode()
    ).hexdigest()


def _commit_stage(directory: Path, stage: str, files: list[str]) -> None:
    atomic_json(
        directory / f"{stage}.complete.json",
        {
            "files": {name: checksum(directory / name) for name in files},
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )


def _completed(directory: Path, stage: str) -> bool:
    marker = directory / f"{stage}.complete.json"
    if not marker.exists():
        return False
    for name, digest in read_json(marker)["files"].items():
        if checksum(directory / name) != digest:
            raise ValueError(f"Completed {stage} output changed: {name}")
    return True


def _validate_run(directory: Path) -> dict[str, Any]:
    config = read_json(directory / "run.json")
    if (
        config["software"] != software()
        or config["implementation_sha256"] != implementation_hash()
    ):
        raise ValueError(
            "Run software or implementation changed; use a new run directory"
        )
    for field, filename in (
        ("source_sha256", "source.json"),
        ("headers_sha256", "source.headers"),
    ):
        if checksum(directory / filename) != config[field]:
            raise ValueError(f"Run input changed: {filename}")
    for name, digest in config["reference_hashes"].items():
        if checksum(Path(config["reference_root"]) / name) != digest:
            raise ValueError(f"Version 1 reference changed: {name}")
    return dict(config)


def sequence_id(accession: str, sequence: str) -> str:
    return f"{accession}__{hashlib.sha256(sequence.encode()).hexdigest()[:16]}"


def assign_partitions(
    frame: pd.DataFrame, original: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Exclude historical groups and reserve new groups before model selection."""
    frame = frame.copy()
    original_groups = set(original["group"])
    historical_groups = set(original.loc[original["split"] == "test", "group"])
    original_ids = set(original["accession"])
    historical_ids = set(original.loc[original["split"] == "test", "accession"])
    historical_groups.update(
        frame.loc[frame["accession"].isin(historical_ids), "group"]
    )
    original_groups.update(frame.loc[frame["accession"].isin(original_ids), "group"])
    frame["split"] = "development"
    frame.loc[frame["group"].isin(historical_groups), "split"] = "excluded_historical"
    eligible = frame.loc[
        ~frame["group"].isin(original_groups) & ~frame["accession"].isin(original_ids)
    ]
    counts = {c: int((eligible["label"] == c).sum()) for c in CLASSES}
    candidates: list[tuple[float, list[int]]] = []
    if all(counts[c] >= 20 for c in CLASSES) and eligible["group"].nunique() >= 5:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for _, test in cv.split(eligible, eligible["label"], eligible["group"]):
            selected = eligible.iloc[test]
            if all((selected["label"] == c).sum() >= 20 for c in CLASSES):
                development = frame.loc[
                    (frame["split"] == "development")
                    & ~frame["group"].isin(selected["group"])
                ]
                if set(development["label"]) == set(CLASSES):
                    candidates.append(
                        (
                            abs(len(selected) / len(eligible) - 0.2),
                            sorted(selected.index),
                        )
                    )
    confirmation: list[int] = min(
        candidates, default=(0, []), key=lambda x: (x[0], x[1])
    )[1]
    frame.loc[confirmation, "split"] = "confirmation"
    status = "available" if confirmation else "unavailable"
    return frame, {
        "status": status,
        "eligible_counts": counts,
        "minimum_count_shortfalls": {c: max(0, 20 - counts[c]) for c in CLASSES},
        "reason": (
            None
            if confirmation
            else (
                "Insufficient independent groups for 20 confirmation proteins "
                "per class in an approximately 20% split"
            )
        ),
        "confirmation_counts": {
            c: int(((frame["split"] == "confirmation") & (frame["label"] == c)).sum())
            for c in CLASSES
        },
    }


def prepare_experiment(
    directory: Path, reference_root: Path, cap: int = 1000, threads: int = 4
) -> dict[str, Any]:
    """Freeze curated records, union similarity groups and evaluation partitions."""
    directory.mkdir(parents=True, exist_ok=True)
    reference_root = reference_root.resolve()
    reference_names = (
        "examples/baseline/dataset.csv",
        "examples/baseline/manifest.json",
        "examples/models/human-baseline.joblib",
    )
    config = {
        "protocol": PROTOCOL,
        "cap": cap,
        "seed": 42,
        "threads": threads,
        "source_sha256": checksum(directory / "source.json"),
        "headers_sha256": checksum(directory / "source.headers"),
        "reference_root": str(reference_root),
        "reference_hashes": {n: checksum(reference_root / n) for n in reference_names},
        "software": software(),
        "implementation_sha256": implementation_hash(),
    }
    run_path = directory / "run.json"
    if run_path.exists() and read_json(run_path) != config:
        raise ValueError("Run inputs/configuration changed; use a new directory")
    atomic_json(run_path, config)
    if _completed(directory, "prepare"):
        return dict(read_json(directory / "preparation.json"))
    original = pd.read_csv(reference_root / reference_names[0])
    original_manifest = read_json(reference_root / reference_names[1])
    if (
        checksum(reference_root / reference_names[0])
        != original_manifest["dataset_sha256"]
    ):
        raise ValueError("Version 1 dataset does not match its provenance")
    fresh, exclusions = curate_snapshot(
        directory / "source.json", cap=1_000_000, seed=42
    )
    if fresh["accession"].duplicated().any():
        raise ValueError("Refreshed snapshot contains duplicate accessions")
    # Preserve original structured location annotations as well as selected evidence.
    raw = read_json(directory / "source.json")["results"]
    annotations = {
        e["primaryAccession"]: [
            c
            for c in e.get("comments", [])
            if c.get("commentType") == "SUBCELLULAR LOCATION"
        ]
        for e in raw
    }
    fresh["localization_annotations"] = fresh["accession"].map(
        lambda a: canonical(annotations[a])
    )
    for table in (original, fresh):
        table["node_id"] = [
            sequence_id(str(a), str(s)) for a, s in zip(table.accession, table.sequence)
        ]
    union = (
        pd.concat([original, fresh], ignore_index=True)
        .drop_duplicates("node_id")
        .sort_values("node_id")
    )
    nodes = union[["node_id", "sequence"]].rename(columns={"node_id": "accession"})
    write_fasta(nodes, directory / "union.fasta")
    pairs = search_similar(
        directory / "union.fasta",
        directory / "union.fasta",
        directory / "all-pairs.tsv",
        threads,
    )
    groups = dict(
        zip(nodes.accession, similarity_groups(nodes.accession.tolist(), pairs))
    )
    original["group"] = original.node_id.map(groups)
    fresh["group"] = fresh.node_id.map(groups)
    fresh = (
        pd.concat(
            [
                g.sample(min(cap, len(g)), random_state=42)
                for _, g in fresh.groupby("label", sort=True)
            ]
        )
        .sort_values("accession")
        .reset_index(drop=True)
    )
    frame, confirmation = assign_partitions(fresh, original)
    development = frame.loc[frame["split"] == "development"]
    if set(development.label) != set(CLASSES):
        raise ValueError("Development partition must contain all five classes")
    historical = original.loc[original["split"] == "test"].copy()
    audit_files = []
    for name, table in (
        ("development", development),
        ("confirmation", frame.loc[frame["split"] == "confirmation"]),
        ("historical", historical),
    ):
        write_fasta(
            table.rename(
                columns={"accession": "original_accession", "node_id": "accession"}
            ),
            directory / f"{name}.fasta",
        )
    for target in ("confirmation", "historical"):
        if (directory / f"{target}.fasta").stat().st_size == 0:
            continue
        for first, second in (("development", target), (target, "development")):
            name = f"audit-{first}-{second}.tsv"
            if search_similar(
                directory / f"{first}.fasta",
                directory / f"{second}.fasta",
                directory / name,
                threads,
            ):
                raise ValueError(
                    "Cross-partition similarity detected; refusing experiment"
                )
            audit_files.append(name)
    frame.to_csv(directory / "dataset.csv", index=False)
    historical.to_csv(directory / "historical.csv", index=False)
    release = next(
        (
            line.split(":", 1)[1].strip()
            for line in (directory / "source.headers").read_text().splitlines()
            if line.lower().startswith("x-uniprot-release:")
        ),
        None,
    )
    if not release:
        raise ValueError("Source release header is missing")
    report = {
        "experiment_id": "human-v2-" + checksum(directory / "run.json")[:12],
        "release": release,
        "source": SOURCE_URL,
        "retrieved_at": datetime.fromtimestamp(
            (directory / "source.json").stat().st_mtime, UTC
        ).isoformat(),
        "source_sha256": config["source_sha256"],
        "dataset_sha256": checksum(directory / "dataset.csv"),
        "class_counts": {str(k): int(v) for k, v in frame.label.value_counts().items()},
        "partition_counts": {
            str(k): int(v) for k, v in frame.split.value_counts().items()
        },
        "confirmation": confirmation,
        "exclusions": exclusions,
        "changed_sequences": int(
            sum(
                a in set(original.accession) and n not in set(original.node_id)
                for a, n in zip(fresh.accession, fresh.node_id)
            )
        ),
        "changed_labels": int(
            sum(
                a in set(original.accession)
                and label != original.set_index("accession").loc[a, "label"]
                for a, label in zip(fresh.accession, fresh.label)
            )
        ),
        "split": {
            "identity": 0.3,
            "bidirectional_coverage": 0.8,
            "cross_partition_hits": 0,
            "historical_test_excluded": True,
            "test_refitted": False,
        },
        "attribution": original_manifest["attribution"],
        "limitations": LIMITATIONS,
    }
    atomic_json(directory / "preparation.json", report)
    _commit_stage(
        directory,
        "prepare",
        [
            "dataset.csv",
            "historical.csv",
            "preparation.json",
            "union.fasta",
            "all-pairs.tsv",
            *audit_files,
        ],
    )
    return report


def candidate_configurations() -> list[dict[str, Any]]:
    grid = {
        "n_estimators": [256, 512],
        "max_depth": [12, 24, None],
        "min_samples_leaf": [1, 3, 5],
        "max_features": ["sqrt", 0.5],
    }
    candidates: list[dict[str, Any]] = []
    for family, weights in (
        ("random_forest", [None, "balanced", "balanced_subsample"]),
        ("extra_trees", [None, "balanced"]),
    ):
        sampled = ParameterSampler(
            {**grid, "class_weight": weights}, n_iter=24, random_state=42
        )
        candidates.extend(
            {"family": family, "params": dict(p), "reference": False} for p in sampled
        )
    candidates.append(
        {
            "family": "random_forest",
            "params": {
                "n_estimators": 128,
                "max_depth": 24,
                "min_samples_leaf": 3,
                "max_features": "sqrt",
                "class_weight": "balanced",
            },
            "reference": True,
        }
    )
    return candidates


def make_estimator(candidate: dict[str, Any], seed: int = 42) -> Any:
    estimator = (
        RandomForestClassifier
        if candidate["family"] == "random_forest"
        else ExtraTreesClassifier
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                estimator(**candidate["params"], n_jobs=1, random_state=seed),
            ),
        ]
    )


def ranking_key(result: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        -result["mean_macro_f1"],
        result["mean_log_loss"],
        result["mean_model_bytes"],
        canonical(result["candidate"]),
    )


def train_experiment(directory: Path, jobs: int = 4) -> dict[str, Any]:
    config = _validate_run(directory)
    if not _completed(directory, "prepare"):
        raise ValueError("Prepare the experiment before training")
    if _completed(directory, "train"):
        return dict(read_json(directory / "training.json"))
    if not 1 <= jobs <= 4:
        raise ValueError("Use between one and four workers")
    data = pd.read_csv(directory / "dataset.csv")
    development = data.loc[data.split == "development"].reset_index(drop=True)
    started = time.perf_counter()
    features = build_feature_matrix(development.sequence)
    folds = []
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
        for fold, (fit, valid) in enumerate(
            cv.split(features, development.label, development.group)
        ):
            if set(development.iloc[fit].label) != set(CLASSES) or set(
                development.iloc[valid].label
            ) != set(CLASSES):
                raise ValueError("Every development fold must contain all classes")
            if set(development.iloc[fit].group) & set(development.iloc[valid].group):
                raise ValueError("Similarity group leakage in training")
            folds.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "fit": fit.tolist(),
                    "validation": valid.tolist(),
                }
            )
    design = {
        "candidates": candidate_configurations(),
        "folds": folds,
        "accessions": development.accession.tolist(),
    }
    design_path = directory / "design.json"
    if design_path.exists() and read_json(design_path) != design:
        raise ValueError("Training design changed; use a new experiment")
    atomic_json(design_path, design)
    design_hash = checksum(design_path)
    (directory / "folds").mkdir(exist_ok=True)

    def fit_one(index: int, fold_index: int) -> dict[str, Any]:
        path = directory / "folds" / f"{index:02d}-{fold_index}.json"
        if path.exists():
            cached = read_json(path)
            if cached["design_sha256"] != design_hash:
                raise ValueError("Cached fold design differs")
            return dict(cached)
        candidate, fold = design["candidates"][index], folds[fold_index]
        fit, valid = fold["fit"], fold["validation"]
        model = make_estimator(candidate, fold["seed"])
        start = time.perf_counter()
        model.fit(features.iloc[fit], development.label.iloc[fit])
        probabilities = model.predict_proba(features.iloc[valid])
        metrics = classification_metrics(
            development.label.iloc[valid],
            model.predict(features.iloc[valid]),
            probabilities,
            list(model.classes_),
        )
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        result = {
            "design_sha256": design_hash,
            "candidate_index": index,
            "fold_index": fold_index,
            "metrics": metrics,
            "model_bytes": buffer.tell(),
            "seconds": time.perf_counter() - start,
        }
        atomic_json(path, result)
        return result

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [
            pool.submit(fit_one, c, f)
            for c in range(len(design["candidates"]))
            for f in range(len(folds))
        ]
        results = [future.result() for future in as_completed(futures)]
    comparisons = []
    for i, candidate in enumerate(design["candidates"]):
        selected = [r for r in results if r["candidate_index"] == i]
        comparisons.append(
            {
                "candidate_index": i,
                "candidate": candidate,
                "mean_macro_f1": float(
                    np.mean([r["metrics"]["macro_f1"] for r in selected])
                ),
                "std_macro_f1": float(
                    np.std([r["metrics"]["macro_f1"] for r in selected])
                ),
                "mean_log_loss": float(
                    np.mean([r["metrics"]["log_loss"] for r in selected])
                ),
                "mean_model_bytes": float(
                    np.mean([r["model_bytes"] for r in selected])
                ),
            }
        )
    comparisons.sort(key=ranking_key)
    winner = min(
        (r for r in comparisons if not r["candidate"]["reference"]), key=ranking_key
    )
    model = make_estimator(winner["candidate"])
    model.fit(features, development.label)
    preparation = read_json(directory / "preparation.json")
    identity = preparation["experiment_id"] + "-" + winner["candidate"]["family"]
    metadata = {
        **preparation,
        "name": "MAP-ExPLoc human CPU candidate",
        "scope": "Reviewed human canonical proteins; five single compartments",
        "model_id": identity,
        "model_family": winner["candidate"]["family"],
        "evaluation_status": "development_only",
        "sample_count": len(development),
        "best_params": winner["candidate"]["params"],
        "best_cv_score": winner["mean_macro_f1"],
        "software_versions": config["software"],
    }
    artifact = directory / "candidate.joblib"
    temporary = directory / "candidate.joblib.part"
    save_model_artifact(model, temporary, metadata=metadata)
    temporary.replace(artifact)
    checks = validate_candidate(artifact, features.iloc[:10])
    report = {
        "winner": winner,
        "comparisons": comparisons,
        "fold_count": len(results),
        "workers": jobs,
        "training_seconds_this_invocation": time.perf_counter() - started,
        "sum_fit_seconds": sum(r["seconds"] for r in results),
        "artifact_sha256": checksum(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "model_id": identity,
        "validation": checks,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
    }
    atomic_json(directory / "training.json", report)
    _commit_stage(
        directory,
        "train",
        [
            "candidate.joblib",
            "training.json",
            "design.json",
            *[
                str(p.relative_to(directory))
                for p in sorted((directory / "folds").glob("*.json"))
            ],
        ],
    )
    return report


def validate_candidate(path: Path, features: pd.DataFrame) -> dict[str, bool]:
    artifact = load_model_artifact(path)
    model = artifact.model
    if (
        list(model.classes_) != sorted(CLASSES)
        or tuple(model.classes_) != artifact.classes
    ):
        raise ValueError("Artifact class order is incompatible")
    probabilities = model.predict_proba(features)
    if not np.isfinite(probabilities).all():
        raise ValueError("Artifact probabilities are not finite")
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
    explanation = ShapExplainer(model).explain_sample(features)
    np.testing.assert_allclose(
        explanation["expected_value"] + explanation["shap_values"].sum(axis=2),
        probabilities,
        atol=1e-7,
    )
    return {
        "reload": True,
        "schema": True,
        "class_order": True,
        "shap_additivity": True,
    }


def bootstrap_comparison(
    truth: np.ndarray,
    predicted: np.ndarray,
    reference: np.ndarray,
    groups: np.ndarray,
    repetitions: int = 2000,
) -> dict[str, Any]:
    """Paired resampling of whole similarity groups with a fixed class universe."""
    rng = np.random.default_rng(42)
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in unique]
    labels = {c: i for i, c in enumerate(sorted(CLASSES))}
    actual = np.array([labels[c] for c in truth])
    candidate = np.array([labels[c] for c in predicted])
    baseline = np.array([labels[c] for c in reference])

    def score(indices: np.ndarray, guesses: np.ndarray) -> float:
        matrix = np.bincount(
            actual[indices] * 5 + guesses[indices], minlength=25
        ).reshape(5, 5)
        denom = matrix.sum(axis=0) + matrix.sum(axis=1)
        return float(
            np.divide(
                2 * matrix.diagonal(), denom, out=np.zeros(5), where=denom != 0
            ).mean()
        )

    scores, differences, degenerate = [], [], 0
    for _ in range(repetitions):
        indices = np.concatenate(
            [members[i] for i in rng.integers(0, len(unique), len(unique))]
        )
        degenerate += len(np.unique(actual[indices])) < 5
        value = score(indices, candidate)
        scores.append(value)
        differences.append(value - score(indices, baseline))
    return {
        "method": "paired whole-group percentile bootstrap",
        "repetitions": repetitions,
        "seed": 42,
        "macro_f1_95_interval": np.quantile(scores, [0.025, 0.975]).tolist(),
        "macro_f1_difference_95_interval": np.quantile(
            differences, [0.025, 0.975]
        ).tolist(),
        "replicates_missing_classes": int(degenerate),
        "missing_class_policy": "retain with fixed five-class macro average",
    }


def promotion_decision(report: dict[str, Any]) -> dict[str, Any]:
    """Evaluate predeclared gates; insufficient evidence is never a passing score."""
    reasons = []
    if report["evaluation_status"] != "independent_confirmation":
        reasons.append("Independent confirmation is unavailable")
    interval = report.get("bootstrap", {}).get("macro_f1_difference_95_interval")
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in interval)
        or not 0 < interval[0] <= interval[1] <= 1
    ):
        reasons.append(
            "Positive paired uncertainty bound is missing or not established"
        )
    metrics, baseline = report["evaluation"], report["reference_evaluation"]
    if metrics["macro_f1"] - baseline["macro_f1"] < 0.03:
        reasons.append("Macro-F1 gain is below 0.03")
    for label in CLASSES:
        if (
            metrics["classification_report"][label]["f1-score"]
            < baseline["classification_report"][label]["f1-score"] - 0.05
        ):
            reasons.append(f"{label} F1 decreased by more than 0.05")
    if metrics["log_loss"] > baseline["log_loss"]:
        reasons.append("Log loss increased")
    for metric in ("prediction_one_ms", "prediction_batch_ms", "shap_one_ms"):
        if (
            report["runtime"]["candidate"][metric]
            > 2 * report["runtime"]["reference"][metric]
        ):
            reasons.append(f"{metric} exceeds twice the reference")
    if not all(
        report["validation"].get(k) is True
        for k in ("reload", "schema", "class_order", "shap_additivity")
    ):
        reasons.append("Artifact validation failed")
    return {"eligible": not reasons, "reasons": reasons}


def matched_runtime(
    candidate: Any, reference: Any, sequences: list[str]
) -> dict[str, Any]:
    features = build_feature_matrix(sequences[:100])
    one = features.iloc[:1]
    models = {"candidate": candidate, "reference": reference}
    explainers = {name: ShapExplainer(model) for name, model in models.items()}
    results: dict[str, Any] = {name: {} for name in models}
    for name, model in models.items():
        model.predict_proba(features)
        explainers[name].explain_sample(one)
    for operation in (
        "prediction_one_ms",
        "prediction_batch_ms",
        "shap_one_ms",
        "end_to_end_one_ms",
        "end_to_end_batch_ms",
    ):
        timings: dict[str, list[float]] = {name: [] for name in models}
        for iteration in range(5):
            for name in (
                list(models) if iteration % 2 == 0 else list(reversed(models))
            ):
                model = models[name]
                start = time.perf_counter()
                if operation == "shap_one_ms":
                    explainers[name].explain_sample(one)
                elif operation.startswith("end_to_end"):
                    model.predict_proba(
                        build_feature_matrix(
                            sequences[: 1 if "one" in operation else 100]
                        )
                    )
                else:
                    model.predict_proba(one if "one" in operation else features)
                timings[name].append((time.perf_counter() - start) * 1000)
        for name in models:
            results[name][operation] = float(np.median(timings[name]))
    measured = []
    for _ in range(5):
        start = time.perf_counter()
        build_feature_matrix(sequences[:1])
        measured.append((time.perf_counter() - start) * 1000)
    return {
        **results,
        "feature_one_ms": float(np.median(measured)),
        "batch_size": len(features),
        "repetitions": 5,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "software_versions": software(),
        "protocol": (
            "Warm five-repetition median, alternating model order, "
            "identical inputs, one estimator worker"
        ),
    }


def evaluate_experiment(directory: Path) -> dict[str, Any]:
    config = _validate_run(directory)
    if not _completed(directory, "prepare") or not _completed(directory, "train"):
        raise ValueError("Prepare and train before evaluation")
    if _completed(directory, "evaluate"):
        return dict(read_json(directory / "evaluation.json"))
    preparation = read_json(directory / "preparation.json")
    training = read_json(directory / "training.json")
    data = pd.read_csv(directory / "dataset.csv")
    independent = preparation["confirmation"]["status"] == "available"
    evaluation = (
        data.loc[data.split == "confirmation"]
        if independent
        else pd.read_csv(directory / "historical.csv")
    )
    status = "independent_confirmation" if independent else "historical_diagnostic"
    candidate = load_model_artifact(directory / "candidate.joblib").model
    reference_path = (
        Path(config["reference_root"]) / "examples/models/human-baseline.joblib"
    )
    reference = load_model_artifact(reference_path).model
    # The reference artifact used multiple inference workers; compare at one worker.
    reference.set_params(**{f"{reference.steps[-1][0]}__n_jobs": 1})
    features = build_feature_matrix(evaluation.sequence)
    probabilities = candidate.predict_proba(features)
    predicted = candidate.predict(features)
    baseline_probabilities = reference.predict_proba(features)
    baseline_predicted = reference.predict(features)
    classes = list(candidate.classes_)
    np.testing.assert_array_equal(classes, reference.classes_)
    development = data.loc[data.split == "development"]
    dummy = DummyClassifier(strategy="prior").fit(
        np.zeros((len(development), 1)), development.label
    )
    dummy_x = np.zeros((len(evaluation), 1))
    output = evaluation[["accession", "label", "group"]].copy()
    output["prediction"] = predicted
    output["reference_prediction"] = baseline_predicted
    for i, label in enumerate(classes):
        output[f"probability_{label}"] = probabilities[:, i]
        output[f"reference_probability_{label}"] = baseline_probabilities[:, i]
    output.to_csv(directory / "predictions.csv", index=False)
    report = {
        "experiment_id": preparation["experiment_id"],
        "evaluation_status": status,
        "confirmation": preparation["confirmation"],
        "evaluated_count": len(evaluation),
        "evaluation": classification_metrics(
            evaluation.label, predicted, probabilities, classes
        ),
        "reference_evaluation": classification_metrics(
            evaluation.label, baseline_predicted, baseline_probabilities, classes
        ),
        "dummy_evaluation": classification_metrics(
            evaluation.label,
            dummy.predict(dummy_x),
            dummy.predict_proba(dummy_x),
            classes,
        ),
        "bootstrap": bootstrap_comparison(
            evaluation.label.to_numpy(),
            predicted,
            baseline_predicted,
            evaluation.group.to_numpy(),
        ),
        "runtime": matched_runtime(candidate, reference, evaluation.sequence.tolist()),
        "validation": validate_candidate(
            directory / "candidate.joblib", features.iloc[:10]
        ),
        "model_id": training["model_id"],
        "artifact_sha256": checksum(directory / "candidate.joblib"),
        "reference_sha256": checksum(reference_path),
        "artifact_bytes": training["artifact_bytes"],
        "dataset_sha256": preparation["dataset_sha256"],
        "software_versions": config["software"],
        "test_refitted": False,
    }
    report["promotion"] = promotion_decision(report)
    atomic_json(directory / "evaluation.json", report)
    _commit_stage(directory, "evaluate", ["evaluation.json", "predictions.csv"])
    return report


def promote_experiment(directory: Path, repository: Path) -> dict[str, Any]:
    """Promote verified independent results and retain rollback assets."""
    _validate_run(directory)
    for stage in ("prepare", "train", "evaluate"):
        if not _completed(directory, stage):
            raise ValueError(f"Missing completed stage: {stage}")
    report = read_json(directory / "evaluation.json")
    decision = promotion_decision(report)
    if not decision["eligible"]:
        outcome = {
            "status": "retained_version_1",
            **decision,
            "model_id": report["model_id"],
        }
        atomic_json(directory / "promotion.json", outcome)
        return outcome
    repository = repository.resolve()
    previous = manifest_selection(repository)
    previous.load()
    artifact = load_model_artifact(directory / "candidate.joblib")
    if checksum(directory / "candidate.joblib") != report["artifact_sha256"]:
        raise ValueError("Evaluated candidate checksum changed")
    metadata = {
        **artifact.metadata,
        "evaluation_status": report["evaluation_status"],
        "evaluation": report["evaluation"],
        "runtime": report["runtime"],
        "promotion": decision,
    }
    destination = repository / "examples/models" / (report["model_id"] + ".joblib")
    if destination.exists():
        loaded = load_model_artifact(destination)
        if loaded.metadata != metadata:
            raise ValueError("Published model exists with different metadata")
    else:
        save_model_artifact(artifact.model, destination, metadata=metadata)
    manifest = repository / "config/default-model.json"
    backup = manifest.with_name(f"default-model.previous-{previous.sha256}.json")
    if not backup.exists():
        atomic_json(backup, read_json(manifest))
    atomic_json(
        manifest,
        {
            "schema_version": 1,
            "artifact_path": str(destination.relative_to(repository)),
            "model_id": report["model_id"],
            "sha256": checksum(destination),
        },
    )
    manifest_selection(repository).load()
    outcome = {
        "status": "promoted",
        "artifact_sha256": checksum(destination),
        "model_id": report["model_id"],
        **decision,
    }
    atomic_json(directory / "promotion.json", outcome)
    return outcome
