"""Development-only nested validation, calibration and auditable research artifacts.

No historical outcome is scored here. A research artifact has a separate format
from the served tree/SHAP artifact and cannot silently replace it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from .baseline import CLASSES, checksum
from .evaluation import classification_metrics
from .experiments import atomic_json, canonical, read_json, software
from .features import AMINO_ACIDS, FEATURE_NAMES, build_feature_matrix
from .validation import grouped_splits

SEEDS = (20260918, 20260919)
GLOBAL = [n for n in FEATURE_NAMES if not n.startswith("dp_")]
TERMINAL = [f"{end}50_{aa}" for end in ("n", "c") for aa in AMINO_ACIDS]


def research_features(sequences: Any) -> pd.DataFrame:
    """Retain the service schema; add explicitly versioned research-only features."""
    base = build_feature_matrix(sequences).reset_index(drop=True)
    rows = []
    for sequence in sequences:
        sequence = "".join(str(sequence).split()).upper()
        row = {}
        for end, fragment in (("n", sequence[:50]), ("c", sequence[-50:])):
            row.update(
                {
                    f"{end}50_{aa}": fragment.count(aa) / len(fragment)
                    for aa in AMINO_ACIDS
                }
            )
        rows.append(row)
    return pd.concat([base, pd.DataFrame(rows, columns=TERMINAL)], axis=1)


def log_length(values: np.ndarray) -> np.ndarray:
    transformed = np.asarray(values, dtype=float).copy()
    transformed[:, 0] = np.log1p(transformed[:, 0])
    return transformed


def candidates() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = [
        {"id": "dummy", "family": "dummy", "features": "length", "complexity": 0},
        {
            "id": "length",
            "family": "logistic",
            "features": "length",
            "C": 1.0,
            "complexity": 1,
        },
    ]
    for feature, rank in (("global", 2), ("terminal", 3), ("full", 4)):
        for c in (0.01, 0.1, 1.0):
            result.append(
                {
                    "id": f"lr_{feature}_{c:g}",
                    "family": "logistic",
                    "features": feature,
                    "C": c,
                    "complexity": rank,
                }
            )
    for name, feature, trees, depth, leaf, fraction, rank in (
        ("rf_global", "global", 128, 24, 3, 0.5, 5),
        ("rf_terminal", "terminal", 128, 24, 3, 0.5, 6),
        ("reference", "full", 128, 24, 3, "sqrt", 7),
        ("rf_expanded", "full", 512, None, 1, 0.5, 8),
    ):
        result.append(
            {
                "id": name,
                "family": "rf",
                "features": feature,
                "complexity": rank,
                "params": {
                    "n_estimators": trees,
                    "max_depth": depth,
                    "min_samples_leaf": leaf,
                    "max_features": fraction,
                    "class_weight": "balanced",
                },
            }
        )
    return result


def feature_columns(candidate: dict[str, Any]) -> list[str]:
    return {
        "length": ["length"],
        "global": GLOBAL,
        "terminal": GLOBAL + TERMINAL,
        "full": list(FEATURE_NAMES),
    }[candidate["features"]]


def make_model(candidate: dict[str, Any], seed: int) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        (
            "columns",
            ColumnTransformer(
                [("selected", "passthrough", feature_columns(candidate))],
                remainder="drop",
            ),
        )
    ]
    if candidate["family"] == "logistic":
        steps += [
            ("log_length", FunctionTransformer(log_length)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=candidate["C"], max_iter=3000, solver="lbfgs", random_state=seed
                ),
            ),
        ]
    elif candidate["family"] == "rf":
        steps += [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    **candidate["params"], random_state=seed, n_jobs=1
                ),
            ),
        ]
    else:
        steps += [("classifier", DummyClassifier(strategy="prior"))]
    return Pipeline(steps)


def score(y: Any, probabilities: np.ndarray) -> dict[str, Any]:
    return classification_metrics(
        y,
        np.asarray(CLASSES)[probabilities.argmax(axis=1)],
        probabilities,
        list(CLASSES),
    )


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    logits = np.log(np.clip(probabilities, 1e-15, 1)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return np.asarray(values / values.sum(axis=1, keepdims=True))


def fit_temperature(y: Any, probabilities: np.ndarray) -> float:
    score(y, probabilities)  # Validate alignment and probabilities before optimizing.
    indices = np.array([list(CLASSES).index(label) for label in y])

    def loss(t: float) -> float:
        p = temperature_scale(probabilities, t)
        return float(-np.log(p[np.arange(len(indices)), indices]).mean())

    result = minimize_scalar(loss, bounds=(0.5, 3.0), method="bounded")
    if not result.success:
        raise ValueError("Temperature optimization did not converge")
    return float(result.x)


def choose(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Preregistered practical tolerance rule, evaluated only on inner predictions."""
    leader = min(
        results,
        key=lambda r: (
            -r["metrics"]["macro_f1"],
            r["metrics"]["log_loss"],
            r["candidate"]["id"],
        ),
    )
    top = leader["metrics"]
    eligible = [
        r
        for r in results
        if r["metrics"]["macro_f1"] >= top["macro_f1"] - 0.01
        and r["metrics"]["log_loss"] <= top["log_loss"] + 0.02
        and all(
            r["metrics"]["classification_report"][c]["f1-score"]
            >= top["classification_report"][c]["f1-score"] - 0.05
            for c in CLASSES
        )
    ]
    return min(
        eligible,
        key=lambda r: (
            r["candidate"]["complexity"],
            r["candidate"].get("C", 0),
            r["candidate"]["id"],
        ),
    )


def has_location_note(annotations: str) -> bool:
    return any(
        str(text.get("value", "")).strip()
        for comment in json.loads(annotations)
        for text in comment.get("note", {}).get("texts", [])
    )


def publications(evidence: str) -> list[str]:
    return sorted(
        {
            str(e["id"])
            for e in json.loads(evidence)
            if e.get("source") == "PubMed" and e.get("id")
        }
    )


def study_groups(frame: pd.DataFrame) -> np.ndarray:
    """Connected components of sequence-group OR supporting-publication links."""
    parent = list(range(len(frame)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen: dict[str, int] = {}
    for i, row in enumerate(frame.itertuples()):
        keys = [f"sequence:{row.group}"] + [
            f"pubmed:{p}" for p in publications(row.evidence)
        ]
        for key in keys:
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    return np.asarray([f"study_{root(i)}" for i in range(len(frame))])


def load_development(
    source: Path, cohort: str = "all"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(source)
    required = [
        "accession",
        "sequence",
        "label",
        "group",
        "split",
        "evidence",
        "localization_annotations",
    ]
    if not set(required).issubset(frame) or frame[required].isna().any().any():
        raise ValueError("Required cohort fields are absent or missing")
    if not set(frame.split).issubset(
        {"development", "excluded_historical", "confirmation"}
    ):
        raise ValueError(
            "Only explicit development and reserved partition names are allowed"
        )
    normalized = frame.sequence.map(lambda s: "".join(str(s).split()).upper())
    if frame.accession.duplicated().any() or normalized.duplicated().any():
        raise ValueError("Duplicate accessions or normalized sequences in cohort")
    development = frame.loc[frame.split == "development"].copy()
    reserved = frame.loc[frame.split != "development"]
    for field in ("accession", "sequence", "group"):
        if set(development[field]) & set(reserved[field]):
            raise ValueError(f"Reserved/development overlap in {field}")
    development["has_location_note"] = development.localization_annotations.map(
        has_location_note
    )
    audit: dict[str, Any] = {
        "source_rows": len(frame),
        "reserved_rows": len(reserved),
        "development_before_filter": len(development),
        "location_note_records": int(development.has_location_note.sum()),
        "duplicate_accessions": 0,
        "duplicate_sequences": 0,
        "reserved_overlap": 0,
    }
    if cohort == "note_free":
        development = development.loc[~development.has_location_note]
    elif cohort != "all":
        raise ValueError("Unknown cohort")
    development = development.sort_values("accession").reset_index(drop=True)
    if set(development.label) != set(CLASSES):
        raise ValueError("Development must contain exactly the five declared classes")
    audit.update(
        {
            "development_rows": len(development),
            "class_counts": {
                str(k): int(v) for k, v in development.label.value_counts().items()
            },
            "sequence_groups": int(development.group.nunique()),
            "largest_sequence_group": int(development.groupby("group").size().max()),
            "mixed_label_groups": int(
                (development.groupby("group").label.nunique() > 1).sum()
            ),
        }
    )
    return development, audit


def implementation_hash() -> str:
    package = Path(__file__).parent
    content = [
        (str(p.relative_to(package)), checksum(p))
        for p in sorted(package.rglob("*.py"))
    ]
    return hashlib.sha256(canonical(content).encode()).hexdigest()


def write_record(path: Path, value: Any) -> None:
    atomic_json(path, value)
    path.with_suffix(".sha256").write_text(checksum(path) + "\n")


def read_record(path: Path) -> Any:
    if not path.with_suffix(".sha256").exists() or (
        path.with_suffix(".sha256").read_text().strip() != checksum(path)
    ):
        raise ValueError(f"Changed or incomplete result: {path}")
    return read_json(path)


def fit_checked(model: Pipeline, x: pd.DataFrame, y: Any) -> Pipeline:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y)
    if list(model.classes_) != list(CLASSES):
        raise ValueError("Fitted class ordering differs from protocol")
    return model


def inner_fit(
    candidate: dict[str, Any],
    x: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> dict[str, Any]:
    probabilities = np.zeros((len(y), len(CLASSES)))
    counts = np.zeros(len(y), dtype=int)
    started = time.perf_counter()
    for train, valid in splits:
        model = fit_checked(make_model(candidate, seed), x.iloc[train], y[train])
        probabilities[valid] = model.predict_proba(x.iloc[valid])
        counts[valid] += 1
    if not (counts == 1).all():
        raise ValueError("Inner predictions do not cover every protein exactly once")
    return {
        "candidate": candidate,
        "metrics": score(y, probabilities),
        "probabilities": probabilities.tolist(),
        "seconds": time.perf_counter() - started,
    }


def group_bootstrap(
    y: np.ndarray,
    probabilities: np.ndarray,
    reference: np.ndarray,
    groups: np.ndarray,
    repetitions: int = 2000,
) -> dict[str, Any]:
    """Repeat predictions stay paired inside each resampled biological group."""
    _, group_indices = np.unique(groups, return_inverse=True)
    n_groups = int(group_indices.max()) + 1
    actual = np.array([list(CLASSES).index(c) for c in y])

    def contributions(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrices = np.zeros((n_groups, 5, 5))
        np.add.at(matrices, (group_indices, actual, p.argmax(axis=1)), 1)
        loss = -np.log(np.clip(p[np.arange(len(y)), actual], 1e-15, 1))
        return matrices, np.bincount(group_indices, weights=loss, minlength=n_groups)

    matrix, loss = contributions(probabilities)
    ref_matrix, ref_loss = contributions(reference)
    counts = np.bincount(group_indices, minlength=n_groups)

    def f1(m: np.ndarray) -> float:
        denominator = m.sum(axis=0) + m.sum(axis=1)
        return float(
            np.divide(
                2 * m.diagonal(), denominator, out=np.zeros(5), where=denominator != 0
            ).mean()
        )

    rng = np.random.default_rng(SEEDS[0])
    records = []
    missing = 0
    for _ in range(repetitions):
        selected = rng.integers(0, n_groups, n_groups)
        m = matrix[selected].sum(axis=0)
        r = ref_matrix[selected].sum(axis=0)
        missing += int((m.sum(axis=1) == 0).any())
        records.append(
            [
                f1(m),
                f1(m) - f1(r),
                (loss[selected].sum() - ref_loss[selected].sum())
                / counts[selected].sum(),
            ]
        )
    values = np.asarray(records)
    return {
        "method": "paired whole-group percentile; all repeats kept together",
        "seed": SEEDS[0],
        "repetitions": repetitions,
        "groups": n_groups,
        "macro_f1_95_interval": np.quantile(values[:, 0], [0.025, 0.975]).tolist(),
        "macro_f1_difference_97_5_interval": np.quantile(
            values[:, 1], [0.0125, 0.9875]
        ).tolist(),
        "log_loss_difference_97_5_interval": np.quantile(
            values[:, 2], [0.0125, 0.9875]
        ).tolist(),
        "replicates_missing_classes": missing,
        "limitation": (
            "Conditional on fitted predictions; excludes full training "
            "and selection uncertainty"
        ),
    }


def summarize(
    frame: pd.DataFrame, records: list[dict[str, Any]], groups: np.ndarray
) -> dict[str, Any]:
    names = list(records[0]["probabilities"])
    labels = np.tile(frame.label.to_numpy(), len(set(r["seed"] for r in records)))
    repeats = sorted(set(r["seed"] for r in records))
    arrays = {}
    for name in names:
        array = np.zeros((len(repeats), len(frame), 5))
        for record in records:
            array[repeats.index(record["seed"]), record["validation"]] = record[
                "probabilities"
            ][name]
        arrays[name] = array.reshape(-1, 5)
    group_rows = np.tile(groups, len(repeats))
    metrics = {name: score(labels, p) for name, p in arrays.items()}
    per_seed = {
        str(seed): {
            name: score(frame.label, p[i * len(frame) : (i + 1) * len(frame)])
            for name, p in arrays.items()
        }
        for i, seed in enumerate(repeats)
    }
    intervals = {
        name: group_bootstrap(labels, p, arrays["reference"], group_rows)
        for name, p in arrays.items()
        if name in ("reference", "selected", "calibrated")
    }
    calibration = group_bootstrap(
        labels, arrays["calibrated"], arrays["selected"], group_rows
    )
    gain = metrics["selected"]["log_loss"] - metrics["calibrated"]["log_loss"]
    retain = bool(
        gain >= 0.02 and calibration["log_loss_difference_97_5_interval"][1] < 0
    )
    lengths = frame.sequence.str.len().to_numpy()
    masks = {
        "short_under_200": lengths < 200,
        "medium_200_599": (lengths >= 200) & (lengths < 600),
        "long_600_plus": lengths >= 600,
        "location_note": frame.has_location_note.to_numpy(),
        "note_free": ~frame.has_location_note.to_numpy(),
        "sequence_singleton": frame.group.map(frame.group.value_counts()).to_numpy()
        == 1,
    }
    subgroups = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        repeated = np.tile(mask, len(repeats))
        subgroups[name] = {
            "proteins": int(mask.sum()),
            "small_sample": int(mask.sum()) < 50,
            **{
                m: score(labels[repeated], arrays[m][repeated])
                for m in ("selected", "reference", "calibrated")
            },
        }
    # Equal total weight per independent group, as a sensitivity to large families.
    _, inverse, counts = np.unique(group_rows, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse]
    from sklearn.metrics import f1_score

    equal_group = {
        name: float(
            f1_score(
                labels,
                np.asarray(CLASSES)[p.argmax(axis=1)],
                labels=list(CLASSES),
                average="macro",
                sample_weight=weights,
                zero_division=0,
            )
        )
        for name, p in arrays.items()
    }
    return {
        "metrics": metrics,
        "per_seed": per_seed,
        "bootstrap": intervals,
        "calibration_comparison": calibration,
        "retain_calibration": retain,
        "calibration_log_loss_reduction": gain,
        "subgroups": subgroups,
        "group_equal_weight_macro_f1": equal_group,
        "selection_counts": {
            c: sum(r["selected"] == c for r in records)
            for c in sorted({r["selected"] for r in records})
        },
        "selected_seed_range": float(
            np.ptp([per_seed[str(s)]["selected"]["macro_f1"] for s in repeats])
        ),
        "status": "internal_development_validation; not independent confirmation",
    }


def run_research(
    source: Path,
    directory: Path,
    jobs: int = 3,
    grouping: str = "sequence",
    cohort: str = "all",
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    if not 1 <= jobs <= 4:
        raise ValueError("Use one to four workers")
    directory.mkdir(parents=True, exist_ok=True)
    frame, audit = load_development(source, cohort)
    groups = frame.group.to_numpy() if grouping == "sequence" else study_groups(frame)
    if grouping not in ("sequence", "study"):
        raise ValueError("Grouping must be sequence or study")
    configs = candidates()
    config = {
        "protocol": 1,
        "source_sha256": checksum(source),
        "source": str(source.resolve()),
        "code_sha256": implementation_hash(),
        "software": software(),
        "scipy": __import__("scipy").__version__,
        "seeds": list(seeds),
        "grouping": grouping,
        "cohort": cohort,
        "candidates": configs,
        "outer_folds": 3,
        "inner_folds": 3,
        "jobs": jobs,
        "evaluation_status": "internal_development_only",
    }
    config_path = directory / "protocol.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError(
            "Protocol, source, code or software changed; use a new run directory"
        )
    if (directory / "complete.json").exists():
        manifest = read_record(directory / "complete.json")
        for name, digest in manifest.items():
            if checksum(directory / name) != digest:
                raise ValueError(f"Completed output changed: {name}")
        return dict(read_record(directory / "results.json"))
    atomic_json(config_path, config)
    run_hash = checksum(config_path)
    design = []
    for seed in seeds:
        for index, (train, valid) in enumerate(
            grouped_splits(frame.label, groups, 3, seed)
        ):
            inner = grouped_splits(
                frame.label.iloc[train], groups[train], 3, seed + index + 100
            )
            design.append(
                {
                    "seed": seed,
                    "fold": index,
                    "train": train.tolist(),
                    "validation": valid.tolist(),
                    "inner": [
                        {"train": t.tolist(), "validation": v.tolist()}
                        for t, v in inner
                    ],
                }
            )
    frozen = {
        "accessions": frame.accession.tolist(),
        "groups": groups.tolist(),
        "folds": design,
    }
    if (directory / "design.json").exists() and read_json(
        directory / "design.json"
    ) != frozen:
        raise ValueError("Fold design changed")
    atomic_json(directory / "design.json", frozen)
    started = time.perf_counter()
    x = research_features(frame.sequence)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Nonfinite features")
    audit.update(
        {
            "feature_missingness": 0,
            "research_feature_count": x.shape[1],
            "analysis_groups": len(np.unique(groups)),
            "largest_analysis_group": int(pd.Series(groups).value_counts().max()),
        }
    )
    atomic_json(directory / "data-audit.json", audit)
    y = frame.label.to_numpy()
    records = []
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=jobs) as pool:
        for fold in design:
            path = directory / "folds" / f'{fold["seed"]}-{fold["fold"]}.json'
            if path.exists():
                cached = read_record(path)
                if (
                    cached["run_sha256"] != run_hash
                    or cached["validation"] != fold["validation"]
                ):
                    raise ValueError("Cached fold does not match protocol/membership")
                records.append(cached)
                continue
            train, valid = np.array(fold["train"]), np.array(fold["validation"])
            splits = [
                (np.array(s["train"]), np.array(s["validation"])) for s in fold["inner"]
            ]
            futures = [
                pool.submit(inner_fit, c, x.iloc[train], y[train], splits, fold["seed"])
                for c in configs
            ]
            inner_results = [f.result() for f in futures]
            selected = choose(inner_results)
            t = fit_temperature(y[train], np.array(selected["probabilities"]))
            predictions = {}
            fitted = {}
            for c in configs:
                model = fit_checked(
                    make_model(c, fold["seed"]), x.iloc[train], y[train]
                )
                predictions[c["id"]] = model.predict_proba(x.iloc[valid]).tolist()
                fitted[c["id"]] = model
            winner_id = selected["candidate"]["id"]
            predictions["selected"] = predictions[winner_id]
            predictions["calibrated"] = temperature_scale(
                np.array(predictions[winner_id]), t
            ).tolist()
            # Truncation stress is deliberately outside model selection.
            truncated = frame.sequence.iloc[valid].map(
                lambda s: s[min(25, len(s) - 1) :]
            )
            stress = score(
                y[valid], fitted[winner_id].predict_proba(research_features(truncated))
            )
            result = {
                "run_sha256": run_hash,
                "seed": fold["seed"],
                "fold": fold["fold"],
                "validation": valid.tolist(),
                "selected": winner_id,
                "temperature": t,
                "probabilities": predictions,
                "inner_results": [
                    {k: v for k, v in r.items() if k != "probabilities"}
                    for r in inner_results
                ],
                "n_terminal_25_residue_deletion": stress,
            }
            write_record(path, result)
            records.append(result)
            print(
                f'Completed seed {fold["seed"]}, fold {fold["fold"]}; '
                f"selected {winner_id}",
                flush=True,
            )
        report = summarize(frame, records, groups)
        # Frozen full-development selection; outer outcomes never choose parameters.
        final_splits = grouped_splits(y, groups, 3, seeds[0] + 100)
        futures = [
            pool.submit(inner_fit, c, x, y, final_splits, seeds[0]) for c in configs
        ]
        final_results = [f.result() for f in futures]
        winner = choose(final_results)
        temperature = (
            fit_temperature(y, np.array(winner["probabilities"]))
            if report["retain_calibration"]
            else 1.0
        )
        model = fit_checked(make_model(winner["candidate"], seeds[0]), x, y)
        artifact = {
            "kind": "mapexploc-research-v1",
            "model": model,
            "temperature": temperature,
            "candidate": winner["candidate"],
            "classes": list(CLASSES),
            "feature_names": list(x.columns),
            "run_sha256": run_hash,
            "evaluation_status": "development_only_not_served",
        }
        joblib.dump(artifact, directory / "research-model.joblib")
        restored = joblib.load(directory / "research-model.joblib")
        np.testing.assert_allclose(
            model.predict_proba(x.iloc[:10]),
            restored["model"].predict_proba(x.iloc[:10]),
            rtol=0,
            atol=0,
        )
        report.update(
            {
                "final_candidate": winner["candidate"],
                "final_temperature": temperature,
                "final_selection": [
                    {k: v for k, v in r.items() if k != "probabilities"}
                    for r in final_results
                ],
                "artifact_sha256": checksum(directory / "research-model.joblib"),
                "artifact_bytes": (directory / "research-model.joblib").stat().st_size,
                "seconds_this_invocation": time.perf_counter() - started,
                "platform": platform.platform(),
                "promotion": False,
            }
        )
    prediction_rows = []
    for record in records:
        for method, probabilities in record["probabilities"].items():
            for i, p in zip(record["validation"], probabilities):
                prediction_rows.append(
                    {
                        "accession": frame.accession.iloc[i],
                        "group": groups[i],
                        "label": y[i],
                        "seed": record["seed"],
                        "fold": record["fold"],
                        "method": method,
                        **{f"p_{c}": value for c, value in zip(CLASSES, p)},
                    }
                )
    pd.DataFrame(prediction_rows).to_csv(directory / "predictions.csv", index=False)
    write_record(directory / "results.json", report)
    files = [
        p for p in directory.rglob("*") if p.is_file() and p.name != "complete.json"
    ]
    write_record(
        directory / "complete.json",
        {str(p.relative_to(directory)): checksum(p) for p in files},
    )
    return report


def predict_research(path: Path, sequences: list[str]) -> np.ndarray:
    """Load a trusted research artifact only; never accept untrusted pickle files."""
    artifact = joblib.load(path)
    if artifact.get("kind") != "mapexploc-research-v1" or artifact.get(
        "classes"
    ) != list(CLASSES):
        raise ValueError("Incompatible research artifact")
    x = research_features(sequences)
    if list(x.columns) != artifact["feature_names"]:
        raise ValueError("Research feature schema mismatch")
    return temperature_scale(
        artifact["model"].predict_proba(x), artifact["temperature"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("examples/experiments/research-revision/dataset.csv"),
    )
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--grouping", choices=("sequence", "study"), default="sequence")
    parser.add_argument("--cohort", choices=("all", "note_free"), default="all")
    parser.add_argument(
        "--single-repeat",
        action="store_true",
        help="Prespecified sensitivity analyses only",
    )
    args = parser.parse_args()
    result = run_research(
        args.source,
        args.directory,
        args.jobs,
        args.grouping,
        args.cohort,
        SEEDS[:1] if args.single_repeat else SEEDS,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "final_candidate": result["final_candidate"],
                "retain_calibration": result["retain_calibration"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    # Keep serialized transformer functions importable in another Python process.
    from mapexploc.research import main as entrypoint

    entrypoint()
