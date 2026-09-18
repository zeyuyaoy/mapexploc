"""Reproducible, human-only research baseline; never run downloads in normal CI."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
import urllib.request
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .artifacts import load_model_artifact, save_model_artifact
from .explainers.shap import ShapExplainer
from .features import build_feature_matrix, normalize_protein_sequence

SOURCE_URL = (
    "https://rest.uniprot.org/uniprotkb/stream?"
    "query=%28reviewed%3Atrue%29%20AND%20%28organism_id%3A9606%29"
    "&format=json&fields=accession,id,sequence,cc_subcellular_location,"
    "organism_id,reviewed"
)
CLASSES = ("Cytoplasm", "Membrane", "Mitochondrion", "Nucleus", "Secreted")
# Deliberately explicit: generic membranes and organelle membranes are not plasma
# membranes. Unmapped compartments are excluded, not silently collapsed.
LOCATION_MAP = {
    "Cytoplasm": "Cytoplasm",
    "Cytosol": "Cytoplasm",
    "Nucleus": "Nucleus",
    "Mitochondrion": "Mitochondrion",
    "Secreted": "Secreted",
    "Cell membrane": "Membrane",
    "Plasma membrane": "Membrane",
}
LIMITATIONS = (
    "Human canonical proteins with one experimentally supported compartment only. "
    "Multilocalized, isoform-specific, unsupported and ambiguous annotations were "
    "excluded. Probabilities are uncalibrated; class-balanced curation does not "
    "represent natural prevalence. Sequence grouping at 30% identity and 80% "
    "bidirectional coverage does not exclude all remote or domain-level homology. "
    "SHAP describes engineered features, not biological causality."
)


def checksum(path: Path) -> str:
    """Hash the exact bytes used by a run."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """Write portable, finite JSON with stable ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def download_snapshot(directory: Path) -> None:
    """Explicitly retrieve public reviewed-human annotations, never user input."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "source.json"
    if target.exists():
        raise ValueError("Snapshot already exists; use a new directory to refresh")
    temporary = directory / "source.json.part"
    try:
        with urllib.request.urlopen(SOURCE_URL, timeout=240) as response:
            headers = str(response.headers)
            with temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        json.loads(temporary.read_text())
        temporary.replace(target)
        (directory / "source.headers").write_text(headers)
    finally:
        temporary.unlink(missing_ok=True)


def curate_entry(entry: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Accept an unambiguous canonical assignment with direct experimental evidence."""
    if entry.get("entryType") != "UniProtKB reviewed (Swiss-Prot)":
        return None, "not_reviewed"
    if entry.get("organism", {}).get("taxonId") != 9606:
        return None, "not_human"
    comments = [
        c
        for c in entry.get("comments", [])
        if c.get("commentType") == "SUBCELLULAR LOCATION"
    ]
    if not comments:
        return None, "no_location"
    labels: set[str] = set()
    evidence: list[dict[str, Any]] = []
    for comment in comments:
        if comment.get("molecule"):
            return None, "isoform_or_product_specific"
        for annotation in comment.get("subcellularLocations", []):
            location = annotation.get("location", {})
            name = location.get("value", "")
            label = LOCATION_MAP.get(name)
            if label is None:
                return None, "unmapped_location"
            labels.add(label)
            evidence.extend(
                e
                for e in location.get("evidences", [])
                if e.get("evidenceCode") == "ECO:0000269"
            )
    if len(labels) != 1:
        return None, "conflicting_locations"
    if not evidence:
        return None, "no_experimental_location_evidence"
    try:
        sequence = normalize_protein_sequence(
            entry.get("sequence", {}).get("value", "")
        )
    except (ValueError, TypeError):
        return None, "invalid_sequence"
    accession = entry.get("primaryAccession", "")
    if not accession or "-" in accession:
        return None, "noncanonical_accession"
    return {
        "accession": accession,
        "entry_name": entry.get("uniProtkbId", accession),
        "sequence": sequence,
        "label": next(iter(labels)),
        "evidence": json.dumps(evidence, sort_keys=True),
    }, "accepted"


def curate_snapshot(
        source: Path, cap: int = 500, seed: int = 42
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Curate, remove duplicates, and reproducibly cap each class."""
    payload = json.loads(source.read_text())
    reasons: Counter[str] = Counter()
    rows = []
    for entry in payload["results"]:
        row, reason = curate_entry(entry)
        reasons[reason] += 1
        if row is not None:
            rows.append(row)
    if not rows:
        raise ValueError(
            "No records satisfy the experimental, single-compartment policy"
        )
    frame = pd.DataFrame(rows).sort_values("accession")
    conflicting = frame.groupby("sequence")["label"].transform("nunique") > 1
    reasons["conflicting_duplicate_sequences"] = int(conflicting.sum())
    frame = frame.loc[~conflicting]
    before = len(frame)
    frame = frame.drop_duplicates("sequence")
    reasons["duplicate_sequences"] = before - len(frame)
    parts = [
        group.sample(n=min(cap, len(group)), random_state=seed)
        for _, group in frame.groupby("label", sort=True)
    ]
    selected = pd.concat(parts).sort_values("accession").reset_index(drop=True)
    reasons["over_class_cap"] = len(frame) - len(selected)
    return selected, dict(reasons)


def similarity_groups(accessions: list[str], pairs: list[tuple[str, str]]) -> list[str]:
    """Connected components keep every detected related pair in the same partition."""
    parents = {accession: accession for accession in accessions}

    def root(accession: str) -> str:
        while parents[accession] != accession:
            parents[accession] = parents[parents[accession]]
            accession = parents[accession]
        return accession

    for first, second in pairs:
        a, b = root(first), root(second)
        parents[max(a, b)] = min(a, b)
    return [root(accession) for accession in accessions]


def grouped_split(frame: pd.DataFrame, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Choose a reproducible approximately 80/20 split without looking at scores."""
    if set(frame["label"]) != set(CLASSES):
        raise ValueError("Curation must retain all five target classes")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    candidates = []
    for train, test in splitter.split(frame, frame["label"], frame["group"]):
        counts = frame.iloc[test]["label"].value_counts()
        if set(frame.iloc[train]["label"]) != set(CLASSES) or any(
                counts.get(c, 0) < 10 for c in CLASSES
        ):
            continue
        if set(frame.iloc[train]["group"]) & set(frame.iloc[test]["group"]):
            raise ValueError("Similarity group leakage in split")
        candidates.append((abs(len(test) / len(frame) - 0.2), train, test))
    if not candidates:
        raise ValueError(
            "Cannot reserve ten test proteins per class with disjoint similarity groups"
        )
    _, train, test = min(candidates, key=lambda item: item[0])
    return train, test


def write_fasta(frame: pd.DataFrame, path: Path) -> None:
    path.write_text(
        "".join(f">{row.accession}\n{row.sequence}\n" for row in frame.itertuples())
    )


def search_similar(
        query: Path, target: Path, output: Path, threads: int
) -> list[tuple[str, str]]:
    """Run the same exhaustive-output search policy for grouping and split audits."""
    command = [
        "mmseqs",
        "easy-search",
        str(query),
        str(target),
        str(output),
        str(output.with_suffix(".tmp")),
        "--min-seq-id",
        "0.3",
        "-c",
        "0.8",
        "--cov-mode",
        "0",
        "--alignment-mode",
        "3",
        "-s",
        "7.5",
        "-e",
        "10",
        "--max-seqs",
        "10000",
        "--threads",
        str(threads),
        "--format-output",
        "query,target,fident,qcov,tcov",
        "-v",
        "1",
    ]
    subprocess.run(command, check=True)
    return [
        (parts[0], parts[1])
        for line in output.read_text().splitlines()
        if len(parts := line.split("\t")) >= 2
    ]


def prepare_baseline(
        directory: Path, cap: int = 500, seed: int = 42, threads: int = 4
) -> dict[str, Any]:
    """Prepare a frozen dataset, group partition and bidirectional leakage audit."""
    if shutil.which("mmseqs") is None:
        raise ValueError(
            "MMseqs2 is required for preparation; install it before continuing"
        )
    if (directory / "manifest.json").exists():
        raise ValueError("Prepared snapshot already exists; use a new directory")
    source = directory / "source.json"
    frame, exclusions = curate_snapshot(source, cap, seed)
    fasta = directory / "selected.fasta"
    write_fasta(frame, fasta)
    pairs = search_similar(fasta, fasta, directory / "all-pairs.tsv", threads)
    frame["group"] = similarity_groups(frame["accession"].tolist(), pairs)
    train, test = grouped_split(frame, seed)
    frame["split"] = "train"
    frame.loc[test, "split"] = "test"
    for name in ("train", "test"):
        write_fasta(frame.loc[frame["split"] == name], directory / f"{name}.fasta")
    audit_pairs = []
    for first, second in (("train", "test"), ("test", "train")):
        audit_pairs.extend(
            search_similar(
                directory / f"{first}.fasta",
                directory / f"{second}.fasta",
                directory / f"audit-{first}-{second}.tsv",
                threads,
            )
        )
    if audit_pairs:
        raise ValueError(
            "Cross-partition similarity detected: refusing to train a leaked benchmark"
        )
    frame.to_csv(directory / "dataset.csv", index=False)
    headers = (directory / "source.headers").read_text()
    release = next(
        (
            line.split(":", 1)[1].strip()
            for line in headers.splitlines()
            if line.lower().startswith("x-uniprot-release:")
        ),
        None,
    )
    if not release:
        raise ValueError(
            "UniProt release header is missing; source provenance is incomplete"
        )
    manifest = {
        "name": "MAP-ExPLoc human baseline",
        "scope": "Reviewed human canonical proteins; single-compartment classification",
        "source": SOURCE_URL,
        "release": release,
        "retrieved_at": datetime.fromtimestamp(source.stat().st_mtime, UTC).isoformat(),
        "source_sha256": checksum(source),
        "dataset_sha256": checksum(directory / "dataset.csv"),
        "seed": seed,
        "class_cap": cap,
        "class_counts": {
            str(k): int(v) for k, v in frame["label"].value_counts().items()
        },
        "exclusions": exclusions,
        "mmseqs_version": subprocess.check_output(
            ["mmseqs", "version"], text=True
        ).strip(),
        "split": {
            "train_count": len(train),
            "test_count": len(test),
            "identity": 0.3,
            "bidirectional_coverage": 0.8,
            "cross_partition_hits": 0,
            "groups": int(frame["group"].nunique()),
            "test_refitted": False,
        },
        "attribution": "UniProt Consortium, UniProtKB/Swiss-Prot, CC BY 4.0 (https://www.uniprot.org/help/license)",
        "limitations": LIMITATIONS,
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def classification_metrics(
        truth: Any, predicted: Any, probabilities: np.ndarray, classes: list[str]
) -> dict[str, Any]:
    """Multiclass discrimination and uncalibrated probability quality."""
    truth_array = np.asarray(truth)
    confidence = probabilities.max(axis=1)
    correct = np.asarray(predicted) == truth_array
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = sum(
        float((bins == i).mean())
        * abs(float(correct[bins == i].mean()) - float(confidence[bins == i].mean()))
        for i in range(10)
        if (bins == i).any()
    )
    one_hot = (truth_array[:, None] == np.asarray(classes)[None, :]).astype(float)
    return {
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(truth, predicted, average="weighted", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "classification_report": classification_report(
            truth, predicted, labels=classes, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=classes).tolist(),
        "classes": classes,
        "log_loss": float(log_loss(truth, probabilities, labels=classes)),
        "multiclass_brier": float(
            np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
        ),
        "top_label_ece_10_bins": ece,
    }


def train_baseline(
        directory: Path, output_model: Path, jobs: int = 4
) -> dict[str, Any]:
    """Select on grouped training CV; evaluate once without fitting on held-out rows."""
    if output_model.exists():
        raise ValueError("Output artifact exists; choose a new path to preserve it")
    manifest = json.loads((directory / "manifest.json").read_text())
    if checksum(directory / "dataset.csv") != manifest["dataset_sha256"]:
        raise ValueError("Dataset checksum differs from the prepared snapshot")
    if manifest["split"]["cross_partition_hits"] != 0:
        raise ValueError("The prepared dataset did not pass the similarity audit")
    frame = pd.read_csv(directory / "dataset.csv")
    train = frame.loc[frame["split"] == "train"]
    test = frame.loc[frame["split"] == "test"]
    if set(train["group"]) & set(test["group"]) or set(train["sequence"]) & set(
            test["sequence"]
    ):
        raise ValueError("Training and test data overlap")
    if any((test["label"] == label).sum() < 10 for label in CLASSES):
        raise ValueError("The test partition needs at least ten proteins per class")
    started = time.perf_counter()
    features = build_feature_matrix(train["sequence"])
    test_features = build_feature_matrix(test["sequence"])
    seed = int(manifest["seed"])
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    splits = list(cv.split(features, train["label"], train["group"]))
    for fit, validation in splits:
        if set(train.iloc[fit]["label"]) != set(CLASSES) or set(
                train.iloc[validation]["label"]
        ) != set(CLASSES):
            raise ValueError("All five classes must occur in every grouped CV fold")
        if set(train.iloc[fit]["group"]) & set(train.iloc[validation]["group"]):
            raise ValueError("Similarity group leakage in cross-validation")
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    class_weight="balanced", random_state=seed, n_jobs=1
                ),
            ),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {
            "rf__n_estimators": [128, 256],
            "rf__max_depth": [12, 24],
            "rf__min_samples_leaf": [1, 3],
        },
        scoring="f1_macro",
        cv=splits,
        n_jobs=jobs,
        error_score="raise",
    )
    search.fit(features, train["label"])
    model = search.best_estimator_
    training_seconds = time.perf_counter() - started
    classes = [str(label) for label in model.classes_]
    probabilities = model.predict_proba(test_features)
    predicted = model.predict(test_features)
    evaluation = classification_metrics(
        test["label"], predicted, probabilities, classes
    )
    dummy = DummyClassifier(strategy="prior").fit(features, train["label"])
    evaluation["dummy"] = classification_metrics(
        test["label"],
        dummy.predict(test_features),
        dummy.predict_proba(test_features),
        classes,
    )
    evaluation["exceeds_dummy_macro_f1"] = (
            evaluation["macro_f1"] > evaluation["dummy"]["macro_f1"]
    )
    cv_report = [
        {"params": params, "mean_macro_f1": float(score), "std_macro_f1": float(std)}
        for params, score, std in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"],
        )
    ]
    explainer = ShapExplainer(model)
    sample = test_features.iloc[:1]
    model.predict_proba(sample)
    explainer.explain_predictions(sample)
    timings: dict[str, float] = {}
    operations: dict[str, Callable[[], Any]] = {
        "prediction_one_ms": lambda: model.predict_proba(sample),
        "prediction_batch_100_ms": lambda: model.predict_proba(
            test_features.iloc[:100]
        ),
        "shap_one_ms": lambda: explainer.explain_predictions(sample),
        "feature_one_ms": lambda: build_feature_matrix([str(test.iloc[0]["sequence"])]),
    }
    for name, operation in operations.items():
        measured = []
        for _ in range(5):
            start = time.perf_counter()
            operation()
            measured.append((time.perf_counter() - start) * 1000)
        timings[name] = float(np.median(measured))
    metadata = {
        **manifest,
        "sample_count": len(train),
        "evaluation": evaluation,
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "software_versions": {
            name: version(name)
            for name in ("scikit-learn", "numpy", "pandas", "shap", "biopython")
        },
        "runtime": {
            "training_seconds": training_seconds,
            **timings,
            "timing_repetitions": 5,
            "batch_size": min(100, len(test)),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "training_jobs": jobs,
        },
        "model_id": f"human-{manifest['release']}-{manifest['dataset_sha256'][:12]}",
    }
    save_model_artifact(model, output_model, metadata=metadata)
    loaded = load_model_artifact(output_model)
    np.testing.assert_allclose(loaded.model.predict_proba(test_features), probabilities)
    report = {
        **metadata,
        "artifact_bytes": output_model.stat().st_size,
        "artifact_sha256": checksum(output_model),
        "cv_results": cv_report,
    }
    write_json(output_model.with_suffix(".report.json"), report)
    predictions = test[["accession", "label", "group"]].copy()
    predictions["prediction"] = predicted
    for index, label in enumerate(classes):
        predictions[f"probability_{label}"] = probabilities[:, index]
    predictions.to_csv(output_model.with_suffix(".predictions.csv"), index=False)
    return report
