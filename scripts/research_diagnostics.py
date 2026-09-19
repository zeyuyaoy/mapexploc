"""Reproduce selected outer fits; run falsification controls without model tuning."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from mapexploc.baseline import CLASSES, checksum
from mapexploc.experiments import atomic_json, read_json
from mapexploc.research import (
    GLOBAL,
    SEEDS,
    TERMINAL,
    feature_columns,
    fit_checked,
    load_development,
    make_model,
    read_record,
    research_features,
    score,
)
from mapexploc.validation import grouped_splits


def block_shuffle(y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    """Shuffle entire label vectors between groups of equal size, not proteins."""
    rng = np.random.default_rng(seed)
    result = y.copy()
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    for size in sorted({len(m) for m in members}):
        same_size = [m for m in members if len(m) == size]
        for destination, source in zip(same_size, rng.permutation(len(same_size))):
            result[destination] = y[same_size[source]]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = read_json(args.run / "protocol.json")
    source = Path(protocol["source"])
    if checksum(source) != protocol["source_sha256"]:
        raise ValueError("Input changed")
    frame, _ = load_development(source)
    x, y, groups = (
        research_features(frame.sequence),
        frame.label.to_numpy(),
        frame.group.to_numpy(),
    )
    design = read_json(args.run / "design.json")
    if design["accessions"] != frame.accession.tolist():
        raise ValueError("Accession order changed")
    records = [
        read_record(args.run / "folds" / f'{f["seed"]}-{f["fold"]}.json')
        for f in design["folds"]
    ]
    by_id = {c["id"]: c for c in protocol["candidates"]}
    blocks = {
        "length": ["length"],
        "global_composition_physics": GLOBAL[1:],
        "n_terminal": [n for n in TERMINAL if n.startswith("n")],
        "c_terminal": [n for n in TERMINAL if n.startswith("c")],
        "dipeptides": [n for n in x if n.startswith("dp_")],
    }
    perturbations, controls, reproduction = [], [], []
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        for fold, record in zip(design["folds"], records):
            train, valid = np.array(fold["train"]), np.array(fold["validation"])
            candidate = by_id[record["selected"]]
            model = fit_checked(
                make_model(candidate, fold["seed"]), x.iloc[train], y[train]
            )
            p = model.predict_proba(x.iloc[valid])
            maximum_difference = float(
                np.abs(p - np.array(record["probabilities"]["selected"])).max()
            )
            np.testing.assert_allclose(
                p, record["probabilities"]["selected"], atol=1e-12, rtol=0
            )
            reproduction.append(
                {
                    "seed": fold["seed"],
                    "fold": fold["fold"],
                    "maximum_probability_difference": maximum_difference,
                }
            )
            baseline = score(y[valid], p)
            for block, columns in blocks.items():
                for repeat in range(5):
                    rng = np.random.default_rng(SEEDS[0] + repeat)
                    altered = x.iloc[valid].copy()
                    altered.loc[:, columns] = altered[columns].to_numpy()[
                        rng.permutation(len(valid))
                    ]
                    metrics = score(y[valid], model.predict_proba(altered))
                    perturbations.append(
                        {
                            "seed": fold["seed"],
                            "fold": fold["fold"],
                            "block": block,
                            "permutation": repeat,
                            "macro_f1_decrease": baseline["macro_f1"]
                            - metrics["macro_f1"],
                            "log_loss_increase": metrics["log_loss"]
                            - baseline["log_loss"],
                        }
                    )
            # Five first-repeat controls; these do not yield a permutation p-value.
            if fold["seed"] == SEEDS[0]:
                for repeat in range(5):
                    shuffled = block_shuffle(y[train], groups[train], SEEDS[0] + repeat)
                    control = fit_checked(
                        make_model(candidate, fold["seed"]), x.iloc[train], shuffled
                    )
                    controls.append(
                        {
                            "fold": fold["fold"],
                            "shuffle_seed": SEEDS[0] + repeat,
                            "validation": valid.tolist(),
                            "probabilities": control.predict_proba(
                                x.iloc[valid]
                            ).tolist(),
                        }
                    )
    null_results = {}
    for seed in sorted({c["shuffle_seed"] for c in controls}):
        p = np.zeros((len(frame), 5))
        for control in controls:
            if control["shuffle_seed"] == seed:
                p[control["validation"]] = control["probabilities"]
        null_results[str(seed)] = score(y, p)
    importance = pd.DataFrame(perturbations)
    importance_summary = importance.groupby("block")[
        ["macro_f1_decrease", "log_loss_increase"]
    ].agg(["mean", "std"])
    final = joblib.load(args.run / "research-model.joblib")
    classifier = final["model"].named_steps["classifier"]
    interpretation = {"kind": "no linear coefficients"}
    if hasattr(classifier, "coef_"):
        columns = feature_columns(final["candidate"])
        interpretation = {
            "kind": "standardized conditional log-odds coefficients; not causal",
            "classes": list(CLASSES),
            "columns": columns,
            "coefficients": classifier.coef_.tolist(),
            "intercepts": classifier.intercept_.tolist(),
        }
    elif hasattr(classifier, "estimators_"):
        import shap

        transformed = final["model"][:-1].transform(x.iloc[:5])
        explanation = shap.TreeExplainer(classifier)(transformed)
        np.testing.assert_allclose(
            explanation.base_values + explanation.values.sum(axis=1),
            final["model"].predict_proba(x.iloc[:5]),
            atol=1e-7,
        )
        interpretation = {
            "kind": "TreeSHAP for raw probability, not temperature-scaled probability",
            "classes": list(CLASSES),
            "columns": feature_columns(final["candidate"]),
            "accessions": frame.accession.iloc[:5].tolist(),
            "values": explanation.values.tolist(),
            "base_values": explanation.base_values.tolist(),
            "additivity_verified": True,
            "caveat": (
                "Engineered feature contributions are not causal biological mechanisms"
            ),
        }
    saved = pd.read_csv(args.run / "predictions.csv")
    selected_rows = saved.loc[saved.method == "selected"].copy()
    pcols = [f"p_{c}" for c in CLASSES]
    selected_rows["predicted"] = np.array(CLASSES)[
        selected_rows[pcols].to_numpy().argmax(axis=1)
    ]
    selected_rows["confidence"] = selected_rows[pcols].max(axis=1)
    selected_rows["incorrect"] = selected_rows.label != selected_rows.predicted
    errors = (
        selected_rows.groupby("accession")
        .agg(
            label=("label", "first"),
            group=("group", "first"),
            incorrect_repeats=("incorrect", "sum"),
            mean_confidence=("confidence", "mean"),
            predicted_labels=("predicted", lambda values: ";".join(values)),
        )
        .reset_index()
        .merge(frame[["accession", "has_location_note"]], on="accession")
    )
    errors = errors.sort_values(
        ["incorrect_repeats", "mean_confidence"], ascending=False
    )
    # Keep repeated predictions together when estimating class intervals.
    gt = selected_rows.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    guessed = selected_rows.predicted.map(
        {c: i for i, c in enumerate(CLASSES)}
    ).to_numpy()
    _, group_indices = np.unique(selected_rows.group, return_inverse=True)
    matrices = np.zeros((group_indices.max() + 1, 5, 5))
    np.add.at(matrices, (group_indices, gt, guessed), 1)
    rng = np.random.default_rng(SEEDS[0])
    class_draws = []
    for _ in range(2000):
        matrix = matrices[rng.integers(0, len(matrices), len(matrices))].sum(axis=0)
        denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
        f1 = np.divide(
            2 * matrix.diagonal(), denominator, out=np.zeros(5), where=denominator != 0
        )
        support = matrix.sum(axis=1)
        recall = np.divide(
            matrix.diagonal(), support, out=np.zeros(5), where=support != 0
        )
        class_draws.append(np.stack([f1, recall]))
    class_intervals = np.quantile(class_draws, [0.025, 0.975], axis=0)
    timings = []
    for _ in range(20):
        begin = time.perf_counter()
        final["model"].predict_proba(x.iloc[:1])
        timings.append((time.perf_counter() - begin) * 1000)
    report = {
        "source_sha256": protocol["source_sha256"],
        "script_sha256": checksum(Path(__file__)),
        "protocol_sha256": checksum(args.run / "protocol.json"),
        "fit_reproduction": reproduction,
        "permutation_importance": {
            block: {
                f"{metric}_{stat}": float(value)
                for (metric, stat), value in values.items()
            }
            for block, values in importance_summary.iterrows()
        },
        "training_label_shuffle": null_results,
        "coefficient_interpretation": interpretation,
        "per_class_exploratory_intervals": {
            c: {
                "f1_95": class_intervals[:, 0, i].tolist(),
                "recall_95": class_intervals[:, 1, i].tolist(),
            }
            for i, c in enumerate(CLASSES)
        },
        "error_counts": {
            "wrong_both_repeats": int((errors.incorrect_repeats == 2).sum()),
            "wrong_at_least_once": int((errors.incorrect_repeats > 0).sum()),
            "wrong_both_mean_confidence_over_0_8": int(
                ((errors.incorrect_repeats == 2) & (errors.mean_confidence > 0.8)).sum()
            ),
        },
        "final_prediction_one_ms_median": float(np.median(timings)),
        "final_selection_folds": [
            {"train": t.tolist(), "validation": v.tolist()}
            for t, v in grouped_splits(y, groups, 3, SEEDS[0] + 100)
        ],
        "seconds": time.perf_counter() - started,
        "limitations": [
            "Postfit diagnostics, not used to choose the model or infer p-values",
            "Permutation breaks feature correlations; "
            "blocks are descriptive not mechanistic",
            "Repeated overlapping-fold standard deviations are descriptive only",
            "Label shuffle keeps group sizes/class counts but large groups "
            "may lack exchange partners",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "diagnostics.json", report)
    importance.to_csv(args.output / "permutation-blocks.csv", index=False)
    errors.to_csv(args.output / "error-analysis.csv", index=False)
    atomic_json(args.output / "null-predictions.json", controls)
    print(
        json.dumps(
            {
                "reproduced_fits": len(reproduction),
                "control_seeds": len(null_results),
                "seconds": report["seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
