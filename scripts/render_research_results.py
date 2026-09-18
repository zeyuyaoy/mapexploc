"""Render figures and tables directly from frozen scientific outputs."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports: dict[str, dict[str, Any]] = {
        name: load(args.results / name / "results.json")
        for name in ("primary", "study-groups", "note-free")
    }
    final = load(args.results / "final/finalization.json")
    diagnostics = load(args.results / "diagnostics/diagnostics.json")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for run, report in reports.items():
        for name, metrics in report["metrics"].items():
            rows.append(
                {
                    "analysis": run,
                    "method": name,
                    **{
                        k: metrics[k]
                        for k in (
                            "macro_f1",
                            "weighted_f1",
                            "balanced_accuracy",
                            "log_loss",
                            "multiclass_brier",
                            "macro_auroc",
                            "macro_average_precision",
                            "top_label_ece_10_bins",
                        )
                    },
                }
            )
    pd.DataFrame(rows).to_csv(args.output / "comparison.csv", index=False)
    protocol = load(args.results / "primary/protocol.json")
    source = Path(protocol["source"])
    if not source.exists():
        source = args.results / "dataset.csv"
    from mapexploc.research import load_development

    frame, _ = load_development(source)
    predictions = np.load(args.results / "final/calibration-predictions.npz")
    probabilities = predictions["raw"].reshape(-1, len(frame), 5)
    labels = np.array(final["raw"]["classes"])
    guesses = labels[probabilities.argmax(axis=2)]
    errors = frame[["accession", "label", "group", "has_location_note"]].copy()
    errors["incorrect_repeats"] = (guesses != frame.label.to_numpy()).sum(axis=0)
    errors["mean_confidence"] = probabilities.max(axis=2).mean(axis=0)
    errors["predicted_labels"] = [";".join(values) for values in guesses.T]
    errors.sort_values(
        ["incorrect_repeats", "mean_confidence"], ascending=False
    ).to_csv(args.output / "final-error-analysis.csv", index=False)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), layout="constrained")
    ax = axes[0, 0]
    methods = [
        ("dummy", "Prior dummy"),
        ("length", "Length only"),
        ("lr_global_0.1", "Global LR, C=0.1"),
        ("lr_full_0.1", "423-feature LR, C=0.1"),
        ("lr_terminal_0.1", "Terminal LR, C=0.1 (final)"),
        ("rf_global", "Global RF"),
        ("rf_terminal", "Terminal RF"),
        ("rf_expanded", "Expanded 512-tree RF"),
        ("reference", "Original RF refitted"),
        ("selected", "Nested selection procedure"),
    ]
    values = [reports["primary"]["metrics"][m]["macro_f1"] for m, _ in methods]
    ax.barh(
        np.arange(len(methods)),
        values,
        color=[
            "#177d8d" if m in ("lr_terminal_0.1", "selected") else "#9aabb6"
            for m, _ in methods
        ],
    )
    ax.set_yticks(np.arange(len(methods)), [label for _, label in methods], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.72)
    for i, value in enumerate(values):
        ax.text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Pooled repeated out-of-fold macro-F1")
    ax.set_title("A  Same proteins and outer folds", loc="left", fontweight="bold")
    ax = axes[0, 1]
    ax.plot([0, 1], [0, 1], "--", color="#aab1b5", lw=1)
    for name, color, label in (
        ("raw", "#177d8d", "Raw (retained)"),
        ("calibrated", "#b77828", "Temperature (rejected)"),
    ):
        bins = [b for b in final[name]["reliability_bins"] if b["count"]]
        ax.plot(
            [b["confidence"] for b in bins],
            [b["accuracy"] for b in bins],
            "o-",
            color=color,
            label=label,
            markersize=4,
        )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean maximum class probability",
        ylabel="Observed correctness",
    )
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(
        "B  Final logistic model: exploratory calibration",
        loc="left",
        fontweight="bold",
    )
    ax = axes[1, 0]
    for i, (name, label) in enumerate(
        (
            ("primary", "Sequence groups, 2 repeats"),
            ("study-groups", "Sequence + publication groups"),
            ("note-free", "No localization notes"),
        )
    ):
        r = reports[name]
        gain = (
            r["metrics"]["selected"]["macro_f1"] - r["metrics"]["reference"]["macro_f1"]
        )
        lower, upper = r["bootstrap"]["selected"]["macro_f1_difference_97_5_interval"]
        ax.errorbar(
            gain,
            i,
            xerr=[[gain - lower], [upper - gain]],
            fmt="o",
            color="#177d8d",
            capsize=4,
        )
        ax.text(upper + 0.005, i, f"{gain:+.3f}", va="center", fontsize=8)
    ax.set_yticks(
        range(3),
        ["Sequence groups, 2 repeats", "Sequence + publications", "Note-free cohort"],
    )
    ax.axvline(0, color="#aab1b5", linestyle="--", lw=1)
    ax.set_xlim(-0.01, 0.16)
    ax.invert_yaxis()
    ax.set_xlabel("Macro-F1 change vs refitted original RF; paired 97.5% interval")
    ax.set_title(
        "C  Gain survives declared sensitivity analyses", loc="left", fontweight="bold"
    )
    ax = axes[1, 1]
    names = [
        "global_composition_physics",
        "n_terminal",
        "length",
        "c_terminal",
        "dipeptides",
    ]
    values = [
        diagnostics["permutation_importance"][n]["log_loss_increase_mean"]
        for n in names
    ]
    ax.barh(range(len(names)), values, color="#177d8d")
    ax.set_yticks(
        range(len(names)),
        [
            "Global composition / physics",
            "N-terminal composition",
            "Length",
            "C-terminal composition",
            "Dipeptides (unused)",
        ],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Mean outer-fold log-loss increase after block permutation")
    ax.set_title(
        "D  Predictive reliance, not causal mechanism", loc="left", fontweight="bold"
    )
    fig.suptitle(
        "MAP-ExPLoc: internal development evidence\n"
        "No independent confirmation; "
        "final candidate is uncalibrated logistic regression",
        fontsize=13,
    )
    fig.savefig(args.output / "research-revision.png", dpi=190)
    fig.savefig(args.output / "research-revision.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
