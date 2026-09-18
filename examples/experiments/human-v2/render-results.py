"""Render the frozen model comparison; run with the optional plotting extra."""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("model-comparison.png"))
args = parser.parse_args()
root = Path(__file__).parent
report = json.loads((root / "evaluation.json").read_text())
training = json.loads((root / "training.json").read_text())
classes = report["evaluation"]["classes"]
selected = [
    next(
        row
        for row in training["comparisons"]
        if row["candidate"]["family"] == family and not row["candidate"]["reference"]
    )
    for family in ("random_forest", "extra_trees")
]
selected.append(
    next(row for row in training["comparisons"] if row["candidate"]["reference"])
)
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    }
)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
fig.suptitle(
    "Human protein localization: CPU model comparison", fontsize=16, weight="bold"
)
colors = ["#3e7c89", "#b38045", "#89949b"]
x = np.arange(3)
values = [r["mean_macro_f1"] for r in selected]
axes[0].bar(x, values, color=colors, width=0.58)
axes[0].set_xticks(
    x, ["Best Random\nForest", "Best Extra\nTrees", "Original\nconfiguration"]
)
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Mean macro-F1")
axes[0].set_title(
    "Grouped development validation\nNine folds per configuration; selection scores",
    pad=15,
)
for i, value in enumerate(values):
    axes[0].text(i, value + 0.02, f"{value:.3f}", ha="center", fontsize=11)
x = np.arange(len(classes))
for offset, key, label, color in [
    (-0.18, "reference_evaluation", "Unchanged version 1", "#89949b"),
    (0.18, "evaluation", "Selected candidate", "#3e7c89"),
]:
    values = [report[key]["classification_report"][c]["f1-score"] for c in classes]
    axes[1].bar(x + offset, values, width=0.34, color=color, label=label)
    for i, value in enumerate(values):
        axes[1].text(i + offset, value + 0.015, f"{value:.2f}", ha="center", fontsize=8)
axes[1].set_xticks(
    x, ["Cytoplasm", "Membrane", "Mitochon-\ndrion", "Nucleus", "Secreted"]
)
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("Class F1")
axes[1].set_title(
    "Previously inspected historical benchmark\n"
    "364 proteins; independent confirmation unavailable",
    pad=15,
)
axes[1].legend(loc="upper left", frameon=False)
for ax in axes:
    ax.yaxis.grid(True, alpha=0.16)
    ax.set_axisbelow(True)
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=180)
plt.close(fig)
