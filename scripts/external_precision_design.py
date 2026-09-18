"""Prospective Monte Carlo sensitivity scenarios, never actual external outcomes.

These stylized correlated errors are design diagnostics, not validated power
estimates. Recruitment-specific simulations must be locked before annotation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mapexploc.external_validation import CLASSES, paired_bootstrap, write_new


def simulate(
    size: int, cluster_size: int, repeats: int, rng: np.random.Generator
) -> dict:
    # Complete-frame proportions: no class caps. Scenarios are declared, not fitted.
    prevalence = np.array([0.20, 0.22, 0.12, 0.34, 0.12])
    scenarios = []
    for gain in (0.0, 0.03, 0.06):
        covered = positive = material = sufficient = 0
        widths = []
        # Symmetric false-label errors give an analytic population confusion matrix.
        population_scores = []
        for accuracy in (0.60 + gain, 0.60):
            matrix = prevalence[:, None] * np.full((5, 5), (1 - accuracy) / 4)
            np.fill_diagonal(matrix, prevalence * accuracy)
            population_scores.append(
                float(np.mean(2 * matrix.diagonal() / (matrix.sum(0) + matrix.sum(1))))
            )
        true_delta = population_scores[0] - population_scores[1]
        for simulation in range(repeats):
            n_groups = size // cluster_size
            group_class = rng.choice(5, n_groups, p=prevalence)
            labels = np.repeat(group_class, cluster_size)
            groups = np.repeat(np.arange(n_groups), cluster_size)
            # A shared group draw induces strong within-cluster dependence;
            # shared pair draws make errors correlated across the two models.
            correlated = rng.random(len(labels)) < 0.5
            uniform = np.where(
                correlated,
                np.repeat(rng.random(n_groups), cluster_size),
                rng.random(len(labels)),
            )
            wrong = (labels + rng.integers(1, 5, len(labels))) % 5
            a = np.where(uniform < 0.60 + gain, labels, wrong)
            b = np.where(uniform < 0.60, labels, wrong)
            # Some discordant errors: improvements are not purely nested.
            independent = rng.random(len(labels)) < 0.25
            b = np.where(
                independent,
                np.where(
                    rng.random(len(labels)) < 0.60,
                    labels,
                    (labels + rng.integers(1, 5, len(labels))) % 5,
                ),
                b,
            )
            result = paired_bootstrap(
                np.asarray(CLASSES)[labels],
                np.eye(5)[a],
                np.eye(5)[b],
                groups,
                repetitions=400,
                seed=20260919 + simulation,
            )
            low, high = result["difference_interval"]
            widths.append(high - low)
            covered += low <= true_delta <= high
            positive += low > 0
            material += low > 0.03
            counts = np.bincount(labels, minlength=5)
            sufficient += bool((counts >= 100).all())
        scenarios.append(
            {
                "assumed_accuracy_gain": gain,
                "population_macro_f1_difference": true_delta,
                "empirical_interval_coverage": covered / repeats,
                "positive_gain_detection_fraction": positive / repeats,
                "material_gain_detection_fraction": material / repeats,
                "median_interval_width": float(np.median(widths)),
                "all_classes_100_fraction": sufficient / repeats,
            }
        )
    return {
        "proteins": size,
        "cluster_size": cluster_size,
        "simulations_per_scenario": repeats,
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    rng = np.random.default_rng(20260919)
    result = {
        "status": "synthetic_design_only_no_external_predictions",
        "seed": 20260919,
        "bootstrap_draws_for_design_only": 400,
        "assumptions": (
            "Five-class symmetric errors; fixed nonuniform prevalence; "
            "50% shared cluster draws; 25% independent RF correctness; "
            "groups homogeneous in class"
        ),
        "limitations": (
            "Simplified cluster/error structures, finite Monte Carlo uncertainty; "
            "not a sample-size guarantee and not evidence about model performance"
        ),
        "designs": [
            simulate(n, k, args.repeats, rng)
            for n, k in ((500, 1), (1000, 5), (2000, 5))
        ],
    }
    write_new(args.output, result)
    print(f"Wrote synthetic design sensitivity: {args.output}")


if __name__ == "__main__":
    main()
