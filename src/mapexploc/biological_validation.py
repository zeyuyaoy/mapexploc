"""Predeclared signal-concordance statistics; no causal interpretation is implied."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .report_v2 import AnalysisReport, LocalExplanation


def _signal_intervals(local: LocalExplanation) -> list[tuple[int, int]]:
    return [
        (a.start, a.end)
        for a in local.annotations
        if a.kind == "signal_peptide"
        and not a.uncertain
        and any(e.get("evidenceCode") == "ECO:0000269" for e in a.evidence)
    ]


def _contrast(
    local: LocalExplanation, class_index: int, signals: list[tuple[int, int]]
) -> dict[str, Any] | None:
    signal, control = [], []
    for feature, value in zip(local.features, local.attributions[class_index]):
        if feature.kind != "sequence_region" or feature.category != "N_terminal":
            continue
        assert feature.start is not None and feature.end is not None
        if feature.end - feature.start != 10:
            continue
        density = max(value, 0.0) / 10
        if any(start <= feature.start and feature.end <= end for start, end in signals):
            signal.append(density)
        elif not any(
            start < feature.end and feature.start < end for start, end in signals
        ):
            control.append(density)
    if not signal or not control:
        return None
    return {
        "signal_density": float(np.mean(signal)),
        "control_density": float(np.mean(control)),
        "effect": float(np.mean(signal) - np.mean(control)),
        "signal_windows": len(signal),
        "control_windows": len(control),
    }


def validate_signal_concordance(
    report: AnalysisReport,
    *,
    class_id: str = "Extracellular",
    seed: int = 1842,
    randomizations: int = 1000,
    bootstrap_replicates: int = 1000,
) -> dict[str, Any]:
    """Matched annotation-profile pairing, preserving length and terminal position.

    Inputs must have one protein per independent sequence group. The bootstrap
    samples these groups. Singleton strata are explicitly excluded from both the
    primary matched comparison and its null distribution. Null outcomes are valid.
    """
    report = AnalysisReport.model_validate(report.model_dump())
    if randomizations < 1 or bootstrap_replicates < 1:
        raise ValueError("Positive randomization and bootstrap counts are required")
    if class_id not in report.model.classes:
        raise ValueError("Requested class is not declared by the model")
    ci = report.model.classes.index(class_id)
    groups = [r.protein.group for r in report.results]
    if any(group is None for group in groups) or len(set(groups)) != len(groups):
        raise ValueError(
            "Biological validation requires one protein per declared independent group"
        )
    strata: dict[tuple[int, int], list[LocalExplanation]] = defaultdict(list)
    exclusions = []
    for local in sorted(report.results, key=lambda r: r.protein.protein_id):
        if local.explainer.get("method") != "region_kernel":
            raise ValueError(
                "Positional validation requires genuine region explanations"
            )
        intervals = _signal_intervals(local)
        if not intervals or _contrast(local, ci, intervals) is None:
            exclusions.append(
                {
                    "protein_id": local.protein.protein_id,
                    "reason": (
                        "No precise experimental signal with both complete signal "
                        "and matched nonsignal N-terminal windows"
                    ),
                }
            )
            continue
        # This protocol requires N-terminal signals.
        if any(start != 0 for start, _ in intervals):
            exclusions.append(
                {
                    "protein_id": local.protein.protein_id,
                    "reason": "Signal does not begin at the N terminus",
                }
            )
            continue
        strata[(len(local.protein.sequence) // 50, 0)].append(local)
    matched = {key: values for key, values in strata.items() if len(values) > 1}
    for values in strata.values():
        if len(values) == 1:
            exclusions.append(
                {
                    "protein_id": values[0].protein.protein_id,
                    "reason": "Singleton length/terminal-position stratum",
                }
            )
    eligible = [local for values in matched.values() for local in values]
    if len(eligible) < 2:
        raise ValueError(
            "At least two independent proteins in matched strata are required"
        )
    rows = [
        {
            "protein_id": local.protein.protein_id,
            "group": local.protein.group,
            "stratum": [len(local.protein.sequence) // 50, 0],
            **(_contrast(local, ci, _signal_intervals(local)) or {}),
        }
        for local in eligible
    ]
    effects = np.array([row["effect"] for row in rows])
    observed = float(effects.mean())
    rng = np.random.default_rng(seed)
    bootstrapped = np.mean(
        rng.choice(effects, size=(bootstrap_replicates, len(effects)), replace=True),
        axis=1,
    )
    null = []
    for _ in range(randomizations):
        differences = []
        for values in matched.values():
            permuted = rng.permutation(len(values))
            for local, donor in zip(values, permuted):
                contrast = _contrast(local, ci, _signal_intervals(values[int(donor)]))
                if contrast is None:
                    raise ValueError(
                        "Matched permutation changed eligibility; refine "
                        "prespecified strata before analysis"
                    )
                differences.append(contrast["effect"])
        null.append(float(np.mean(differences)))
    p = float((1 + sum(value >= observed for value in null)) / (randomizations + 1))
    return {
        "analysis": (
            "Signal-attribution concordance under a fixed shuffled-reference game"
        ),
        "model": report.model.model_id,
        "checkpoint_sha256": report.model.checkpoint_sha256,
        "class_id": class_id,
        "seed": seed,
        "randomizations": randomizations,
        "bootstrap_replicates": bootstrap_replicates,
        "input_count": len(report.results),
        "included_count": len(rows),
        "independent_group_count": len(rows),
        "exclusions": exclusions,
        "coverage_fraction": len(rows) / len(report.results),
        "proteins": rows,
        "statistic": (
            "Mean across independent groups of mean positive "
            "SHAP/residue in complete signal windows minus mean "
            "positive SHAP/residue in matched nonsignal N-terminal "
            "windows"
        ),
        "effect": observed,
        "group_bootstrap_95_interval": (
            np.quantile(bootstrapped, [0.025, 0.975]).tolist()
        ),
        "randomized_pairing_distribution": null,
        "null_mean": float(np.mean(null)),
        "effect_beyond_matched_null": observed - float(np.mean(null)),
        "one_sided_randomization_p": p,
        "conclusion": (
            "Concordance beyond the matched null"
            if observed > 0 and p < 0.05
            else "No support for concordance beyond the matched null"
        ),
        "limitations": [
            (
                "Model behavior, not biological causality or independent "
                "predictive generalization."
            ),
            (
                "Signal annotation and attribution pairing preserves coarse"
                " length and N-terminal bias; identical signal endpoints "
                "can yield a degenerate null."
            ),
            (
                "Known supervision overlap must be audited separately; "
                "foundation-model pretraining overlap may be unknown."
            ),
        ],
    }
