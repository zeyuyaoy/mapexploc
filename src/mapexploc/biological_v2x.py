"""Versioned multi-family concordance tests with explicit null degeneracy gates."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .report_v2 import AnalysisReport, LocalExplanation
from .study import FAMILIES, required_groups


def interval_contrast(
    local: LocalExplanation, ci: int, intervals: list[tuple[int, int]]
) -> dict[str, Any] | None:
    signal = []
    control = []
    masks = []
    for i, (feature, value) in enumerate(zip(local.features, local.attributions[ci])):
        if feature.start is None or feature.end is None:
            raise ValueError(
                "Biological interval validation requires region attributions"
            )
        if not any(
            start <= feature.start and feature.end <= end for start, end in intervals
        ):
            continue
        candidates = [
            j
            for j, g in enumerate(local.features)
            if g.start is not None
            and g.end is not None
            and g.category == feature.category
            and g.end - g.start == feature.end - feature.start
            and not any(start < g.end and g.start < end for start, end in intervals)
        ]
        if not candidates:
            continue
        length = feature.end - feature.start
        signal.append(max(value, 0) / length)
        control.append(
            float(
                np.mean(
                    [max(local.attributions[ci][j], 0) / length for j in candidates]
                )
            )
        )
        masks.append([i, candidates])
    if not signal:
        return None
    return {
        "effect": float(np.mean(signal) - np.mean(control)),
        "signal_density": float(np.mean(signal)),
        "control_density": float(np.mean(control)),
        "eligible_masks": masks,
    }


def evaluate_family(
    report: AnalysisReport,
    manifest: dict[str, Any],
    family: str,
    seed: int = 20260919,
    randomizations: int = 1000,
) -> dict[str, Any]:
    if family not in FAMILIES:
        raise ValueError("Unknown preregistered determinant family")
    if randomizations < 1000:
        raise ValueError(
            "Formal biological evaluation requires at least 1000 randomizations"
        )
    members = {m["protein_id"]: m for m in manifest["members"]}
    if len(members) != len(manifest["members"]):
        raise ValueError("Duplicate identifiers in frozen evaluation manifest")
    if {r.protein.protein_id for r in report.results} != set(members):
        raise ValueError("Complete frozen evaluation membership is required")
    if any(m["split"] != "evaluation" for m in members.values()):
        raise ValueError("Untouched evaluation manifest required")
    ci = list(report.model.classes).index(FAMILIES[family])
    seen = set()
    strata: dict[tuple[Any, ...], list[LocalExplanation]] = defaultdict(list)
    annotations = {}
    exclusions = []
    for local in sorted(report.results, key=lambda r: r.protein.protein_id):
        member = members.get(local.protein.protein_id)
        if (
            not member
            or member["sequence_sha256"] != local.sequence_sha256
            or member["group"] != local.protein.group
        ):
            raise ValueError(
                "Explanation membership differs from frozen evaluation manifest"
            )
        if local.protein.group in seen:
            raise ValueError("Independent groups must have exactly one representative")
        seen.add(local.protein.group)
        intervals = [(f["start"], f["end"]) for f in member["families"].get(family, [])]
        annotations[local.protein.protein_id] = intervals
        contrast = interval_contrast(local, ci, intervals) if intervals else None
        if contrast is None:
            exclusions.append(
                {
                    "protein_id": local.protein.protein_id,
                    "reason": "insufficient_resolution_or_matched_non_signal_regions",
                }
            )
            continue
        positional = tuple(
            sorted(
                {
                    str(f.category)
                    for f in local.features
                    if f.start is not None
                    and f.end is not None
                    and any(a <= f.start and f.end <= b for a, b in intervals)
                }
            )
        )
        strata[(len(local.protein.sequence) // 50, positional)].append(local)
    if family in {"peroxisomal_pts1", "er_retention"}:
        return {
            "family": family,
            "status": "exploratory_fixed_terminal_controls_required",
            "formal_p": None,
            "exclusions": exclusions,
            "interpretation": (
                "Terminal overlap is not sequence specificity; matched sequence"
                " controls and independent motif/order interventions are required."
            ),
        }
    matched = []
    for group in strata.values():
        # Fix membership from coordinates before examining effects.
        if len(group) < 2 or any(
            interval_contrast(local, ci, annotations[donor.protein.protein_id]) is None
            for local in group
            for donor in group
        ):
            exclusions.extend(
                {
                    "protein_id": r.protein.protein_id,
                    "reason": (
                        "annotation_pairing_changes_control_support_or_singleton_stratum"
                    ),
                }
                for r in group
            )
        else:
            matched.append(group)
    eligible = [r for group in matched for r in group]
    if len(eligible) < 2:
        return {
            "family": family,
            "status": "insufficient_matched_groups",
            "formal_p": None,
            "included_groups": len(eligible),
            "exclusions": exclusions,
        }
    rows = [
        {
            "protein_id": r.protein.protein_id,
            "group": r.protein.group,
            **(interval_contrast(r, ci, annotations[r.protein.protein_id]) or {}),
        }
        for r in eligible
    ]
    effects = np.array([r["effect"] for r in rows])
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(effects, size=(1000, len(effects))).mean(axis=1)
    null = []
    mask_permutations = set()
    for _ in range(randomizations):
        samples = []
        signature = []
        for group in matched:
            for local, donor in zip(group, rng.permutation(len(group))):
                contrast = interval_contrast(
                    local, ci, annotations[group[int(donor)].protein.protein_id]
                )
                assert contrast is not None
                samples.append(contrast["effect"])
                signature.append(str(contrast["eligible_masks"]))
        null.append(float(np.mean(samples)))
        mask_permutations.add(tuple(signature))
    status = (
        "complete"
        if len(eligible) >= required_groups() and len(mask_permutations) >= 20
        else (
            "degenerate_matched_null"
            if len(mask_permutations) < 20
            else "exploratory_insufficient_power"
        )
    )
    p = (1 + sum(v >= effects.mean() for v in null)) / (1 + randomizations)
    return {
        "family": family,
        "class_id": FAMILIES[family],
        "model_id": report.model.model_id,
        "checkpoint_sha256": report.model.checkpoint_sha256,
        "status": status,
        "included_groups": len(eligible),
        "input_count": len(report.results),
        "coverage": len(eligible) / len(report.results),
        "effect": float(effects.mean()),
        "bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "null_mean": float(np.mean(null)),
        "effect_beyond_null": float(effects.mean() - np.mean(null)),
        "randomized_distribution": null,
        "distinct_coordinate_mask_permutations": len(mask_permutations),
        "unadjusted_p": p,
        "formal_p": p if status == "complete" else None,
        "seed": seed,
        "randomizations": randomizations,
        "proteins": rows,
        "exclusions": exclusions,
        "interpretation": (
            "Concordance with model behavior under the frozen reference game; not"
            " causal biology or independent predictive generalization. Apply"
            " family/mode Holm correction before claims."
        ),
    }
