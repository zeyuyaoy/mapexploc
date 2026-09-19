"""Sampling diagnostics and held-out perturbation effects on model outputs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .adapter import BaseModelAdapter, predict_probabilities
from .report_v2 import FeatureDefinition


def attribution_agreement(
    left: Any,
    right: Any,
    classes: Sequence[str],
    comparison: str = "repeat",
    methodological: bool = False,
) -> list[dict[str, Any]]:
    a, b = np.asarray(left, float), np.asarray(right, float)
    if (
        a.shape != b.shape
        or a.ndim != 2
        or len(classes) != a.shape[0]
        or not np.isfinite([a, b]).all()
    ):
        raise ValueError("Agreement requires finite aligned class-by-feature tensors")
    rows: list[dict[str, Any]] = []
    for label, x, y in zip(classes, a, b):
        delta = float(np.max(np.abs(x - y)))
        informative = bool(max(np.max(np.abs(x)), np.max(np.abs(y))) > 0.005)
        rho = (
            float(spearmanr(np.abs(x), np.abs(y)).statistic)
            if np.ptp(np.abs(x)) > 1e-12 and np.ptp(np.abs(y)) > 1e-12
            else None
        )
        significant = (np.abs(x) > 0.005) | (np.abs(y) > 0.005)
        signs = (
            float(np.mean(np.sign(x[significant]) == np.sign(y[significant])))
            if significant.any()
            else None
        )
        k = max(1, math.ceil(len(x) * 0.2))
        # Break ties by feature order; near-zero explanations are uninformative.
        ix = set(np.argsort(-np.abs(x), kind="stable")[:k])
        iy = set(np.argsort(-np.abs(y), kind="stable")[:k])
        overlap = len(ix & iy) / len(ix | iy) if informative else None
        unstable = (
            delta > 0.02
            or (rho is not None and rho < 0.8)
            or (signs is not None and signs < 0.9)
            or (overlap is not None and overlap < 0.6)
        )
        status = (
            "uninformative"
            if not informative or rho is None
            else (
                "methodological_sensitivity"
                if methodological
                else "unstable" if unstable else "stable"
            )
        )
        rows.append(
            dict(
                comparison=comparison,
                class_id=label,
                status=status,
                max_absolute_delta=delta,
                magnitude_spearman=rho,
                sign_agreement=signs,
                top_region_overlap=overlap,
            )
        )
    return rows


@dataclass(frozen=True)
class _Interval:
    start: int
    end: int
    category: str | None


def faithfulness(
    adapter: BaseModelAdapter,
    sequence: str,
    features: list[FeatureDefinition],
    values: Any,
    seed: int,
    draws: int = 4,
) -> list[dict[str, Any]]:
    """Independent interventions; rankings use SHAP, never annotations or labels.

    Each target uses the same length/position-matched control and random draws.
    Empty control strata are excluded, never silently replaced by unmatched ones.
    """
    regions: list[_Interval] = []
    for region in features:
        if region.start is None or region.end is None:
            raise ValueError("Faithfulness requires sequence regions")
        regions.append(_Interval(region.start, region.end, region.category))
    rng = np.random.default_rng(seed)
    native = predict_probabilities(adapter, [sequence])[0]
    rows: list[dict[str, Any]] = []
    for ci, label in enumerate(adapter.descriptor.classes):
        phi = np.asarray(values[ci])
        for direction in (1, -1):
            order = np.argsort(-direction * phi, kind="stable")
            targets = [int(i) for i in order if direction * phi[i] > 0.005][:2]
            for size in (1, 2):
                if len(targets) < size:
                    continue
                selected = targets[:size]
                controls = []
                for target in selected:
                    f = regions[target]
                    candidates = [
                        i
                        for i, g in enumerate(regions)
                        if i not in selected
                        and i not in controls
                        and g.end - g.start == f.end - f.start
                        and g.category == f.category
                    ]
                    if not candidates:
                        break
                    controls.append(min(candidates, key=lambda i: (abs(phi[i]), i)))
                if len(controls) != size:
                    rows.append(
                        dict(
                            class_id=label,
                            direction=direction,
                            region_count=size,
                            status="no_matched_controls",
                            target_indices=selected,
                        )
                    )
                    continue
                for intervention in (
                    "heldout_region_shuffle",
                    "heldout_position_substitution",
                ):
                    target_effects = []
                    control_effects = []
                    random_effects = []
                    restoration = []
                    for _ in range(draws):
                        random_indices = []
                        for target in selected:
                            f = regions[target]
                            eligible = [
                                i
                                for i, g in enumerate(regions)
                                if i not in selected
                                and i not in random_indices
                                and g.end - g.start == f.end - f.start
                                and g.category == f.category
                            ]
                            random_indices.append(int(rng.choice(eligible)))

                        def perturb(indices: list[int]) -> str:
                            residues = list(sequence)
                            for index in indices:
                                f = regions[index]
                                original = np.array(list(sequence[f.start : f.end]))
                                if intervention == "heldout_region_shuffle":
                                    replacement = rng.permutation(original)
                                else:
                                    # Independent draws from matching positional
                                    # strata; fitted references are not reused.
                                    pool = list(
                                        "".join(
                                            sequence[g.start : g.end]
                                            for g in regions
                                            if g.category == f.category
                                        )
                                    )
                                    replacement = rng.choice(pool, size=len(original))
                                residues[f.start : f.end] = replacement
                            return "".join(residues)

                        altered_target = perturb(selected)
                        control = perturb(controls)
                        random = perturb(random_indices)
                        # Restore selected native intervals into an
                        # independently altered full sequence.
                        background = perturb(list(range(len(regions))))
                        restored = list(background)
                        for index in selected:
                            f = regions[index]
                            restored[f.start : f.end] = sequence[f.start : f.end]
                        p = predict_probabilities(
                            adapter,
                            [
                                altered_target,
                                control,
                                random,
                                background,
                                "".join(restored),
                            ],
                        )[:, ci]
                        target_effects.append(float(direction * (native[ci] - p[0])))
                        control_effects.append(float(direction * (native[ci] - p[1])))
                        random_effects.append(float(direction * (native[ci] - p[2])))
                        restoration.append(float(direction * (p[4] - p[3])))
                    rows.append(
                        dict(
                            class_id=label,
                            direction=direction,
                            region_count=size,
                            status="assessed",
                            intervention=intervention,
                            target_indices=selected,
                            low_attribution_indices=controls,
                            fraction_residues_perturbed=sum(
                                regions[i].end - regions[i].start for i in selected
                            )
                            / len(sequence),
                            target_effects=target_effects,
                            low_attribution_effects=control_effects,
                            random_effects=random_effects,
                            restoration_effects=restoration,
                            matched_advantage=float(
                                np.mean(target_effects) - np.mean(control_effects)
                            ),
                            draws=draws,
                            seed=seed,
                            interpretation=(
                                "Directional effects on this model under specified"
                                " held-out interventions; not causal biological effects"
                            ),
                        )
                    )
    return rows
