"""KernelSHAP on binary region-presence masks and fixed shuffled references."""

from __future__ import annotations

import hashlib
import threading
from typing import Any

import numpy as np
import shap

from .shap import ShapExplainer
from ..adapter import BaseModelAdapter, predict_probabilities
from ..methods import MethodConfiguration
from ..provenance import sequence_sha256
from ..report_v2 import AnalysisConfiguration, FeatureDefinition

_KERNEL_LOCK = threading.Lock()  # SHAP samples coalitions with NumPy's global RNG.
REGION_POLICY = "terminal-50-by-10/interior-up-to-six/short-contiguous-10:v1"
REFERENCE_POLICY = "mean-probability-over-fixed-whole-sequence-shuffles:v1"


def sequence_regions(length: int, strategy: str = "legacy") -> list[FeatureDefinition]:
    if length < 1:
        raise ValueError("Cannot partition an empty protein")
    intervals: list[tuple[int, int, str]] = []
    if strategy in {"windows_5", "windows_10"}:
        from .references import position_stratum

        width = int(strategy.split("_")[1])
        intervals = [
            (start, min(start + width, length), position_stratum(start, length))
            for start in range(0, length, width)
        ]
    elif strategy != "legacy":
        raise ValueError("Unsupported region strategy")
    elif length < 100:
        for start in range(0, length, 10):
            end = min(start + 10, length)
            # Categorize short proteins separately.
            # Do not count terminal regions twice.
            intervals.append((start, end, "short_sequence"))
    else:
        intervals += [(start, start + 10, "N_terminal") for start in range(0, 50, 10)]
        interior = length - 100
        if interior:
            edges = np.linspace(50, length - 50, min(6, interior) + 1, dtype=int)
            intervals += [
                (int(a), int(b), "interior") for a, b in zip(edges, edges[1:])
            ]
        intervals += [
            (start, start + 10, "C_terminal")
            for start in range(length - 50, length, 10)
        ]
    return [
        FeatureDefinition(
            feature_id=f"region:{start}:{end}",
            kind="sequence_region",
            start=start,
            end=end,
            category=category,
            units="binary presence",
            definition=(
                "This complete sequence interval is native (1) or replaced "
                "from a fixed whole-protein shuffled reference (0)."
            ),
        )
        for start, end, category in intervals
    ]


def derived_seed(seed: int, sequence: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{sequence_sha256(sequence)}".encode()).digest()[:4],
        "big",
    )


def shuffled_references(sequence: str, seed: int, count: int) -> list[str]:
    rng = np.random.default_rng(derived_seed(seed, sequence))
    residues = np.array(list(sequence))
    return ["".join(rng.permutation(residues)) for _ in range(count)]


class RegionGame:
    """A coalition invokes the complete predictor, including fresh preprocessing."""

    def __init__(
        self,
        adapter: BaseModelAdapter,
        sequence: str,
        configuration: AnalysisConfiguration,
    ):
        self.adapter = adapter
        self.sequence = sequence
        self.configuration = configuration
        self.regions = sequence_regions(
            len(sequence), getattr(configuration, "region_strategy", "legacy")
        )
        if (
            isinstance(configuration, MethodConfiguration)
            and configuration.method_profile == "v2x-development"
        ):
            from .references import references

            self.references = references(sequence, configuration)
            self.regions = [
                f.model_copy(
                    update={
                        "definition": (
                            "This complete interval is native (1) or replaced from the"
                            " recorded fixed reference distribution (0)."
                        )
                    }
                )
                for f in self.regions
            ]
        else:
            self.references = shuffled_references(
                sequence, configuration.seed, configuration.references
            )
        self.batch_size = configuration.inference_batch_size
        self.cache: dict[tuple[int, ...], np.ndarray] = {}
        self.evaluated_sequences = 0

    def __call__(self, masks: np.ndarray) -> np.ndarray:
        masks = np.asarray(masks)
        if (
            masks.ndim != 2
            or masks.shape[1] != len(self.regions)
            or not np.isin(masks, [0, 1]).all()
        ):
            raise ValueError(
                "Region game requires binary masks with one column per region"
            )
        keys = [tuple(int(v) for v in row) for row in masks]
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        sequences: list[str] = []
        owners: list[tuple[int, ...]] = []
        for key in missing:
            # Use native inference directly for the full coalition.
            references = [self.sequence] if all(key) else self.references
            for reference in references:
                sequences.append(
                    "".join(
                        (self.sequence if present else reference)[
                            region.start: region.end
                        ]
                        for region, present in zip(self.regions, key)
                    )
                )
                owners.append(key)
        totals: dict[tuple[int, ...], list[np.ndarray]] = {key: [] for key in missing}
        for start in range(0, len(sequences), self.batch_size):
            batch = sequences[start: start + self.batch_size]
            predictions = predict_probabilities(self.adapter, batch)
            self.evaluated_sequences += len(batch)
            for key, value in zip(owners[start: start + self.batch_size], predictions):
                totals[key].append(value)
        for key, values in totals.items():
            self.cache[key] = np.mean(values, axis=0)
        return np.stack([self.cache[key] for key in keys])


def _kernel(
    game: RegionGame, seed: int, budget: int
) -> tuple[np.ndarray, np.ndarray, int]:
    regions = len(game.regions)
    if (
        isinstance(game.configuration, MethodConfiguration)
        and regions > 1
        and budget < min(2 * regions, 2 ** regions - 2)
    ):
        raise ValueError(
            "Coalition budget is insufficient for this region resolution; increase it"
            " explicitly"
        )
    with _KERNEL_LOCK:
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            explainer = shap.KernelExplainer(
                game, np.zeros((1, regions)), link="identity"
            )
            values = explainer.shap_values(
                np.ones((1, regions)),
                nsamples=min(budget, 2 ** regions - 2),
                l1_reg=0,
                silent=True,
            )
            normalized = ShapExplainer._normalise_values(
                values, 1, regions, len(game.adapter.descriptor.classes)
            )[0]
            sampled = int(getattr(explainer, "nsamplesAdded", 0))
            if isinstance(game.configuration, MethodConfiguration) and regions > 1:
                design = explainer.maskMatrix[:sampled]
                constrained = design[:, :-1] - design[:, -1, None]
                weighted = constrained * np.sqrt(
                    explainer.kernelWeights[:sampled, None]
                )
                if (
                    np.linalg.matrix_rank(weighted) < regions - 1
                    or np.linalg.cond(weighted) > 1e10
                ):
                    raise ValueError(
                        "Rank-deficient or poorly conditioned coalition design;"
                        " increase budget"
                    )
            return normalized, np.asarray(explainer.expected_value), sampled
        finally:
            np.random.set_state(state)


def explain_regions(
    adapter: BaseModelAdapter, sequence: str, configuration: AnalysisConfiguration
) -> dict[str, Any]:
    game = RegionGame(adapter, sequence, configuration)
    seed = derived_seed(configuration.seed, sequence)
    values, base, samples = _kernel(game, seed, configuration.coalition_budget)
    native = predict_probabilities(adapter, [sequence])[0]
    if not np.allclose(
        game(np.ones((1, len(game.regions))))[0], native, atol=1e-7, rtol=0
    ):
        raise ValueError("Full coalition differs from native model inference")
    residuals = base + values.sum(axis=1) - native
    if not np.isfinite(values).all() or np.max(np.abs(residuals)) > 1e-5:
        raise ValueError("Region SHAP probability reconstruction failed")
    exact = configuration.coalition_budget >= 2 ** len(game.regions) - 2
    stability: dict[str, Any] = {
        "status": "exact_enumeration" if exact else "not_assessed"
    }
    if configuration.stability_checks and not exact:
        repeated, _, _ = _kernel(
            game, (seed + 1) % 2 ** 32, configuration.coalition_budget
        )
        doubled, _, _ = _kernel(game, seed, configuration.coalition_budget * 2)
        delta = float(
            max(np.max(np.abs(values - repeated)), np.max(np.abs(values - doubled)))
        )
        stability = {
            "status": (
                "stable" if delta <= configuration.stability_tolerance else "unstable"
            ),
            "max_absolute_delta": delta,
            "tolerance": configuration.stability_tolerance,
            "repeat_seed": (seed + 1) % 2 ** 32,
            "doubled_budget": configuration.coalition_budget * 2,
            "repeated_attributions": repeated.tolist(),
            "doubled_attributions": doubled.tolist(),
        }
    diagnostics: dict[str, Any] = {}
    if isinstance(configuration, MethodConfiguration):
        from ..diagnostics import attribution_agreement, faithfulness
        from .references import distribution_shift

        comparisons = []
        if configuration.stability_checks:
            repeated, _, _ = _kernel(
                game, (seed + 1) % 2 ** 32, configuration.coalition_budget
            )
            doubled, _, _ = _kernel(game, seed, configuration.coalition_budget * 2)
            comparisons += attribution_agreement(
                values, repeated, adapter.descriptor.classes, "coalition_seed"
            )
            comparisons += attribution_agreement(
                values, doubled, adapter.descriptor.classes, "doubled_budget"
            )
        if configuration.reference_sensitivity:
            field = (
                "seed"
                if configuration.method_profile == "v2x-legacy"
                else "reference_seed"
            )
            alternate = configuration.model_copy(
                update={field: (getattr(configuration, field) + 1) % 2 ** 32}
            )
            other = RegionGame(adapter, sequence, alternate)
            varied, _, _ = _kernel(other, seed, configuration.coalition_budget)
            comparisons += attribution_agreement(
                values, varied, adapter.descriptor.classes, "reference_draw"
            )
            if configuration.references < 16:
                other = RegionGame(
                    adapter,
                    sequence,
                    configuration.model_copy(update={"references": 16}),
                )
                expanded, _, _ = _kernel(other, seed, configuration.coalition_budget)
                comparisons += attribution_agreement(
                    values,
                    expanded,
                    adapter.descriptor.classes,
                    "sixteen_nested_references",
                )
        diagnostics = dict(
            stability=comparisons,
            distribution_shift=distribution_shift(sequence, game.references),
            faithfulness=(
                faithfulness(
                    adapter,
                    sequence,
                    game.regions,
                    values,
                    derived_seed(configuration.diagnostic_seed, sequence),
                    configuration.diagnostic_draws,
                )
                if configuration.faithfulness
                else []
            ),
        )
    warnings = [
        (
            "Region SHAP under a shuffled-sequence reference game; "
            "hybrid sequences may be outside the natural sequence "
            "distribution."
        ),
        "Additivity is not evidence of approximation convergence or causal biology.",
    ]
    if stability["status"] in {"not_assessed", "unstable"}:
        warnings.append(f"Approximation stability: {stability['status']}.")
    if (
        isinstance(configuration, MethodConfiguration)
        and configuration.method_profile == "v2x-development"
    ):
        warnings[0] = (
            f"Region SHAP under the recorded {configuration.reference_strategy}"
            " reference game; hybrid sequences may be outside the natural sequence"
            " distribution."
        )
    if any(r["status"] == "unstable" for r in diagnostics.get("stability", [])):
        warnings.append(
            "Attribution sampling is unstable under recorded sensitivity checks."
        )
    return {
        "features": game.regions,
        "feature_values": [1.0] * len(game.regions),
        "attributions": values.tolist(),
        "base_values": base.tolist(),
        "residuals": residuals.tolist(),
        "explainer": {
            "method": "region_kernel",
            "output_space": "probability",
            "link": "identity",
            "l1_reg": 0,
            "region_policy": (
                REGION_POLICY
                if getattr(configuration, "region_strategy", "legacy") == "legacy"
                else getattr(configuration, "region_strategy") + ":v1"
            ),
            "reference_policy": (
                REFERENCE_POLICY
                if getattr(configuration, "reference_strategy", "whole_shuffle")
                   == "whole_shuffle"
                else getattr(configuration, "reference_strategy") + ":v1"
            ),
            "references": [
                {"sequence": s, "sha256": sequence_sha256(s)} for s in game.references
            ],
            "seed": seed,
            "run_seed": configuration.seed,
            "coalition_budget": configuration.coalition_budget,
            "sampled_nontrivial_coalitions": samples,
            "unique_coalitions_evaluated": len(game.cache),
            "native_sequence_evaluations": game.evaluated_sequences,
            "stability": stability,
            **(
                {
                    "diagnostics": diagnostics,
                    "reference_seed": (
                        configuration.seed
                        if configuration.method_profile == "v2x-legacy"
                        else configuration.reference_seed
                    ),
                    "reference_pool_id": configuration.reference_pool_id,
                }
                if isinstance(configuration, MethodConfiguration)
                else {}
            ),
        },
        "warnings": warnings,
    }
