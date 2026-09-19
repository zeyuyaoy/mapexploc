"""Exploratory equal-group summaries. Additive SHAP is never overwritten."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from scipy.stats import trim_mean

from .report_v2 import LocalExplanation


def group_summaries(
    results: list[LocalExplanation], seed: int, replicates: int
) -> list[dict[str, Any]]:
    if len(results) < 2:
        return []
    by_sequence: dict[str, str] = {}
    collected: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    counts: dict[tuple[str, str, str], int] = {}
    for local in sorted(results, key=lambda r: r.protein.protein_id):
        group = local.protein.group or local.sequence_sha256
        if (
            local.sequence_sha256 in by_sequence
            and by_sequence[local.sequence_sha256] != group
        ):
            raise ValueError(
                "Identical sequences cannot belong to different independent groups"
            )
        by_sequence[local.sequence_sha256] = group
        for ci, label in enumerate(local.explained_classes):
            within: dict[tuple[str, str, str], list[float]] = {}
            for feature, value in zip(local.features, local.attributions[ci]):
                identity = (
                    feature.category
                    if feature.kind == "sequence_region"
                    else feature.feature_id
                )
                measures = {"signed": value, "absolute": abs(value)}
                if feature.start is not None and feature.end is not None:
                    measures.update(
                        signed_density=value / (feature.end - feature.start),
                        absolute_density=abs(value) / (feature.end - feature.start),
                    )
                for statistic, number in measures.items():
                    within.setdefault((label, str(identity), statistic), []).append(
                        number
                    )
            for key, within_values in within.items():
                collected.setdefault(key, {}).setdefault(group, []).append(
                    float(np.mean(within_values))
                )
                counts[key] = counts.get(key, 0) + 1
    rows = []
    for key, groups in sorted(collected.items()):
        values = np.asarray([np.mean(groups[g]) for g in sorted(groups)])
        derived = int.from_bytes(
            hashlib.sha256(f"{seed}:{key}".encode()).digest()[:4], "big"
        )
        rng = np.random.default_rng(derived)
        # Chunking keeps the bootstrap bounded even for large cohorts.
        draws: list[float] = []
        for start in range(0, replicates, 100):
            sampled = rng.choice(
                values, size=(min(100, replicates - start), len(values))
            )
            draws.extend(sampled.mean(axis=1))
        rows.append(
            dict(
                class_id=key[0],
                feature_id=key[1],
                statistic=key[2],
                mean=float(values.mean()),
                median=float(np.median(values)),
                trimmed_mean=float(trim_mean(values, 0.1)),
                interval_95=[float(v) for v in np.quantile(draws, [0.025, 0.975])],
                group_count=len(values),
                protein_count=counts[key],
                bootstrap_replicates=replicates,
                denominator=(
                    "Equal independent-group weight; within-group protein mean;"
                    " within-protein feature/category mean"
                ),
                status=(
                    "exploratory_small_sample" if len(values) < 20 else "exploratory"
                ),
            )
        )
    return rows
