"""Model-behavior comparisons; incompatible reference games are never pooled."""

from __future__ import annotations

from typing import Any

import numpy as np

from .diagnostics import attribution_agreement
from .report_v2 import AnalysisReport, FeatureDefinition, LocalExplanation


def _region_length(feature: FeatureDefinition) -> int:
    assert feature.start is not None and feature.end is not None
    return feature.end - feature.start


def _game(local: LocalExplanation) -> dict[str, Any]:
    return {
        key: local.explainer.get(key)
        for key in (
            "method",
            "output_space",
            "reference_policy",
            "region_policy",
            "references",
            "reference_pool_id",
        )
    }


def compare_reports(left: AnalysisReport, right: AnalysisReport) -> dict[str, Any]:
    """Pair by ID AND sequence checksum, align class labels, disclose unmatched rows."""
    a = {(r.protein.protein_id, r.sequence_sha256): r for r in left.results}
    b = {(r.protein.protein_id, r.sequence_sha256): r for r in right.results}
    classes = [c for c in left.model.classes if c in right.model.classes]
    rows = []
    for key in sorted(a.keys() & b.keys()):
        x, y = a[key], b[key]
        indices_a = [x.explained_classes.index(c) for c in classes]
        indices_b = [y.explained_classes.index(c) for c in classes]
        compatible = _game(x) == _game(y) and [
            (f.feature_id, f.start, f.end) for f in x.features
        ] == [(f.feature_id, f.start, f.end) for f in y.features]
        decision_union = set(x.decisions) | set(y.decisions)
        row = dict(
            protein_id=key[0],
            sequence_sha256=key[1],
            probability_differences={
                c: float(y.probabilities[j] - x.probabilities[i])
                for c, i, j in zip(classes, indices_a, indices_b)
            },
            decisions_equal=set(x.decisions) == set(y.decisions),
            decision_jaccard=(
                len(set(x.decisions) & set(y.decisions)) / len(decision_union)
                if decision_union
                else 1.0
            ),
            attribution_comparability=(
                "common_reference_game" if compatible else "confounded_by_methodology"
            ),
            attribution_agreement=(
                attribution_agreement(
                    np.asarray(x.attributions)[indices_a],
                    np.asarray(y.attributions)[indices_b],
                    classes,
                    "cross_model",
                    methodological=True,
                )
                if compatible and classes
                else []
            ),
        )
        # Coordinate overlap does not imply residue attribution.
        coverage = []
        for label, ia, ib in zip(classes, indices_a, indices_b):
            sets = []
            for local, index in ((x, ia), (y, ib)):
                ranked = sorted(
                    zip(local.features, local.attributions[index]),
                    key=lambda fv: -abs(fv[1]),
                )
                selected = ranked[: max(1, int(np.ceil(0.2 * len(ranked))))]
                sets.append(
                    {
                        p
                        for f, v in selected
                        if f.start is not None and f.end is not None
                        for p in range(f.start, f.end)
                    }
                )
            union = sets[0] | sets[1]
            coverage.append(
                dict(
                    class_id=label,
                    top_region_residue_coverage_jaccard=(
                        len(sets[0] & sets[1]) / len(union) if union else None
                    ),
                )
            )
        row["coverage_overlap"] = coverage
        row["annotation_overlap"] = [
            dict(
                model_id=model.model.model_id,
                annotations=[
                    dict(
                        annotation_id=ann.annotation_id,
                        start=ann.start,
                        end=ann.end,
                        source=ann.source,
                        sequence_sha256=ann.sequence_sha256,
                    )
                    for ann in local.annotations
                ],
            )
            for model, local in ((left, x), (right, y))
        ]
        row["biological_region_concordance"] = [
            {
                "model_id": model.model.model_id,
                "annotations": [
                    {
                        "annotation_id": ann.annotation_id,
                        "class_id": label,
                        "complete_region_count": len(indices),
                        "positive_attribution_density": (
                            (
                                sum(
                                    max(local.attributions[ci][i], 0.0) for i in indices
                                )
                                / sum(
                                    _region_length(local.features[i]) for i in indices
                                )
                            )
                            if indices
                            else None
                        ),
                        "status": (
                            "descriptive_overlap"
                            if indices
                            else "insufficient_resolution"
                        ),
                        "uncertain_annotation": ann.uncertain,
                    }
                    for ann in local.annotations
                    for ci, label in enumerate(local.explained_classes)
                    for indices in [
                        [
                            i
                            for i, f in enumerate(local.features)
                            if not ann.uncertain
                            and f.start is not None
                            and f.end is not None
                            and ann.start <= f.start
                            and f.end <= ann.end
                        ]
                    ]
                ],
            }
            for model, local in ((left, x), (right, y))
        ]
        rows.append(row)
    return dict(
        schema_version=1,
        kind="model_behavior_comparison",
        left_model=left.model.model_dump(mode="json"),
        right_model=right.model.model_dump(mode="json"),
        classes=classes,
        unmatched_left=[
            dict(protein_id=k[0], sequence_sha256=k[1])
            for k in sorted(a.keys() - b.keys())
        ],
        unmatched_right=[
            dict(protein_id=k[0], sequence_sha256=k[1])
            for k in sorted(b.keys() - a.keys())
        ],
        results=rows,
        interpretation=(
            "Model-specific predictions and explanatory behavior. Disagreement is not"
            " evidence that one model is biologically wrong."
        ),
    )
