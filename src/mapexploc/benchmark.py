"""Development-only method selection with explicit, conservative promotion gates."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .report_v3 import AnalysisReportV3


def development_scores(report: AnalysisReportV3) -> dict[str, Any]:
    if any(r.protein.split != "development" for r in report.results):
        raise ValueError("Method selection accepts development proteins only")
    diagnostics = {d.protein_id: d for d in report.diagnostics}
    groups: dict[str, list[float]] = defaultdict(list)
    stable = informative = 0
    for local in report.results:
        diagnostic = diagnostics[local.protein.protein_id]
        for row in diagnostic.stability:
            if row.status != "uninformative":
                informative += 1
                stable += int(row.status == "stable")
        effects = [
            r["matched_advantage"]
            for r in diagnostic.faithfulness
            if r.get("status") == "assessed"
        ]
        if effects:
            groups[local.protein.group or local.sequence_sha256].append(
                float(np.mean(effects))
            )
    return {
        "stability_fraction": stable / informative if informative else None,
        "informative_comparisons": informative,
        "group_faithfulness": {g: float(np.mean(v)) for g, v in sorted(groups.items())},
        "elapsed_seconds": report.measurements.elapsed_seconds,
    }


def select_profile(
    candidates: dict[str, dict[str, AnalysisReportV3]],
    baseline: dict[str, AnalysisReportV3],
    seed: int = 20260919,
) -> dict[str, Any]:
    if set(baseline) != {"fast", "accurate"} or any(
        set(modes) != {"fast", "accurate"} for modes in candidates.values()
    ):
        raise ValueError(
            "Profile selection requires complete real Fast and Accurate comparisons"
        )
    baseline_scores = {
        mode: development_scores(report) for mode, report in baseline.items()
    }
    outcomes: dict[str, Any] = {}
    for name, modes in sorted(candidates.items()):
        checks = {}
        for mode, report in modes.items():
            if (
                report.runtime.model_mode != mode
                or baseline[mode].model.checkpoint_sha256
                != report.model.checkpoint_sha256
            ):
                raise ValueError("Candidate and baseline model identities differ")
            score = development_scores(report)
            reference = baseline_scores[mode]
            if {(r.protein.protein_id, r.sequence_sha256) for r in report.results} != {
                (r.protein.protein_id, r.sequence_sha256)
                for r in baseline[mode].results
            }:
                raise ValueError("Candidate and baseline development membership differ")
            shared = sorted(
                set(score["group_faithfulness"]) & set(reference["group_faithfulness"])
            )
            differences = np.asarray(
                [
                    score["group_faithfulness"][g] - reference["group_faithfulness"][g]
                    for g in shared
                ]
            )
            interval = None
            if len(shared) >= 12:
                rng = np.random.default_rng(seed)
                boot = rng.choice(differences, size=(1000, len(shared))).mean(axis=1)
                interval = np.quantile(boot, [0.025, 0.975]).tolist()
            passed = bool(
                interval
                and interval[0] > 0
                and score["stability_fraction"] is not None
                and score["stability_fraction"] >= 0.9
            )
            checks[mode] = {
                **score,
                "paired_groups": len(shared),
                "paired_faithfulness_improvement": (
                    float(differences.mean()) if len(shared) else None
                ),
                "paired_group_bootstrap_95": interval,
                "passed": passed,
            }
        outcomes[name] = {
            "modes": checks,
            "eligible": all(c["passed"] for c in checks.values()),
        }
    eligible = [name for name, result in outcomes.items() if result["eligible"]]

    def rank(name: str) -> tuple[float, float, int, str]:
        modes = outcomes[name]["modes"]
        return (
            -min(m["paired_faithfulness_improvement"] for m in modes.values()),
            sum(m["elapsed_seconds"] for m in modes.values()),
            {"whole_shuffle": 0, "block_shuffle": 1, "positional_pool": 2}.get(
                name.split(":")[0], 3
            ),
            name,
        )

    selected = min(eligible, key=rank) if eligible else "v2x-legacy"
    return dict(
        status="selected" if eligible else "legacy_retained",
        selected=selected,
        seed=seed,
        candidates=outcomes,
        used_biological_overlap_for_selection=False,
        final_evaluation_accessed=False,
        interpretation=(
            "Development-only selection. Promotion requires independent final"
            " evaluation and release qualification; this object alone does not register"
            " a new default."
        ),
    )
