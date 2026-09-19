"""Resumable development benchmark. No final-evaluation manifest is accepted."""

import argparse
import json
from pathlib import Path

from mapexploc import ExecutionOptions, Protein, run_analysis, write_report
from mapexploc.benchmark import select_profile
from mapexploc.deeploc import DeepLocAdapter
from mapexploc.execution import atomic_json, fingerprint
from mapexploc.explainers.regions import sequence_regions
from mapexploc.study import FAMILIES


def prepare_panel(development):
    if any(m["split"] != "development" for m in development["members"]):
        raise ValueError("Only a frozen development manifest is accepted")
    members = sorted(
        development["members"],
        key=lambda m: (fingerprint(m["protein_id"]), m["protein_id"]),
    )
    selected = {}
    # At least two per available family, then balanced length bins. Rare families
    # may contribute fewer; no replacement is based on explanations or scores.
    for family in FAMILIES:
        for m in [m for m in members if family in m["families"]][:2]:
            selected[m["protein_id"]] = m
    for lower, upper in [(10, 100), (101, 250), (251, 512), (513, 1022)]:
        for m in [m for m in members if lower <= len(m["sequence"]) <= upper][:6]:
            if len(selected) == 24:
                break
            selected[m["protein_id"]] = m
    for m in members:
        if len(selected) == 24:
            break
        selected[m["protein_id"]] = m
    if len(selected) != 24:
        raise ValueError("Need 24 independent development proteins")
    return list(selected.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--development", type=Path, required=True)
    p.add_argument("--reference-pool", type=Path, required=True)
    p.add_argument("--fast-config", type=Path, required=True)
    p.add_argument("--accurate-config", type=Path, required=True)
    p.add_argument("--fast-parity", type=Path)
    p.add_argument("--accurate-parity", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    panel = prepare_panel(json.loads(args.development.read_text()))
    pool = json.loads(args.reference_pool.read_text())
    if any(m["split"] != "reference_pool" for m in pool["members"]):
        raise ValueError("Reference pool split is not independent")
    if {m["group"] for m in panel} & {m["group"] for m in pool["members"]}:
        raise ValueError("Reference pool overlaps development groups")
    manifest = {
        "protocol": "development-reference-comparison-v1",
        "proteins": panel,
        "resolution_subset": [m["protein_id"] for m in panel[:12]],
        "reference_pool_sha256": fingerprint(pool),
        "candidate_strategies": ["whole_shuffle", "block_shuffle", "positional_pool"],
        "seed": 42,
        "reference_seed": 42,
        "diagnostic_seed": 20260919,
        "coalition_budget": 512,
        "references": 4,
        "stability_checks": True,
        "reference_sensitivity": True,
        "faithfulness": True,
        "resolution_stage": (
            "Advance at most two references by complete both-mode development gates;"
            " compare legacy, windows_10, windows_5 on fixed first twelve proteins"
            " before final selection"
        ),
        "status": "protocol_frozen_before_explanations",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = args.output / "benchmark-manifest.json"
    if frozen.exists() and json.loads(frozen.read_text()) != manifest:
        raise ValueError(
            "Benchmark manifest changed; do not overwrite a frozen"
            " method-development run"
        )
    atomic_json(frozen, manifest)
    frozen_selection = args.output / "selected-profile.json"
    if frozen_selection.exists():
        selected = json.loads(frozen_selection.read_text())
        checksum = selected.pop("selection_sha256")
        if fingerprint(selected) != checksum or selected.get(
            "benchmark_manifest_sha256"
        ) != fingerprint(manifest):
            raise ValueError(
                "Frozen selection integrity or benchmark identity mismatch"
            )
        print(json.dumps({"status": "already_frozen", "selection_sha256": checksum}))
        return
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "prepared_not_executed",
                    "proteins": len(panel),
                    "models": 2,
                    "candidates": 3,
                    "base_evaluations_upper_bound": (
                        sum(
                            (
                                min(
                                    512,
                                    2 ** len(sequence_regions(len(m["sequence"]))) - 2,
                                )
                                + 1
                            )
                            * 4
                            + 1
                            for m in panel
                        )
                        * 6
                    ),
                    "additional_diagnostics": (
                        "reference resampling, 16 references, repeated/doubled"
                        " coalition budgets and held-out perturbations add substantial"
                        " cost"
                    ),
                    "estimated_seconds": None,
                    "manifest": str(frozen),
                },
                indent=2,
            )
        )
        return
    qualifications = {}
    for mode, path in [("fast", args.fast_parity), ("accurate", args.accurate_parity)]:
        if path is None:
            raise ValueError(
                "Execution requires passed --fast-parity and --accurate-parity evidence"
            )
        evidence = json.loads(path.read_text())
        if (
            evidence.get("status") != "passed"
            or not evidence.get("native_decisions_agree")
            or evidence["model"]["provenance"].get("mode") != mode
            or evidence["errors"]["full_precision"] > 1e-5
            or evidence["errors"]["native_csv"] > 5.1e-5
        ):
            raise ValueError(f"Missing or failed real native {mode} qualification")
        qualifications[mode] = evidence["model"]
    proteins = [
        Protein.model_validate(
            {k: m[k] for k in ("protein_id", "sequence", "group", "split")}
        )
        for m in panel
    ]
    reports = {strategy: {} for strategy in manifest["candidate_strategies"]}
    # Check both actual identities and current resource readiness before launching
    # the expensive Fast stage. A missing Accurate prerequisite is not a skip.
    from mapexploc.catalog import ConfiguredModel

    for mode, path in [("fast", args.fast_config), ("accurate", args.accurate_config)]:
        config = json.loads(path.read_text())
        readiness = ConfiguredModel("deeploc2", config).summary(mode)
        if readiness["readiness"] == "unavailable":
            raise RuntimeError(f"{mode} preflight failed: {readiness['issues']}")
        with DeepLocAdapter.from_config(config) as adapter:
            if adapter.descriptor.model_dump(mode="json") != qualifications[mode]:
                raise ValueError(
                    f"{mode} model/runtime differs from native parity evidence"
                )
    for mode, path in [("fast", args.fast_config), ("accurate", args.accurate_config)]:
        configuration = json.loads(path.read_text())
        if configuration.get("mode", "fast") != mode:
            raise ValueError("Native mode configuration mismatch")
        with DeepLocAdapter.from_config(configuration) as adapter:
            for strategy in reports:
                output = args.output / mode / strategy
                config = dict(
                    method_profile="v2x-development",
                    reference_strategy=strategy,
                    cohort_id="development-24",
                    selection_criteria="Fixed attribution-blind benchmark panel",
                    seed=42,
                    reference_seed=42,
                    diagnostic_seed=20260919,
                    coalition_budget=512,
                    references=4,
                    stability_checks=True,
                    reference_sensitivity=True,
                    faithfulness=True,
                )
                if strategy == "positional_pool":
                    config.update(
                        reference_pool=[m["sequence"] for m in pool["members"]],
                        reference_pool_id=fingerprint(pool),
                    )
                report = run_analysis(
                    adapter,
                    proteins,
                    config,
                    execution=ExecutionOptions(
                        cache_directory=args.output / "cache",
                        restart_directory=output / "restart",
                        progress=lambda event: print(
                            json.dumps({"mode": mode, "strategy": strategy, **event}),
                            flush=True,
                        ),
                    ),
                )
                write_report(report, output)
                reports[strategy][mode] = report
    result = select_profile(
        {k: v for k, v in reports.items() if k != "whole_shuffle"},
        reports["whole_shuffle"],
    )
    result["stage"] = "reference_screen"
    atomic_json(args.output / "reference-selection.json", result)
    # Rank for advancement before looking at any finer-grid results. The legacy
    # reference stays as a comparator even when it is not among the top two.
    ranking = []
    for name, outcome in result["candidates"].items():
        scores = outcome["modes"]
        improvement = min(
            (
                s["paired_faithfulness_improvement"]
                if s["paired_faithfulness_improvement"] is not None
                else -1
            )
            for s in scores.values()
        )
        ranking.append(
            (
                not outcome["eligible"],
                -improvement,
                sum(s["elapsed_seconds"] for s in scores.values()),
                name,
            )
        )
    ranking.append(
        (
            False,
            0,
            sum(
                r.measurements.elapsed_seconds
                for r in reports["whole_shuffle"].values()
            ),
            "whole_shuffle",
        )
    )
    advanced = [row[-1] for row in sorted(ranking)[:2]]
    atomic_json(
        args.output / "resolution-advancement.json",
        {
            "strategies": advanced,
            "protein_ids": manifest["resolution_subset"],
            "rule": (
                "Both-mode eligibility, worst-mode faithfulness, then cost; no"
                " evaluation attributions accessed"
            ),
        },
    )
    resolution = {}
    subset = proteins[:12]
    configurations = {}
    for mode, path in [("fast", args.fast_config), ("accurate", args.accurate_config)]:
        with DeepLocAdapter.from_config(json.loads(path.read_text())) as adapter:
            for strategy in list(dict.fromkeys(["whole_shuffle", *advanced])):
                for grid in (
                    ["legacy", "windows_10", "windows_5"]
                    if strategy in advanced
                    else ["legacy"]
                ):
                    name = strategy + ":" + grid
                    config = reports[strategy][mode].configuration.model_dump()
                    config.update(
                        region_strategy=grid,
                        cohort_id="development-resolution-12",
                        selection_criteria=(
                            "Prespecified first twelve development panel proteins"
                        ),
                    )
                    configurations[name] = config
                    output = args.output / "resolution" / mode / name.replace(":", "-")
                    report = run_analysis(
                        adapter,
                        subset,
                        config,
                        execution=ExecutionOptions(
                            cache_directory=args.output / "cache",
                            restart_directory=output / "restart",
                            progress=lambda event: print(
                                json.dumps({"mode": mode, "candidate": name, **event}),
                                flush=True,
                            ),
                        ),
                    )
                    write_report(report, output)
                    resolution.setdefault(name, {})[mode] = report
    baseline = resolution.pop("whole_shuffle:legacy")
    final = select_profile(resolution, baseline)
    final["stage"] = "resolution_selection_complete"
    final["configuration"] = configurations[
        (
            "whole_shuffle:legacy"
            if final["selected"] == "v2x-legacy"
            else final["selected"]
        )
    ]
    final["configuration"]["cohort_id"] = None
    final["configuration"]["selection_criteria"] = "All supplied proteins; no sampling"
    final["benchmark_manifest_sha256"] = fingerprint(manifest)
    final["models"] = {
        mode: report.model.model_dump(mode="json") for mode, report in baseline.items()
    }
    final["selection_sha256"] = fingerprint(final)
    frozen_selection = args.output / "selected-profile.json"
    if frozen_selection.exists() and json.loads(frozen_selection.read_text()) != final:
        raise ValueError(
            "Existing selected profile differs; a completed freeze cannot be"
            " overwritten"
        )
    atomic_json(frozen_selection, final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
