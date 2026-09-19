"""Attribution-blind coordinate/control feasibility, before final explanations."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from mapexploc.execution import atomic_json, fingerprint
from mapexploc.explainers.regions import sequence_regions
from mapexploc.study import FAMILIES, planned_power, required_groups


def masks(regions, intervals):
    result = []
    for i, region in enumerate(regions):
        if not any(a <= region.start and region.end <= b for a, b in intervals):
            continue
        controls = [
            j
            for j, other in enumerate(regions)
            if other.category == region.category
            and other.end - other.start == region.end - region.start
            and not any(a < other.end and other.start < b for a, b in intervals)
        ]
        if controls:
            result.append((i, tuple(controls)))
    return tuple(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if any(m["split"] != "evaluation" for m in manifest["members"]):
        raise ValueError("Expected the frozen evaluation manifest")
    rows = []
    for strategy in ("legacy", "windows_10", "windows_5"):
        for family in FAMILIES:
            strata = defaultdict(list)
            failures = []
            members = [m for m in manifest["members"] if family in m["families"]]
            for member in members:
                regions = sequence_regions(len(member["sequence"]), strategy)
                intervals = [(f["start"], f["end"]) for f in member["families"][family]]
                selected = masks(regions, intervals)
                if not selected:
                    failures.append(member["protein_id"])
                    continue
                categories = tuple(
                    sorted(
                        {
                            str(r.category)
                            for r in regions
                            if any(a <= r.start and r.end <= b for a, b in intervals)
                        }
                    )
                )
                strata[(len(member["sequence"]) // 50, categories)].append(
                    (member["protein_id"], regions, intervals)
                )
            matched = []
            varying = []
            excluded = []
            for group in strata.values():
                valid = len(group) >= 2 and all(
                    masks(regions, donor)
                    for _, regions, _ in group
                    for _, _, donor in group
                )
                if not valid:
                    excluded.extend(pid for pid, _, _ in group)
                    continue
                matched.extend(pid for pid, _, _ in group)
                if any(
                    len({masks(regions, donor) for _, _, donor in group}) > 1
                    for _, regions, _ in group
                ):
                    varying.extend(pid for pid, _, _ in group)
            rows.append(
                {
                    "family": family,
                    "region_strategy": strategy,
                    "initial_groups": len(members),
                    "matched_groups": len(matched),
                    "groups_in_variable_coordinate_strata": len(varying),
                    "no_region_or_control": failures,
                    "incompatible_pairing_or_singleton": excluded,
                    "retained_nominal_power": planned_power(len(matched)),
                    "required_groups": required_groups(),
                    "status": (
                        "exploratory_fixed_terminal_sequence_controls_required"
                        if family in {"peroxisomal_pts1", "er_retention"}
                        else (
                            "underpowered_after_coordinate_exclusions"
                            if len(matched) < required_groups()
                            else (
                                "degenerate_coordinate_pairing"
                                if not varying
                                else "pending_development_power_and_null_checks"
                            )
                        )
                    ),
                }
            )
    payload = {
        "manifest_sha256": fingerprint(manifest),
        "rows": rows,
        "attributions_accessed": False,
        "scope": (
            "Coordinate feasibility only, not final family eligibility or biological "
            "results. The profile is selected from development stability/faithfulness, "
            "never by maximizing these evaluation counts."
        ),
    }
    if args.output.exists() and json.loads(args.output.read_text()) != payload:
        raise ValueError("Do not overwrite frozen coordinate feasibility")
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
