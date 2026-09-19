"""Attribution-blind curation and preregistered study integrity checks."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from scipy.stats import nct, t

from .provenance import sequence_sha256

FAMILIES = {
    "secretion_signal": "Extracellular",
    "mitochondrial_targeting": "Mitochondrion",
    "nuclear_localization": "Nucleus",
    "plasma_membrane_helix": "Cell membrane",
    "peroxisomal_pts1": "Peroxisome",
    "er_retention": "Endoplasmic reticulum",
}


def experimentally_supported(feature: dict[str, Any]) -> bool:
    return any(
        e.get("evidenceCode") == "ECO:0000269" for e in feature.get("evidences", [])
    )


def precise_feature(feature: dict[str, Any], length: int) -> bool:
    location = feature.get("location", {})
    endpoints = [location.get(k, {}) for k in ("start", "end")]
    return (
        all(
            isinstance(p.get("value"), int) and p.get("modifier", "EXACT") == "EXACT"
            for p in endpoints
        )
        and 1 <= endpoints[0]["value"] <= endpoints[1]["value"] <= length
    )


def eligible_features(record: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Evidence belongs to this feature. Reviewed entry status alone is insufficient."""
    sequence = record["sequence"]["value"]
    locations = [
        loc.get("location", {})
        for comment in record.get("comments", [])
        if comment.get("commentType") == "SUBCELLULAR LOCATION"
        for loc in comment.get("subcellularLocations", [])
    ]
    location_names = [loc.get("value", "").lower() for loc in locations]
    plasma = any(
        loc.get("value", "").lower() in {"cell membrane", "plasma membrane"}
        and experimentally_supported(loc)
        for loc in locations
    )
    result: dict[str, list[dict[str, Any]]] = {}
    for feature in record.get("features", []):
        if not experimentally_supported(feature) or not precise_feature(
            feature, len(sequence)
        ):
            continue
        if feature["location"].get("sequence") not in (
                None,
                record["primaryAccession"],
        ):
            continue
        kind = feature["type"]
        description = feature.get("description", "").lower()
        start, end = (feature["location"][p]["value"] for p in ("start", "end"))
        family = None
        subtype = None
        if (
            kind == "Signal"
            and start == 1
            and any("secreted" in name for name in location_names)
        ):
            family = "secretion_signal"
        elif kind == "Transit peptide" and start == 1 and "mitochondri" in description:
            family = "mitochondrial_targeting"
        elif kind == "Motif" and "nuclear localization" in description:
            family = "nuclear_localization"
        elif kind == "Transmembrane" and plasma and "helical" in description:
            family = "plasma_membrane_helix"
        elif (
            kind == "Motif"
            and "peroxisom" in description
            and end == len(sequence)
            and (
                "pts1" in description
                or "microbody" in description
                or end - start + 1 == 3
            )
        ):
            family, subtype = "peroxisomal_pts1", "terminal_pts1"
        elif (
            kind == "Motif"
            and "retention" in description
            and ("er " in description or "endoplasmic" in description)
            and "golgi" not in description
            and end == len(sequence)
        ):
            family = "er_retention"
            motif = sequence[start - 1: end]
            subtype = (
                "KDEL_like"
                if motif.endswith(("KDEL", "HDEL", "RDEL"))
                else (
                    "dilysine"
                    if len(motif) >= 4
                       and (
                           motif[-4:-2] == "KK"
                           or (len(motif) >= 5 and motif[-5] == "K" and motif[-3] == "K")
                       )
                    else None
                )
            )
            if subtype is None:
                family = None
        if family:
            result.setdefault(family, []).append(
                {"start": start - 1, "end": end, "subtype": subtype, "feature": feature}
            )
    return result


def planned_power(n: int, effect: float = 0.5, hypotheses: int = 12) -> float:
    """Two-sided paired group effect, conservative Bonferroni planning bound."""
    if n < 2:
        return 0.0
    critical = t.ppf(1 - 0.05 / (2 * hypotheses), n - 1)
    return float(
        nct.sf(critical, n - 1, effect * np.sqrt(n))
        + nct.sf(critical, n - 1, -effect * np.sqrt(n))
    )


def required_groups() -> int:
    return next(n for n in range(60, 10000) if planned_power(n) >= 0.8)


def split_group(group: str, seed: int = 20260919) -> str:
    # Split before choosing representatives; exclude evaluation data from references.
    bucket = (
        int.from_bytes(hashlib.sha256(f"{seed}:{group}".encode()).digest()[:4], "big")
        % 10
    )
    return (
        "development"
        if bucket < 2
        else "reference_pool" if bucket < 4 else "evaluation"
    )


def audit_manifests(
    manifests: list[dict[str, Any]], historical: dict[str, str]
) -> None:
    seen_groups: dict[str, str] = {}
    seen_sequences: dict[str, str] = {}
    for manifest in manifests:
        ids = set()
        for member in manifest["members"]:
            pid = member["protein_id"]
            group = member["group"]
            split = member["split"]
            if pid in ids:
                raise ValueError("Duplicate cohort representative")
            ids.add(pid)
            if pid in historical or group in set(historical.values()):
                raise ValueError("Historical cohort groups must be excluded")
            for seen, key in (
                    (seen_groups, group),
                    (seen_sequences, member["sequence_sha256"]),
            ):
                if key in seen and seen[key] != split:
                    raise ValueError("Cross-split sequence/group leakage")
                seen[key] = split
            if (
                member.get("sequence")
                and sequence_sha256(member["sequence"]) != member["sequence_sha256"]
            ):
                raise ValueError("Frozen cohort sequence hash mismatch")


def holm(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid hypothesis p-values")
    order = np.argsort(values, kind="stable")
    adjusted = np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1))
    result = np.empty_like(values)
    result[order] = np.minimum(1, adjusted)
    return [float(value) for value in result]
