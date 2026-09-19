"""Annotation-independent reference distributions for development comparisons."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

import numpy as np

from ..features import normalize_protein_sequence
from ..methods import MethodConfiguration
from ..provenance import sequence_sha256

AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))


def length_bin(length: int) -> int:
    return int(np.searchsorted([100, 250, 512, 1022, 2048, 4000], length, side="left"))


def position_stratum(position: int, length: int) -> str:
    terminal = min(50, length // 2)
    return (
        "N_terminal"
        if position < terminal
        else "C_terminal" if position >= length - terminal else "interior"
    )


def references(sequence: str, config: MethodConfiguration) -> list[str]:
    seed = int.from_bytes(
        hashlib.sha256(
            f"{config.reference_seed}:{sequence_sha256(sequence)}".encode()
        ).digest()[:4],
        "big",
    )
    rng = np.random.default_rng(seed)
    residues = np.array(list(sequence))
    if config.reference_strategy == "whole_shuffle":
        return ["".join(rng.permutation(residues)) for _ in range(config.references)]
    if config.reference_strategy == "block_shuffle":
        return [
            "".join(
                "".join(rng.permutation(residues[i: i + 20]))
                for i in range(0, len(sequence), 20)
            )
            for _ in range(config.references)
        ]
    pool = sorted({normalize_protein_sequence(s) for s in config.reference_pool})
    eligible = [s for s in pool if length_bin(len(s)) == length_bin(len(sequence))]
    if not eligible:
        raise ValueError(
            "Frozen positional pool has no sequences in the requested length stratum"
        )
    counts: dict[str, Counter[str]] = {
        name: Counter() for name in ("N_terminal", "interior", "C_terminal")
    }
    for s in eligible:
        for i, residue in enumerate(s):
            counts[position_stratum(i, len(s))][residue] += 1
    distributions = {}
    for name, count in counts.items():
        if count:
            p = np.array([count[a] for a in AA], dtype=float)
            distributions[name] = p / p.sum()
    result = []
    for _ in range(config.references):
        result.append(
            "".join(
                str(rng.choice(AA, p=distributions[position_stratum(i, len(sequence))]))
                for i in range(len(sequence))
            )
        )
    return result


def distribution_shift(sequence: str, altered: list[str]) -> dict[str, Any]:
    """Descriptive distances, not evidence of naturalness or distribution membership."""

    def frequencies(s: str, k: int) -> dict[str, float]:
        c = Counter(s[i: i + k] for i in range(len(s) - k + 1))
        n = max(len(s) - k + 1, 1)
        return {a: v / n for a, v in c.items()}

    def distance(a: str, b: str, k: int) -> float:
        x, y = frequencies(a, k), frequencies(b, k)
        return sum(abs(x.get(t, 0) - y.get(t, 0)) for t in x.keys() | y.keys())

    def charge(s: str) -> float:
        return sum(s.count(a) for a in "KR") / len(s) - sum(
            s.count(a) for a in "DE"
        ) / len(s)

    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    analysis: Any = ProteinAnalysis
    rows = []
    for s in altered:
        rows.append(
            dict(
                composition_l1=distance(sequence, s, 1),
                dipeptide_l1=distance(sequence, s, 2),
                charge_delta=charge(s) - charge(sequence),
                hydropathy_delta=analysis(s).gravy() - analysis(sequence).gravy(),
                n_terminal_composition_l1=distance(sequence[:50], s[:50], 1),
                c_terminal_composition_l1=distance(sequence[-50:], s[-50:], 1),
                length_preserved=len(s) == len(sequence),
                tokens_valid=not bool(set(s) - set(AA)),
            )
        )
    return {
        "scope": "fixed_reference_sequences",
        "interpretation": (
            "Descriptive input-shift diagnostics, not a calibrated naturalness score."
            " Hybrid coalitions can change composition even when references"
            " preserve it."
        ),
        "references": rows,
    }
