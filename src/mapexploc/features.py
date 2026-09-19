"""Deterministic feature extraction for protein sequences."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MAX_SEQUENCE_LENGTH = 100_000
DIPEPTIDES = tuple(a + b for a in AMINO_ACIDS for b in AMINO_ACIDS)
FEATURE_NAMES = (
    "length",
    *(f"aa_{aa}" for aa in AMINO_ACIDS),
    *(f"dp_{pair}" for pair in DIPEPTIDES),
    "gravy",
    "isoelectric_point",
)
FASTA_SUFFIXES = {".fa", ".faa", ".fasta", ".fna"}


def normalize_protein_sequence(sequence: str) -> str:
    """Normalize whitespace and case; reject empty or ambiguous sequences."""

    if not isinstance(sequence, str):
        raise TypeError("Protein sequences must be strings")
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError("Protein sequence must not be empty")
    if len(normalized) > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"Protein sequence exceeds the {MAX_SEQUENCE_LENGTH:,}-residue limit"
        )
    invalid = sorted(set(normalized).difference(AMINO_ACIDS))
    if invalid:
        residues = ", ".join(invalid)
        raise ValueError(f"Protein sequence contains unsupported residues: {residues}")
    return normalized


def _read_sequences(source: str | Path) -> tuple[list[str], list[str], bool]:
    path = Path(source)
    try:
        exists = path.exists()
    except (OSError, ValueError):
        exists = False
    if exists:
        if not path.is_file():
            raise ValueError(f"Sequence source is not a file: {path}")
        if path.suffix.lower() not in FASTA_SUFFIXES:
            raise ValueError(
                f"Unsupported sequence file format: {path.suffix or '<none>'}"
            )
        records = list(SeqIO.parse(path, "fasta"))  # type: ignore[no-untyped-call]
        if not records:
            raise ValueError(f"No FASTA records found in {path}")
        ids = [record.id for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("FASTA record identifiers must be unique")
        return [str(record.seq) for record in records], ids, True

    if isinstance(source, Path) or path.suffix.lower() in FASTA_SUFFIXES:
        raise FileNotFoundError(f"Sequence file not found: {path}")
    return [str(source)], ["seq_0"], False


def build_feature_matrix(
    sequences: str | Path | Sequence[str] | pd.Series,
    annotations: str | Path | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build 423 features from a sequence, FASTA path or sequence batch.

    Join FASTA metadata by a recognized identifier column, or positionally
    when row counts match and no identifier column is present."""

    from_fasta = False
    if isinstance(sequences, (str, Path)):
        seq_list, ids, from_fasta = _read_sequences(sequences)
    elif isinstance(sequences, (list, tuple, pd.Series, np.ndarray)):
        seq_list = [str(sequence) for sequence in sequences]
        ids = [f"seq_{index}" for index in range(len(seq_list))]
        if not seq_list:
            raise ValueError("At least one protein sequence is required")
    else:
        raise TypeError(f"Unsupported sequence source: {type(sequences).__name__}")

    normalized = [normalize_protein_sequence(sequence) for sequence in seq_list]
    frame = pd.DataFrame(
        [_extract_features(sequence) for sequence in normalized],
        columns=FEATURE_NAMES,
        index=ids,
    )
    frame.index.name = "sequence_id"

    if annotations is None:
        return frame
    annotation_frame = (
        pd.read_csv(annotations)
        if isinstance(annotations, (str, Path))
        else annotations.copy() if isinstance(annotations, pd.DataFrame) else None
    )
    if annotation_frame is None:
        raise TypeError(f"Unsupported annotations source: {type(annotations).__name__}")

    id_column = next(
        (
            column
            for column in ("sequence_id", "entry_name", "accession", "id")
            if column in annotation_frame.columns
        ),
        None,
    )
    if from_fasta and id_column is not None:
        if annotation_frame[id_column].duplicated().any():
            raise ValueError(f"Annotation identifiers in '{id_column}' must be unique")
        indexed = annotation_frame.set_index(id_column)
        missing = frame.index.difference(indexed.index)
        if len(missing):
            raise ValueError(
                "Annotations are missing FASTA identifiers: " + ", ".join(missing[:5])
            )
        return frame.join(indexed, how="left")

    if len(annotation_frame) != len(frame):
        raise ValueError(
            "Annotations must have the same row count as sequences when no matching "
            "identifier column is available"
        )
    return pd.concat(
        [frame.reset_index(drop=False), annotation_frame.reset_index(drop=True)], axis=1
    ).set_index("sequence_id")


def _extract_features(sequence: str) -> dict[str, Any]:
    """Extract amino-acid, dipeptide, and physicochemical features."""

    sequence = normalize_protein_sequence(sequence)
    length = len(sequence)
    residue_counts = Counter(sequence)
    pair_counts = Counter(sequence[index: index + 2] for index in range(length - 1))
    pair_total = max(length - 1, 1)
    analyser = ProteinAnalysis(sequence)  # type: ignore[no-untyped-call]

    features: dict[str, Any] = {"length": length}
    features.update(
        {f"aa_{aa}": residue_counts.get(aa, 0) / length for aa in AMINO_ACIDS}
    )
    features.update(
        {f"dp_{pair}": pair_counts.get(pair, 0) / pair_total for pair in DIPEPTIDES}
    )
    features["gravy"] = float(analyser.gravy())  # type: ignore[no-untyped-call]
    features["isoelectric_point"] = float(
        analyser.isoelectric_point()  # type: ignore[no-untyped-call]
    )
    return features


__all__ = [
    "AMINO_ACIDS",
    "FEATURE_NAMES",
    "MAX_SEQUENCE_LENGTH",
    "build_feature_matrix",
    "normalize_protein_sequence",
]
