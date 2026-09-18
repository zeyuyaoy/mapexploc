"""Tests for the biological input and feature schema."""

from pathlib import Path

import pandas as pd
import pytest
from mapexploc.features import FEATURE_NAMES, build_feature_matrix
from mapexploc.preprocessing import _clean_and_primary


def test_feature_schema_is_complete_and_version_stable() -> None:
    frame = build_feature_matrix(["AAAA", "ACDE"])

    assert tuple(frame.columns) == FEATURE_NAMES
    assert frame.shape == (2, 423)
    assert frame.loc["seq_0", "aa_A"] == 1.0
    assert frame.loc["seq_0", "dp_AA"] == 1.0
    assert frame.loc["seq_1", "aa_A"] == 0.25


def test_sequences_are_normalized_without_silent_residue_removal() -> None:
    lower_wrapped = build_feature_matrix("mk tii\nalsy")
    assert lower_wrapped.loc["seq_0", "length"] == 9

    with pytest.raises(ValueError, match="unsupported residues: X"):
        build_feature_matrix(["MKTX"])
    with pytest.raises(ValueError, match="must not be empty"):
        build_feature_matrix(["  \n"])


def test_fasta_annotations_join_by_identifier(tmp_path: Path) -> None:
    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">p2\nACDE\n>p1\nAAAA\n", encoding="utf-8")
    annotations = pd.DataFrame({"sequence_id": ["p1", "p2"], "label": ["one", "two"]})

    frame = build_feature_matrix(fasta, annotations)

    assert list(frame.index) == ["p2", "p1"]
    assert frame.loc["p2", "label"] == "two"
    assert frame.loc["p1", "label"] == "one"


def test_swissprot_location_prefix_and_evidence_are_cleaned() -> None:
    assert (
            _clean_and_primary(["SUBCELLULAR LOCATION: Cytoplasm. {ECO:0000269|PubMed:1}"])
            == "Cytoplasm"
    )
    assert _clean_and_primary(["SUBCELLULAR LOCATION: Cell membrane."]) == "Membrane"
    assert _clean_and_primary(["SUBCELLULAR LOCATION: Nucleus; Cytoplasm."]) == "Other"


def test_explicit_nonexperimental_location_is_not_promoted() -> None:
    assert (
            _clean_and_primary(["SUBCELLULAR LOCATION: Nucleus. {ECO:0000250}"]) == "Other"
    )
