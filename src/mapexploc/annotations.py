"""Frozen positional annotations, always bound to an exact protein sequence."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .features import normalize_protein_sequence
from .provenance import sequence_sha256


class Annotation(BaseModel):
    """Zero-based half-open interval; original source coordinates are retained."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    annotation_id: str = Field(min_length=1)
    protein_id: str = Field(min_length=1)
    sequence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    accession: str = Field(min_length=1)
    isoform: str | None = None
    kind: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    source: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    retrieved_at: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    original_coordinates: dict[str, Any]
    uncertain: bool = False
    description: str = ""

    @model_validator(mode="after")
    def interval(self) -> Annotation:
        if self.start >= self.end:
            raise ValueError("Annotation start must precede its exclusive end")
        return self

    def validate_sequence(self, protein_id: str, sequence: str) -> None:
        if self.protein_id != protein_id or self.sequence_sha256 != sequence_sha256(
            sequence
        ):
            raise ValueError(
                "Annotation requires an exact protein ID and sequence match"
            )
        if self.end > len(sequence):
            raise ValueError("Annotation extends beyond the protein sequence")


def import_uniprot(
    record: dict[str, Any],
    *,
    protein_id: str,
    sequence: str,
    release: str,
    retrieved_at: str,
    source_url: str,
) -> list[Annotation]:
    """Import frozen UniProt annotations without fetching or aligning sequences.

    Reject unknown endpoints. Retain approximate numeric endpoints but exclude
    them from exact-coordinate statistics. Ignore unsupported feature types."""
    sequence = normalize_protein_sequence(sequence)
    if record["sequence"]["value"] != sequence:
        raise ValueError("UniProt sequence/isoform does not exactly match input")
    kinds = {
        "Signal": "signal_peptide",
        "Transit peptide": "transit_peptide",
        "Transmembrane": "transmembrane",
        "Motif": "motif",
        "Domain": "domain",
    }
    annotations = []
    for index, feature in enumerate(record.get("features", [])):
        if feature["type"] not in kinds:
            continue
        location = feature["location"]
        if location.get("sequence") not in (None, record["primaryAccession"]):
            raise ValueError(
                "Isoform-specific annotation cannot map to a canonical sequence"
            )
        start, end = location["start"], location["end"]
        if not isinstance(start.get("value"), int) or not isinstance(
            end.get("value"), int
        ):
            raise ValueError(
                "UniProt annotation has an unknown endpoint; curate it before import"
            )
        annotation = Annotation(
            annotation_id=feature.get(
                "featureId", f"{record['primaryAccession']}:{index}"
            ),
            protein_id=protein_id,
            sequence_sha256=sequence_sha256(sequence),
            accession=record["primaryAccession"],
            isoform=location.get("sequence"),
            kind=kinds[feature["type"]],
            start=start["value"] - 1,
            end=end["value"],
            source="UniProt",
            source_version=release,
            retrieved_at=retrieved_at,
            source_url=source_url,
            evidence=feature.get("evidences", []),
            original_coordinates=location,
            uncertain=any(p.get("modifier", "EXACT") != "EXACT" for p in (start, end)),
            description=feature.get("description", ""),
        )
        annotation.validate_sequence(protein_id, sequence)
        annotations.append(annotation)
    return annotations
