"""Portable scientific analysis schema. All producers validate this same object."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .annotations import Annotation
from .contracts import AdapterDescriptor, class_decisions, validate_probabilities
from .features import normalize_protein_sequence
from .provenance import sequence_sha256


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Protein(Record):
    protein_id: str = Field(min_length=1)
    sequence: str
    group: str | None = None
    split: str | None = None
    evaluation_labels: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize(self) -> Protein:
        self.sequence = normalize_protein_sequence(self.sequence)
        return self


class AnalysisConfiguration(Record):
    explainer: Literal["auto", "tree", "region_kernel"] = "auto"
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    references: int = Field(default=4, ge=1, le=32)
    coalition_budget: int = Field(default=512, ge=32)
    inference_batch_size: int = Field(default=16, ge=1, le=1024)
    stability_checks: bool = False
    stability_tolerance: float = Field(default=0.02, gt=0)
    cohort_id: str | None = None
    selection_criteria: str = "All supplied proteins; no sampling"


class FeatureDefinition(Record):
    feature_id: str
    kind: Literal[
        "engineered_descriptor",
        "sequence_region",
        "residue",
        "token",
        "embedding_dimension",
        "latent_dimension",
    ]
    definition: str
    units: str
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, gt=0)
    category: str | None = None

    @model_validator(mode="after")
    def coordinates(self) -> FeatureDefinition:
        if self.kind == "sequence_region":
            if (
                self.start is None
                or self.end is None
                or self.end <= self.start
                or not self.category
            ):
                raise ValueError(
                    "Sequence regions require valid coordinates and a category"
                )
        elif self.start is not None or self.end is not None:
            raise ValueError("Only sequence-region coordinates are computed by v2")
        if self.kind not in {"engineered_descriptor", "sequence_region"}:
            raise ValueError("This attribution kind is not computed by MAP-ExPLoc v2")
        return self


class LocalExplanation(Record):
    protein: Protein
    sequence_sha256: str
    probabilities: list[float]
    decisions: list[str]
    explained_classes: list[str]
    features: list[FeatureDefinition]
    feature_values: list[float]
    # Full attributions in [class][feature] order.
    attributions: list[list[float]]
    base_values: list[float]
    residuals: list[float]
    annotations: list[Annotation] = Field(default_factory=list)
    explainer: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_local(self) -> LocalExplanation:
        if self.sequence_sha256 != sequence_sha256(self.protein.sequence):
            raise ValueError("Protein sequence checksum mismatch")
        f, c = len(self.features), len(self.explained_classes)
        if not f or not c or len(set(self.explained_classes)) != c:
            raise ValueError("Explanation requires unique classes and features")
        if len({v.feature_id for v in self.features}) != f:
            raise ValueError("Feature identities must be unique")
        values = np.asarray(self.attributions)
        method = self.explainer.get("method")
        if method not in {"tree", "region_kernel"}:
            raise ValueError("Unsupported explanation method")
        expected_kind = (
            "engineered_descriptor" if method == "tree" else "sequence_region"
        )
        if any(feature.kind != expected_kind for feature in self.features):
            raise ValueError("Feature kinds do not match the explanation method")
        if values.shape != (c, f) or len(self.feature_values) != f:
            raise ValueError("Attribution axes must be classes by features")
        if (
            len(self.base_values) != c
            or len(self.residuals) != c
            or len(self.probabilities) != c
        ):
            raise ValueError("Probability/base/residual class axes disagree")
        reconstructed = np.asarray(self.base_values) + values.sum(axis=1)
        if any(value < -1e-8 or value > 1 + 1e-8 for value in self.base_values):
            raise ValueError("Probability-space base values must be probabilities")
        residuals = reconstructed - self.probabilities
        if (
            not np.allclose(residuals, self.residuals, atol=1e-10, rtol=0)
            or np.max(np.abs(residuals)) > 1e-5
        ):
            raise ValueError("SHAP probability reconstruction failed")
        regions = [v for v in self.features if v.kind == "sequence_region"]
        if regions:
            if (
                len(regions) != f
                or regions[0].start != 0
                or regions[-1].end != len(self.protein.sequence)
            ):
                raise ValueError("Regions must cover the entire protein exactly once")
            if any(a.end != b.start for a, b in zip(regions, regions[1:])):
                raise ValueError("Regions overlap or leave gaps")
        for annotation in self.annotations:
            annotation.validate_sequence(self.protein.protein_id, self.protein.sequence)
        if len({a.annotation_id for a in self.annotations}) != len(self.annotations):
            raise ValueError("Annotation IDs must be unique within a protein")
        return self


class GlobalContribution(Record):
    class_id: str
    feature_id: str
    mean_absolute: float = Field(ge=0)
    mean_signed: float
    protein_count: int = Field(ge=1)
    region_count: int = Field(ge=1)
    denominator: str


class CohortSummary(Record):
    cohort_id: str
    selection_criteria: str
    protein_ids: list[str]
    sequence_hashes: list[str]
    input_count: int
    included_count: int
    exclusions: list[dict[str, str]] = Field(default_factory=list)
    contributions: list[GlobalContribution]
    aggregation: str


def aggregate_cohort(
    results: list[LocalExplanation], config: AnalysisConfiguration
) -> CohortSummary | None:
    if len(results) < 2:
        return None
    if not config.cohort_id:
        raise ValueError("A multi-protein analysis requires a declared cohort_id")
    contributions = []
    for class_index, class_id in enumerate(results[0].explained_classes):
        collected: dict[str, list[tuple[float, float, int]]] = {}
        for local in results:
            within: dict[str, list[float]] = {}
            for feature, value in zip(local.features, local.attributions[class_index]):
                key = (
                    feature.feature_id
                    if feature.kind == "engineered_descriptor"
                    else str(feature.category)
                )
                within.setdefault(key, []).append(value)
            for key, values in within.items():
                # Weight proteins equally and report region counts separately.
                collected.setdefault(key, []).append(
                    (
                        float(np.mean(np.abs(values))),
                        float(np.mean(values)),
                        len(values),
                    )
                )
        for key, rows in sorted(collected.items()):
            contributions.append(
                GlobalContribution(
                    class_id=class_id,
                    feature_id=key,
                    mean_absolute=float(np.mean([r[0] for r in rows])),
                    mean_signed=float(np.mean([r[1] for r in rows])),
                    protein_count=len(rows),
                    region_count=sum(r[2] for r in rows),
                    denominator=(
                        "Equal weight per protein with this feature/category; "
                        "within-protein mean across its regions"
                    ),
                )
            )
    return CohortSummary(
        cohort_id=config.cohort_id,
        selection_criteria=config.selection_criteria,
        protein_ids=[r.protein.protein_id for r in results],
        sequence_hashes=[r.sequence_sha256 for r in results],
        input_count=len(results),
        included_count=len(results),
        contributions=contributions,
        aggregation=(
            "Per fixed output class: mean absolute and mean signed "
            "SHAP. Region categories describe terminal/interior "
            "position, not sequence alignment."
        ),
    )


class AnalysisReport(Record):
    schema_version: Literal[2] = 2
    status: Literal["complete"] = "complete"
    model: AdapterDescriptor
    configuration: AnalysisConfiguration
    output_space: Literal["probability"] = "probability"
    attribution_axes: Literal["protein,class,feature"] = "protein,class,feature"
    results: list[LocalExplanation] = Field(min_length=1)
    cohort: CohortSummary | None = None
    software: dict[str, str]
    created_at: str
    warnings: list[str] = Field(default_factory=list)
    interaction_status: Literal["unsupported"] = "unsupported"

    @model_validator(mode="after")
    def validate_report(self) -> AnalysisReport:
        ids = [r.protein.protein_id for r in self.results]
        if len(ids) != len(set(ids)):
            raise ValueError("Protein IDs must be unique")
        for local in self.results:
            if local.explained_classes != list(self.model.classes):
                raise ValueError(
                    "Local class axis differs from declared model class order"
                )
            probabilities = validate_probabilities(self.model, [local.probabilities], 1)
            if local.decisions != class_decisions(self.model, probabilities)[0]:
                raise ValueError(
                    "Decisions do not match the native model decision policy"
                )
            if (
                not self.model.min_length
                <= len(local.protein.sequence)
                <= self.model.max_length
            ):
                raise ValueError("Protein length is outside the adapter contract")
        policies = {
            json.dumps(
                {
                    k: r.explainer.get(k)
                    for k in (
                        "method",
                        "output_space",
                        "reference_policy",
                        "region_policy",
                    )
                },
                sort_keys=True,
            )
            for r in self.results
        }
        if len(policies) != 1:
            raise ValueError("Cannot pool incompatible explanation policies")
        expected = aggregate_cohort(self.results, self.configuration)
        if self.cohort != expected:
            raise ValueError(
                "Cohort summary does not reproduce from the complete local results"
            )
        # Validate nested metadata as strict JSON too.
        json.dumps(self.model_dump(mode="json"), allow_nan=False)
        return self


def write_report(report: AnalysisReport, directory: Path) -> dict[str, Path]:
    """JSON, CSV and HTML are derived from the same fully validated values."""
    from .report_v3 import load_report

    report = load_report(report.model_dump())
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        name: directory / name
        for name in (
            "report.json",
            "predictions.csv",
            "attributions.csv",
            "report.html",
        )
    }
    serialized = report.model_dump_json(indent=2)
    paths["report.json"].write_text(serialized + "\n", encoding="utf-8")
    with paths["predictions.csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["protein_id", "sequence_sha256", "class", "probability", "selected"]
        )
        for local in report.results:
            for label, probability in zip(report.model.classes, local.probabilities):
                writer.writerow(
                    [
                        local.protein.protein_id,
                        local.sequence_sha256,
                        label,
                        probability,
                        label in local.decisions,
                    ]
                )
    with paths["attributions.csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "protein_id",
                "class",
                "feature_id",
                "kind",
                "start_0",
                "end_exclusive",
                "value",
                "shap",
                "base_value",
                "residual",
            ]
        )
        for local in report.results:
            for ci, label in enumerate(local.explained_classes):
                for fi, feature in enumerate(local.features):
                    writer.writerow(
                        [
                            local.protein.protein_id,
                            label,
                            feature.feature_id,
                            feature.kind,
                            feature.start,
                            feature.end,
                            local.feature_values[fi],
                            local.attributions[ci][fi],
                            local.base_values[ci],
                            local.residuals[ci],
                        ]
                    )
    sections = []
    for local in report.results:
        rows = []
        for ci, label in enumerate(local.explained_classes):
            cells = []
            for feature, value in zip(local.features, local.attributions[ci]):
                positions = (
                    ""
                    if feature.start is None
                    else f"{feature.start + 1}–{feature.end}"
                )
                cells.append(
                    f"<tr><td>{html.escape(feature.feature_id)}</td>"
                    f"<td>{positions}</td><td>{value:+.8g}</td></tr>"
                )
            rows.append(
                f"<details><summary>{html.escape(label)}: "
                f"{local.probabilities[ci]:.6g} · base "
                f"{local.base_values[ci]:.6g}</summary><table><tr>"
                "<th>Feature</th><th>Positions (1-based inclusive)</th>"
                f"<th>SHAP</th></tr>{''.join(cells)}</table></details>"
            )
        annotation_items = []
        for annotation in local.annotations:
            uncertain = " (uncertain)" if annotation.uncertain else ""
            annotation_items.append(
                f"<li>{html.escape(annotation.kind)} "
                f"{annotation.start + 1}–{annotation.end}{uncertain} · "
                f"{html.escape(annotation.source)} "
                f"{html.escape(annotation.source_version)} · "
                f"{html.escape(annotation.source_url)}</li>"
            )
        sections.append(
            f"<section><h2>{html.escape(local.protein.protein_id)}</h2>"
            f"<p>{len(local.protein.sequence)} residues · "
            f"{html.escape(', '.join(local.decisions))}</p>"
            f"<code>{html.escape(local.protein.sequence)}</code>{''.join(rows)}"
            "<h3>Biological annotations</h3>"
            f"<ul>{''.join(annotation_items)}</ul>"
            f"<p>{html.escape('; '.join(local.warnings))}</p></section>"
        )
    global_rows = ""
    aggregation = "Local analysis only"
    if report.cohort:
        aggregation = html.escape(report.cohort.aggregation)
        global_rows = "".join(
            f"<tr><td>{html.escape(row.class_id)}</td>"
            f"<td>{html.escape(row.feature_id)}</td>"
            f"<td>{row.mean_absolute:.8g}</td><td>{row.mean_signed:+.8g}</td>"
            f"<td>{row.protein_count}</td></tr>"
            for row in report.cohort.contributions
        )
    methodology_summary = ""
    from .report_v3 import AnalysisReportV3, diagnostic_html

    if isinstance(report, AnalysisReportV3):
        methodology_summary = diagnostic_html(report)
    document = f"""<!doctype html>
<html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width">
<title>MAP-ExPLoc explanation report</title>
<style>
body{{font:16px system-ui;max-width:1100px;margin:2rem auto;
padding:1rem;color:#152f2a}}
table{{border-collapse:collapse;width:100%}}
td,th{{padding:.4rem;text-align:left;border-bottom:1px solid #ddd}}
code,pre{{overflow-wrap:anywhere;white-space:pre-wrap}}
section,details{{margin:1rem 0}}summary{{cursor:pointer}}
h1,h2{{color:#17644e}}
</style><h1>MAP-ExPLoc complete explanation report</h1>
<p>{html.escape(report.model.model_id)} · {html.escape(report.created_at)}</p>
<p>SHAP describes model behavior under the recorded reference policy.
Biological overlap is not proof of causality.
Region values are not individual-residue attributions.</p>
<p>{html.escape('; '.join(report.warnings))}</p>{methodology_summary}{''.join(sections)}
<h2>Cohort summary</h2><p>{aggregation}</p>
<table><tr><th>Class</th><th>Feature/category</th><th>Mean absolute</th>
<th>Mean signed</th><th>Proteins</th></tr>{global_rows}</table>
<details><summary>Complete data and reproducibility metadata</summary>
<pre>{html.escape(serialized)}</pre></details></html>"""
    paths["report.html"].write_text(document + "\n", encoding="utf-8")
    return paths
