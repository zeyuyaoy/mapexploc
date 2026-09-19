"""Backward-compatible report entry points and explicit v2.x scientific metadata."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from .methods import MethodConfiguration
from .report_v2 import AnalysisReport, Record
from .statistics import group_summaries


class StructuredWarning(Record):
    code: str
    message: str
    scope: str = "run"
    severity: Literal["information", "warning"] = "warning"


class RuntimeIdentity(Record):
    model_mode: str | None = None
    embedding_model: str | None = None
    embedding_revision: str | None = None
    checkpoint_sha256: str | None = None
    preprocessing_id: str
    device: str = "adapter-defined"
    precision: str = "adapter-defined"
    batching_policy: str = "adapter-defined"
    software: dict[str, str] = Field(default_factory=dict)


class RuntimeMeasurements(Record):
    elapsed_seconds: float = Field(ge=0)
    native_calls: int = Field(default=0, ge=0)
    native_sequences: int = Field(default=0, ge=0)
    peak_host_bytes: int | None = Field(default=None, ge=0)
    peak_scope: str = "not_measured"
    cache_hits: int = Field(default=0, ge=0)


class GlobalStatistic(Record):
    class_id: str
    feature_id: str
    statistic: Literal["signed", "absolute", "signed_density", "absolute_density"]
    mean: float
    median: float
    trimmed_mean: float
    interval_95: tuple[float, float]
    group_count: int = Field(ge=1)
    protein_count: int = Field(ge=1)
    bootstrap_replicates: int = Field(ge=100)
    denominator: str
    status: Literal["exploratory", "exploratory_small_sample"]


class DiagnosticMetric(Record):
    comparison: str
    class_id: str
    status: Literal["stable", "unstable", "uninformative", "methodological_sensitivity"]
    max_absolute_delta: float = Field(ge=0)
    magnitude_spearman: float | None = Field(default=None, ge=-1, le=1)
    sign_agreement: float | None = Field(default=None, ge=0, le=1)
    top_region_overlap: float | None = Field(default=None, ge=0, le=1)


class ProteinDiagnostics(Record):
    protein_id: str
    stability: list[DiagnosticMetric] = Field(default_factory=list)
    faithfulness: list[dict[str, Any]] = Field(default_factory=list)
    distribution_shift: dict[str, Any] = Field(default_factory=dict)
    supervision_overlap: dict[str, str] = Field(
        default_factory=lambda: {
            "status": "unknown",
            "foundation_pretraining": "unknown",
        }
    )


class AnalysisReportV3(AnalysisReport):
    schema_version: Literal[3] = 3  # type: ignore[assignment]
    configuration: MethodConfiguration
    methodology_sha256: str
    qualification: Literal["legacy_method", "development_only"]
    runtime: RuntimeIdentity
    measurements: RuntimeMeasurements
    diagnostics: list[ProteinDiagnostics]
    global_statistics: list[GlobalStatistic] = Field(default_factory=list)
    structured_warnings: list[StructuredWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_v3(self) -> "AnalysisReportV3":
        expected = methodology_hash(self.configuration)
        if self.methodology_sha256 != expected:
            raise ValueError("Methodology fingerprint mismatch")
        expected_qualification = (
            "legacy_method"
            if self.configuration.method_profile == "v2x-legacy"
            else "development_only"
        )
        if self.qualification != expected_qualification:
            raise ValueError("Unqualified method cannot claim release qualification")
        if (
            self.runtime.checkpoint_sha256 != self.model.checkpoint_sha256
            or self.runtime.preprocessing_id != self.model.preprocessing_id
        ):
            raise ValueError("Runtime identity disagrees with model identity")
        if [d.protein_id for d in self.diagnostics] != [
            r.protein.protein_id for r in self.results
        ]:
            raise ValueError("Diagnostics must cover each protein in report order")
        for diagnostic, local in zip(self.diagnostics, self.results):
            if any(
                row.class_id not in self.model.classes for row in diagnostic.stability
            ):
                raise ValueError("Diagnostic class is not declared by the model")
            expected_diagnostic = ProteinDiagnostics(
                protein_id=local.protein.protein_id,
                **local.explainer.get("diagnostics", {}),
            )
            if diagnostic != expected_diagnostic:
                raise ValueError(
                    "Report diagnostics disagree with local explanation evidence"
                )
        for key, runtime_field in (
            ("mode", "model_mode"),
            ("embedding_model", "embedding_model"),
            ("embedding_revision", "embedding_revision"),
        ):
            if self.model.provenance.get(key) != getattr(self.runtime, runtime_field):
                raise ValueError(
                    "Runtime representation identity disagrees with model provenance"
                )
        expected_rows = group_summaries(
            self.results,
            self.configuration.diagnostic_seed,
            self.configuration.bootstrap_replicates,
        )
        if self.global_statistics != [
            GlobalStatistic.model_validate(r) for r in expected_rows
        ]:
            raise ValueError("Group statistics do not reproduce from local results")
        return self


def methodology_hash(configuration: MethodConfiguration) -> str:
    # Cohort identity and batching do not define the explanation game.
    parameters = configuration.model_dump(
        exclude={"cohort_id", "selection_criteria", "inference_batch_size"}
    )
    return hashlib.sha256(
        json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_report(
    value: Path | str | bytes | dict[str, Any],
) -> AnalysisReport | AnalysisReportV3:
    """Read either schema without modifying the source file or upgrading its data."""
    if isinstance(value, Path):
        value = value.read_bytes()
    data = json.loads(value) if isinstance(value, (str, bytes)) else value
    version = data.get("schema_version")
    if version not in {2, 3}:
        raise ValueError(f"Unsupported report schema {version!r}")
    return (AnalysisReport if version == 2 else AnalysisReportV3).model_validate(data)


def export_original_report(source: Path, destination: Path) -> None:
    """Validate then preserve byte identity, including legacy JSON formatting."""
    original = source.read_bytes()
    load_report(original)
    destination.write_bytes(original)


def diagnostic_html(report: AnalysisReportV3) -> str:
    """Human-readable diagnostics; the full signed values remain in the main report."""
    esc = html.escape
    warnings = "".join(
        f"<li><strong>{esc(w.code)}</strong>: {esc(w.message)}</li>"
        for w in report.structured_warnings
    )
    rows = "".join(
        f"<tr><td>{esc(d.protein_id)}</td><td>{esc(s.class_id)}</td>"
        f"<td>{esc(s.comparison)}</td><td>{esc(s.status)}</td>"
        f"<td>{s.max_absolute_delta:.6g}</td></tr>"
        for d in report.diagnostics
        for s in d.stability
    )
    interventions = "".join(
        f"<details><summary>{esc(d.protein_id)}: "
        "held-out perturbation effects</summary>"
        f"<pre>{esc(json.dumps(d.faithfulness, indent=2))}</pre></details>"
        for d in report.diagnostics
        if d.faithfulness
    )
    statistics = "".join(
        f"<tr><td>{esc(s.class_id)}</td><td>{esc(s.feature_id)}</td>"
        f"<td>{esc(s.statistic)}</td><td>{s.mean:.6g}</td><td>{s.median:.6g}</td>"
        f"<td>{s.trimmed_mean:.6g}</td>"
        f"<td>[{s.interval_95[0]:.6g}, {s.interval_95[1]:.6g}]</td>"
        f"<td>{s.group_count}</td></tr>"
        for s in report.global_statistics
    )
    return (
        "<section><h2>Methodology and uncertainty</h2>"
        f"<p>Profile: {esc(report.configuration.method_profile)}; "
        f"mode: {esc(report.runtime.model_mode or 'adapter-defined')}; "
        f"device: {esc(report.runtime.device)}; "
        f"precision: {esc(report.runtime.precision)}.</p>"
        "<p>Prediction scores, attribution magnitude, stability, biological overlap "
        "and perturbation effects describe different quantities.</p>"
        f"<ul>{warnings}</ul><h3>Attribution stability</h3>"
        "<table><tr><th>Protein</th><th>Class</th><th>Comparison</th><th>Status</th>"
        f"<th>Maximum absolute change</th></tr>{rows}</table>{interventions}"
        "<details><summary>Equal-group exploratory statistics</summary>"
        "<table><tr><th>Class</th><th>Feature/category</th><th>Statistic</th>"
        "<th>Mean</th><th>Median</th><th>10% trimmed mean</th>"
        f"<th>95% group-bootstrap interval</th><th>Groups</th></tr>{statistics}</table>"
        "</details></section>"
    )
