"""One public orchestration call for installed pretrained sequence predictors."""

from __future__ import annotations

import platform
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from .adapter import (
    BaseModelAdapter,
    FeatureModelAdapter,
    load_adapter,
    predict_probabilities,
)
from .annotations import Annotation
from .contracts import class_decisions
from .execution import (
    CachedAdapter,
    ExecutionOptions,
    RestartStore,
    explanation_implementation,
)
from .explainers.regions import explain_regions
from .explainers.shap import ShapExplainer
from .features import FEATURE_NAMES
from .methods import MethodConfiguration, parse_configuration
from .provenance import sequence_sha256
from .report_v2 import (
    AnalysisConfiguration,
    AnalysisReport,
    FeatureDefinition,
    LocalExplanation,
    Protein,
    aggregate_cohort,
)


def engineered_definitions() -> list[FeatureDefinition]:
    definitions = []
    for name in FEATURE_NAMES:
        units = "fraction"
        if name == "length":
            description, units = (
                "Number of amino acids in the whole sequence",
                "residues",
            )
        elif name.startswith("aa_"):
            description = (
                f"Whole-sequence frequency of {name[3:]}: "
                "count divided by sequence length"
            )
        elif name.startswith("dp_"):
            description = (
                f"Whole-sequence frequency of overlapping {name[3:]} pairs: "
                "count divided by max(length - 1, 1); "
                "not a particular occurrence"
            )
        elif name == "gravy":
            description, units = (
                (
                    "Mean Kyte-Doolittle hydropathy across the complete "
                    "sequence (Biopython ProteinAnalysis)"
                ),
                "hydropathy index",
            )
        else:
            description, units = (
                (
                    "Theoretical isoelectric point (Biopython "
                    "ProteinAnalysis/IsoelectricPoint)"
                ),
                "pH",
            )
        definitions.append(
            FeatureDefinition(
                feature_id=name,
                kind="engineered_descriptor",
                definition=description,
                units=units,
            )
        )
    return definitions


def runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in (
        "mapexploc",
        "numpy",
        "scipy",
        "pandas",
        "shap",
        "scikit-learn",
        "pydantic",
        "biopython",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed-source-checkout"
    return versions


def run_analysis(
    adapter: BaseModelAdapter,
    proteins: Sequence[Protein | dict[str, Any]],
    configuration: AnalysisConfiguration | dict[str, Any] | None = None,
    annotations: Sequence[Annotation | dict[str, Any]] = (),
    *,
    execution: ExecutionOptions | None = None,
) -> AnalysisReport:
    """Explain every supplied protein and every native class; never silently sample.

    A multi-protein run declares its cohort identity and selection in configuration.
    Positional annotations and evaluation labels never enter model inference or
    mask construction. Installed adapters need only the public adapter contract.
    """
    started = time.monotonic()
    adapter = load_adapter(adapter)
    config = (
        configuration
        if isinstance(configuration, AnalysisConfiguration)
        else parse_configuration(configuration or {})
    )
    records = [
        p if isinstance(p, Protein) else Protein.model_validate(p) for p in proteins
    ]
    if not records or len({p.protein_id for p in records}) != len(records):
        raise ValueError("Supply at least one protein with unique identifiers")
    if len(records) > 1 and not config.cohort_id:
        raise ValueError(
            "Declare cohort_id and selection_criteria for a multi-protein analysis"
        )
    intervals = [
        a if isinstance(a, Annotation) else Annotation.model_validate(a)
        for a in annotations
    ]
    by_id = {p.protein_id: p for p in records}
    for annotation in intervals:
        if annotation.protein_id not in by_id:
            raise ValueError("Annotation refers to an unknown protein")
        annotation.validate_sequence(
            annotation.protein_id, by_id[annotation.protein_id].sequence
        )
    sequences = [p.sequence for p in records]
    prediction_adapter = adapter
    if execution and execution.cache_directory:
        prediction_adapter = CachedAdapter(
            adapter, execution.cache_directory, execution.max_cache_entries
        )
    restart = (
        RestartStore(
            execution.restart_directory,
            dict(
                model=adapter.descriptor.model_dump(mode="json"),
                configuration=config.model_dump(mode="json"),
                implementation_sha256=explanation_implementation(),
                proteins=[p.model_dump(mode="json") for p in records],
            ),
        )
        if execution and execution.restart_directory
        else None
    )
    measurement_start = len(getattr(adapter, "measurements", []))
    probabilities = predict_probabilities(prediction_adapter, sequences)
    decisions = class_decisions(adapter.descriptor, probabilities)
    method = config.explainer
    if method == "auto":
        method = (
            "tree" if "tree" in adapter.descriptor.capabilities else "region_kernel"
        )
    if method not in adapter.descriptor.capabilities:
        raise ValueError(f"Adapter does not support {method}")
    tree: dict[str, Any] | None = None
    features = None
    if method == "tree":
        if not isinstance(adapter, FeatureModelAdapter):
            raise TypeError(
                "The verified tree route requires FeatureModelAdapter; use "
                "region_kernel for other adapters"
            )
        features = adapter.prepare(sequences)
        tree = ShapExplainer(adapter.model).explain_sample(
            features, sample_size=len(records)
        )
        if list(map(str, tree["classes"])) != list(adapter.descriptor.classes):
            raise ValueError("Tree and adapter class order disagree")
        if not np.allclose(
            np.asarray(tree["expected_value"]) + tree["shap_values"].sum(axis=2),
            probabilities,
            atol=1e-6,
            rtol=0,
        ):
            raise ValueError("Tree SHAP differs from the complete served adapter")
    results = []
    for index, protein in enumerate(records):
        restored = restart.load(protein.protein_id) if restart else None
        if restored is not None:
            data = restored
        elif tree is not None and features is not None:
            data = {
                "features": engineered_definitions(),
                "feature_values": features.iloc[index].tolist(),
                "attributions": tree["shap_values"][index].tolist(),
                "base_values": tree["expected_value"].tolist(),
                "residuals": tree["residuals"][index].tolist(),
                "explainer": {
                    "method": "tree",
                    "output_space": "probability",
                    "reference_policy": "fitted weighted tree-path training counts",
                    "feature_perturbation": "tree_path_dependent",
                    "transformed_feature_names": tree["transformed_feature_names"],
                    "stability": {"status": "exact_tree_algorithm"},
                },
                "warnings": [
                    "Engineered descriptors describe the whole sequence; "
                    "contributions cannot be assigned to individual residues or"
                    " dipeptide occurrences."
                ],
            }
        else:
            data = explain_regions(prediction_adapter, protein.sequence, config)
        # record residuals against the original served batch, including harmless
        # floating-point differences between native batch shapes
        data["residuals"] = (
            np.asarray(data["base_values"])
            + np.asarray(data["attributions"]).sum(axis=1)
            - probabilities[index]
        ).tolist()
        results.append(
            LocalExplanation(
                protein=protein,
                sequence_sha256=sequence_sha256(protein.sequence),
                probabilities=probabilities[index].tolist(),
                decisions=decisions[index],
                explained_classes=list(adapter.descriptor.classes),
                annotations=[
                    a for a in intervals if a.protein_id == protein.protein_id
                ],
                **data,
            )
        )
        if restart and restored is None:
            serialized = results[-1].model_dump(mode="json")
            restart.save(protein.protein_id, {k: serialized[k] for k in data})
        if execution and execution.progress:
            execution.progress(
                dict(
                    completed=index + 1,
                    total=len(records),
                    protein_id=protein.protein_id,
                    resumed=restored is not None,
                    elapsed_seconds=time.monotonic() - started,
                )
            )
    warnings = [
        "Biological annotation overlap describes model behavior, "
        "not proof of a causal biological mechanism."
    ]
    if len({p.sequence for p in records}) != len(records):
        warnings.append(
            "Cohort contains duplicate sequences; aggregate counts are "
            "proteins, not independent sequences."
        )
    if adapter.descriptor.checkpoint_sha256 is None:
        warnings.append(
            "Adapter did not provide a checkpoint fingerprint; exact "
            "model reproduction is not established."
        )
    payload: dict[str, Any] = dict(
        model=adapter.descriptor,
        configuration=config,
        results=results,
        cohort=aggregate_cohort(results, config),
        software=runtime_versions(),
        created_at=datetime.now(timezone.utc).isoformat(),
        warnings=warnings,
    )
    if isinstance(config, MethodConfiguration):
        from .report_v3 import (
            AnalysisReportV3,
            GlobalStatistic,
            ProteinDiagnostics,
            RuntimeIdentity,
            RuntimeMeasurements,
            StructuredWarning,
            methodology_hash,
        )
        from .statistics import group_summaries

        provenance = adapter.descriptor.provenance
        measurements = getattr(adapter, "measurements", [])[measurement_start:]
        return AnalysisReportV3(
            **payload,
            methodology_sha256=methodology_hash(config),
            qualification=(
                "legacy_method"
                if config.method_profile == "v2x-legacy"
                else "development_only"
            ),
            runtime=RuntimeIdentity(
                model_mode=provenance.get("mode"),
                embedding_model=provenance.get("embedding_model"),
                embedding_revision=provenance.get("embedding_revision"),
                checkpoint_sha256=adapter.descriptor.checkpoint_sha256,
                preprocessing_id=adapter.descriptor.preprocessing_id,
                device=provenance.get("device", "adapter-defined"),
                precision=provenance.get("precision", "adapter-defined"),
                batching_policy=provenance.get("batching_policy", "adapter-defined"),
                software={
                    **runtime_versions(),
                    "explanation_engine_sha256": explanation_implementation(),
                    **{
                        key: value
                        for key, value in provenance.get("runtime", {}).items()
                        if isinstance(value, str)
                    },
                },
            ),
            measurements=RuntimeMeasurements(
                elapsed_seconds=time.monotonic() - started,
                native_calls=len(measurements),
                native_sequences=sum(m.get("sequences", 0) for m in measurements),
                peak_host_bytes=max(
                    (m.get("peak_host_bytes", 0) for m in measurements), default=0
                )
                or None,
                peak_scope=(
                    "worker_lifetime_high_water_mark"
                    if measurements
                    else "not_measured"
                ),
                cache_hits=getattr(prediction_adapter, "cache_hits", 0),
            ),
            diagnostics=[
                ProteinDiagnostics(
                    protein_id=r.protein.protein_id,
                    **r.explainer.get("diagnostics", {}),
                )
                for r in results
            ],
            global_statistics=[
                GlobalStatistic.model_validate(row)
                for row in group_summaries(
                    results, config.diagnostic_seed, config.bootstrap_replicates
                )
            ],
            structured_warnings=[
                StructuredWarning(
                    code="historical_biological_evidence",
                    message=(
                        "Historical 30-protein signal-peptide result: matched-null"
                        " p=0.077922; one unstable sensitivity check. No stronger"
                        " biological conclusion is established by software validation."
                    ),
                )
            ]
            + (
                [
                    StructuredWarning(
                        code="development_method",
                        message=(
                            "This method profile is experimental; no replacement"
                            " default has been qualified."
                        ),
                    )
                ]
                if config.method_profile == "v2x-development"
                else []
            )
            + (
                [
                    StructuredWarning(
                        code="small_cohort",
                        message=(
                            "Fewer than 20 independent groups; global summaries are"
                            " exploratory."
                        ),
                    )
                ]
                if len({p.group or sequence_sha256(p.sequence) for p in records}) < 20
                else []
            ),
        )
    return AnalysisReport(**payload)
