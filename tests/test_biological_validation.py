import numpy as np
import pytest

from mapexploc.annotations import Annotation
from mapexploc.biological_validation import validate_signal_concordance
from mapexploc.contracts import AdapterDescriptor
from mapexploc.explainers.regions import sequence_regions
from mapexploc.provenance import sequence_sha256
from mapexploc.report_v2 import (
    AnalysisConfiguration,
    AnalysisReport,
    LocalExplanation,
    Protein,
    aggregate_cohort,
)


def synthetic_report(variable_endpoints=True):
    results = []
    for i in range(30):
        sequence = "A" * 100
        end = 30 if variable_endpoints and i % 2 else 20
        values = np.zeros(10)
        values[: end // 10] = 0.2
        probability = 0.1 + values.sum()
        annotation = Annotation(
            annotation_id="signal",
            protein_id=str(i),
            sequence_sha256=sequence_sha256(sequence),
            accession="synthetic",
            kind="signal_peptide",
            start=0,
            end=end,
            source="synthetic-software-control",
            source_version="1",
            retrieved_at="2026-09-18",
            source_url="https://example.org/synthetic",
            evidence=[
                {
                    "evidenceCode": "ECO:0000269",
                    "note": "Synthetic evidence tag for software testing only",
                }
            ],
            original_coordinates={"start": 1, "end": end},
        )
        results.append(
            LocalExplanation(
                protein=Protein(protein_id=str(i), sequence=sequence, group=str(i)),
                sequence_sha256=sequence_sha256(sequence),
                probabilities=[probability, 1 - probability],
                decisions=["Extracellular" if probability >= 0.5 else "Inside"],
                explained_classes=["Extracellular", "Inside"],
                features=sequence_regions(100),
                feature_values=[1.0] * 10,
                attributions=[values.tolist(), (-values).tolist()],
                base_values=[0.1, 0.9],
                residuals=[0.0, 0.0],
                annotations=[annotation],
                explainer={
                    "method": "region_kernel",
                    "reference_policy": "synthetic-known-game",
                },
            )
        )
    config = AnalysisConfiguration(cohort_id="synthetic-controls")
    return AnalysisReport(
        model=AdapterDescriptor(
            model_id="synthetic",
            classes=("Extracellular", "Inside"),
            preprocessing_id="synthetic",
        ),
        configuration=config,
        results=results,
        cohort=aggregate_cohort(results, config),
        software={"fixture": "synthetic"},
        created_at="2026-09-18",
    )


def test_known_signal_positive_and_matched_negative_control():
    report = synthetic_report()
    result = validate_signal_concordance(report)
    assert result["effect"] == pytest.approx(0.02)
    assert result["effect_beyond_matched_null"] > 0
    assert result["one_sided_randomization_p"] < 0.05
    assert len(result["randomized_pairing_distribution"]) == 1000
    assert result == validate_signal_concordance(report)


def test_terminal_bias_alone_does_not_pass_negative_control():
    result = validate_signal_concordance(synthetic_report(False))
    assert result["effect"] > 0
    assert result["effect_beyond_matched_null"] == pytest.approx(0)
    assert result["one_sided_randomization_p"] == 1
    assert result["conclusion"].startswith("No support")
