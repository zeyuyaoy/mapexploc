import numpy as np
import pytest

from mapexploc import AdapterDescriptor, run_analysis
from mapexploc.benchmark import select_profile
from mapexploc.biological_v2x import evaluate_family, interval_contrast
from mapexploc.comparison import compare_reports


class SignalModel:
    descriptor = AdapterDescriptor(
        model_id="synthetic",
        classes=("Extracellular", "Nucleus"),
        preprocessing_id="test",
    )

    def predict_proba(self, batch):
        p = np.array([0.1 + 0.8 * s[:10].count("L") / 10 for s in batch])
        return np.c_[p, 1 - p]


def test_comparison_class_alignment_and_reference_mismatch():
    r = run_analysis(
        SignalModel(),
        [dict(protein_id="a", sequence="L" * 10 + "A" * 90)],
        {"method_profile": "v2x-legacy"},
    )
    comparison = compare_reports(r, r)
    assert comparison["results"][0]["decision_jaccard"] == 1
    assert (
        comparison["results"][0]["attribution_agreement"][0]["max_absolute_delta"] == 0
    )
    other = r.model_copy(deep=True)
    other.results[0].sequence_sha256 = "0" * 64
    assert len(compare_reports(r, other)["unmatched_left"]) == 1
    other = r.model_copy(deep=True)
    other.results[0].explainer["reference_policy"] = "different-game"
    result = compare_reports(r, other)["results"][0]
    assert result["attribution_comparability"] == "confounded_by_methodology"
    assert result["attribution_agreement"] == []


def test_degenerate_annotation_control_and_resolution():
    proteins = [
        dict(
            protein_id=str(i),
            sequence="L" * 10 + "A" * 90,
            group=str(i),
            split="evaluation",
        )
        for i in range(4)
    ]
    report = run_analysis(SignalModel(), proteins, {"cohort_id": "fixture"})
    assert interval_contrast(report.results[0], 0, [(0, 3)]) is None
    manifest = {
        "members": [
            dict(
                protein_id=r.protein.protein_id,
                sequence_sha256=r.sequence_sha256,
                group=r.protein.group,
                split="evaluation",
                families={"secretion_signal": [dict(start=0, end=20)]},
            )
            for r in report.results
        ]
    }
    result = evaluate_family(report, manifest, "secretion_signal")
    assert result["status"] == "degenerate_matched_null"
    assert result["formal_p"] is None
    assert result["distinct_coordinate_mask_permutations"] == 1
    incomplete = report.model_copy(deep=True)
    incomplete.results.pop()
    with pytest.raises(ValueError, match="Complete frozen evaluation membership"):
        evaluate_family(incomplete, manifest, "secretion_signal")


def test_profile_selection_requires_both_modes():
    with pytest.raises(ValueError, match="Fast and Accurate"):
        select_profile({}, {})
