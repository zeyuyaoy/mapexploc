import numpy as np
import pytest

from mapexploc.adapter import AdapterDescriptor
from mapexploc.analysis import run_analysis
from mapexploc.annotations import import_uniprot
from mapexploc.explainers.regions import RegionGame, sequence_regions
from mapexploc.report_v2 import AnalysisConfiguration, AnalysisReport, write_report


class PositionModel:
    descriptor = AdapterDescriptor(
        model_id="synthetic-position-control",
        classes=("outside", "inside"),
        preprocessing_id="count-first-ten:v1",
    )

    def predict_proba(self, batch):
        p = np.array([0.1 + 0.8 * s[:10].count("L") / 10 for s in batch])
        return np.column_stack([p, 1 - p])


def test_region_game_exact_additive_and_reference_identity(tmp_path):
    sequence = "LLLLLLLLLLACDEFGHIKMACDEFGHIKM"
    config = AnalysisConfiguration(explainer="region_kernel")
    game = RegionGame(PositionModel(), sequence, config)
    baseline = np.mean(PositionModel().predict_proba(game.references), axis=0)
    np.testing.assert_allclose(game(np.zeros((1, 3)))[0], baseline)
    np.testing.assert_allclose(
        game(np.ones((1, 3))), PositionModel().predict_proba([sequence])
    )
    local = run_analysis(
        PositionModel(), [{"protein_id": "p", "sequence": sequence}], config
    )
    values = np.array(local.results[0].attributions)
    np.testing.assert_allclose(values[:, 0], [0.9, 0.1] - baseline, atol=1e-10)
    np.testing.assert_allclose(values[:, 1:], 0, atol=1e-10)
    assert values[0, 0] > 0 and values[1, 0] < 0
    assert local.results[0].explainer["stability"]["status"] == "exact_enumeration"
    paths = write_report(local, tmp_path)
    assert AnalysisReport.model_validate_json(paths["report.json"].read_text()) == local
    assert "causal" in paths["report.html"].read_text()


@pytest.mark.parametrize("length", [1, 10, 11, 50, 99, 100, 101, 105, 110, 1022])
def test_region_partition(length):
    regions = sequence_regions(length)
    assert regions[0].start == 0 and regions[-1].end == length
    assert sum(r.end - r.start for r in regions) == length
    assert all(a.end == b.start for a, b in zip(regions, regions[1:]))
    assert len(regions) <= 16


def test_cohort_order_invariance_and_no_one_protein_global():
    proteins = [
        {"protein_id": "p1", "sequence": "L" * 10 + "A" * 20},
        {"protein_id": "p2", "sequence": "A" * 10 + "L" * 20},
    ]
    configuration = {
        "cohort_id": "synthetic-paired",
        "selection_criteria": "Two fixed synthetic controls",
    }
    a = run_analysis(PositionModel(), proteins, configuration)
    b = run_analysis(PositionModel(), proteins[::-1], configuration)
    assert a.cohort.contributions == b.cohort.contributions
    assert a.results[0].attributions == b.results[1].attributions
    assert run_analysis(PositionModel(), proteins[:1]).cohort is None
    invalid = a.model_dump()
    invalid["cohort"]["contributions"][0]["mean_signed"] += 1
    with pytest.raises(ValueError, match="Cohort"):
        AnalysisReport.model_validate(invalid)
    invalid = a.model_dump()
    invalid["results"][0]["attributions"][0][0] = float("nan")
    with pytest.raises(ValueError):
        AnalysisReport.model_validate(invalid)


def test_uniprot_indexing_overlap_uncertainty_and_mismatch():
    sequence = "ACDEFGHIKL"
    record = {
        "primaryAccession": "TEST",
        "sequence": {"value": sequence},
        "features": [
            {
                "type": "Signal",
                "location": {"start": {"value": 1}, "end": {"value": 1}},
            },
            {
                "type": "Domain",
                "location": {
                    "start": {"value": 1},
                    "end": {"value": 10, "modifier": "OUTSIDE"},
                },
            },
            {
                "type": "Motif",
                "location": {"start": {"value": 10}, "end": {"value": 10}},
            },
        ],
    }
    kwargs = dict(
        protein_id="p",
        sequence=sequence,
        release="frozen-test",
        retrieved_at="2026-09-18",
        source_url="https://example.org/TEST",
    )
    annotations = import_uniprot(record, **kwargs)
    assert [(a.start, a.end) for a in annotations] == [(0, 1), (0, 10), (9, 10)]
    assert annotations[1].uncertain
    assert not annotations[0].uncertain
    run_analysis(
        PositionModel(),
        [{"protein_id": "p", "sequence": sequence}],
        annotations=annotations,
    )
    with pytest.raises(ValueError, match="exactly match"):
        import_uniprot(record, **{**kwargs, "sequence": sequence[::-1]})
    with pytest.raises(ValueError, match="exact protein"):
        annotations[0].validate_sequence("p", sequence[::-1])
