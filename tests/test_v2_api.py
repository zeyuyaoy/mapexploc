from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mapexploc import AdapterDescriptor, create_app
from mapexploc.adapter import adapter_from_artifact
from mapexploc.report_v2 import AnalysisReport


class Multi:
    descriptor = AdapterDescriptor(
        model_id="multi",
        classes=("b", "a"),
        preprocessing_id="fixture",
        task_type="multilabel",
        thresholds=(0.6, 0.5),
    )

    def predict_proba(self, batch):
        return np.tile([0.9, 0.8], (len(batch), 1))


def test_v2_multilabel_and_configured_identifiers():
    client = TestClient(create_app(adapters={"external": Multi()}))
    body = {
        "adapter_id": "external",
        "proteins": [{"protein_id": "p", "sequence": "ACDE"}],
    }
    response = client.post("/v2/predict", json=body)
    assert response.status_code == 200
    assert response.json()["results"][0]["decisions"] == ["b", "a"]
    assert response.json()["results"][0]["probabilities"] == [0.9, 0.8]
    assert client.post("/v2/analyze", json=body).status_code == 409
    assert (
        client.post(
            "/v2/predict", json={**body, "adapter_id": "/tmp/arbitrary.py"}
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/v2/predict", json={**body, "adapter_config": {"path": "x"}}
        ).status_code
        == 422
    )


def test_v2_complete_rf_report_and_bounds():
    adapter = adapter_from_artifact(Path("examples/models/human-baseline.joblib"))
    client = TestClient(create_app(adapter))
    body = {"proteins": [{"protein_id": "p", "sequence": "MALWMRLLPLLALLALWGPDPAAA"}]}
    response = client.post("/v2/analyze", json=body)
    assert response.status_code == 200, response.text
    report = AnalysisReport.model_validate(response.json())
    assert len(report.results[0].features) == 423
    assert report.model.checkpoint_sha256
    assert len(report.results[0].attributions) == 5
    served_metadata = client.get("/model").json()["metadata"]
    assert report.model.provenance["limitations"] == served_metadata["limitations"]
    assert report.model.provenance["interpretation_revision"]
    assert "original_limitations" in report.model.provenance
    assert (
        client.post(
            "/v2/analyze",
            json={
                "proteins": [
                    {"protein_id": str(i), "sequence": "ACDE"} for i in range(21)
                ]
            },
        ).status_code
        == 413
    )


@pytest.mark.parametrize(
    "scores,prediction,status",
    [
        ([0.9, 0.1], "b", 200),
        ([0.9, 0.8], "b", 500),
        ([0.1, 0.9], "b", 500),
        ([0.9, 0.1], "unknown", 500),
    ],
)
def test_legacy_api_preserves_class_score_and_rejects_invalid_decisions(
    scores, prediction, status
):
    class Legacy:
        classes_ = ["b", "a"]

        def predict(self, batch):
            return [prediction] * len(batch)

        def predict_proba(self, batch):
            return [scores] * len(batch)

    client = TestClient(create_app(Legacy()))
    response = client.post("/predict", json={"sequences": ["ACDE"]})
    assert response.status_code == status
    if status == 200:
        body = response.json()
        assert body["model_classes"] == ["b", "a"]
        assert body["results"][0]["confidence"] == 0.9
        assert body["results"][0]["prediction"] == "b"
