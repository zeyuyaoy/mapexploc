"""Check an actual same-origin deployment using only the Python standard library.

Usage: python scripts/smoke_web.py https://your-deployment.vercel.app
The fixed sequence is a software regression probe, not biological validation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from urllib.request import Request, urlopen

SEQUENCE = "MALWMRLLPLLALLALWGPDPAAA"
MODEL_ID = "human-2026_03-56fb89f09b33"
EXPECTED = [
    0.05639880952380953,
    0.15355592757936506,
    0.1687437996031746,
    0.1039287968975469,
    0.517372666396104,
]


def smoke(origin: str, timeout: float = 60) -> None:
    def request(path: str, body: dict | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        bypass = os.environ.get("VERCEL_AUTOMATION_BYPASS_SECRET")
        if bypass:
            headers["x-vercel-protection-bypass"] = bypass
        req = Request(
            f"{origin.rstrip('/')}/api{path}",
            data=json.dumps(body).encode() if body else None,
            headers=headers,
        )
        with urlopen(req, timeout=timeout) as response:
            return json.load(response)

    def check(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(message)

    health = request("/health")
    check(
        health.get("status") == "ready" and health.get("model_available"),
        "Model is not ready",
    )
    model = request("/model")
    check(model["metadata"]["model_id"] == MODEL_ID, "Unexpected model identity")
    check(model["feature_count"] == 423, "Unexpected feature schema")
    catalogue = request("/v3/models")["models"]
    check(
        len(catalogue) == 1 and catalogue[0]["model_id"] == MODEL_ID,
        "Public catalogue differs from the approved release",
    )
    body = {"sequences": [SEQUENCE]}
    prediction = request("/predict", body)["results"][0]
    check(
        model["model_classes"]
        == ["Cytoplasm", "Membrane", "Mitochondrion", "Nucleus", "Secreted"],
        "Unexpected class order",
    )
    check(prediction["prediction"] == "Secreted", "Prediction regression")
    probabilities = [item["probability"] for item in prediction["probabilities"]]
    check(
        len(probabilities) == len(EXPECTED)
        and all(
            math.isclose(actual, expected, abs_tol=1e-8, rel_tol=0)
            for actual, expected in zip(probabilities, EXPECTED)
        ),
        "Prediction probabilities changed",
    )
    explanation = request("/explain", {**body, "top_n": 12})["results"][0]
    reconstructed = (
        explanation["base_value"]
        + explanation["remainder"]
        + sum(item["contribution"] for item in explanation["feature_contributions"])
    )
    check(
        bool(explanation["feature_contributions"])
        and math.isclose(reconstructed, EXPECTED[-1], abs_tol=1e-6, rel_tol=0),
        "Explanation failed probability reconstruction",
    )
    print("PASS: /api health, approved catalogue, model, prediction and explanation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("origin", help="Deployment origin, without /api")
    parser.add_argument("--timeout", type=float, default=60)
    arguments = parser.parse_args()
    smoke(arguments.origin, arguments.timeout)
