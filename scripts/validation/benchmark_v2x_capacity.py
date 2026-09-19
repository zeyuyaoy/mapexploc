"""Measured inference capacity on a fixed synthetic panel; not biological evidence."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mapexploc.deeploc import DeepLocAdapter
from mapexploc.execution import atomic_json, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.configuration.read_text())
    mode = config.get("mode", "fast")
    lengths = [10, 100, 250, 512, 1022]
    if mode == "accurate":
        lengths.extend([2048, 4000])
    protocol = {
        "mode": mode,
        "lengths": lengths,
        "cohort_sizes": [1, 4, 16],
        "seed": 20260919,
        "repeats": 3,
        "configuration_sha256": fingerprint(config),
        "interpretation": (
            "Synthetic input capacity, not predictive or biological validation"
        ),
        "peak_device_memory": "not_measured; host high-water mark recorded separately",
    }
    output = {"protocol": protocol, "rows": []}
    if args.output.exists():
        output = json.loads(args.output.read_text())
        if output["protocol"] != protocol:
            raise ValueError("Capacity restart protocol mismatch")
    completed = {(r["length"], r["cohort_size"]) for r in output["rows"]}
    for length in lengths:
        for size in protocol["cohort_sizes"]:
            if (length, size) in completed:
                continue
            rng = np.random.default_rng(protocol["seed"] + length)
            sequences = [
                "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=length))
                for _ in range(size)
            ]
            row = dict(length=length, cohort_size=size)
            try:
                with DeepLocAdapter.from_config(config) as adapter:
                    cold = adapter.predict_proba(sequences)
                    timings = []
                    delta = 0.0
                    for _ in range(protocol["repeats"]):
                        started = time.monotonic()
                        result = adapter.predict_proba(sequences)
                        timings.append(time.monotonic() - started)
                        delta = max(delta, float(np.max(np.abs(result - cold))))
                    if delta > 1e-5:
                        raise ValueError(f"Repeated inference divergence: {delta}")
                    row.update(
                        status="passed",
                        model=adapter.descriptor.model_dump(mode="json"),
                        cold_seconds=adapter.measurements[0]["elapsed_seconds"],
                        warm_seconds=timings,
                        warm_sequences_per_second=size / float(np.median(timings)),
                        max_repeated_probability_delta=delta,
                        measurements=adapter.measurements,
                    )
            except (RuntimeError, MemoryError, TimeoutError, ValueError) as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            output["rows"].append(row)
            atomic_json(args.output, output)
            print(
                json.dumps({k: v for k, v in row.items() if k != "model"}), flush=True
            )
            if row["status"] == "failed" and length == 10:
                raise RuntimeError(
                    "Minimum-length capacity failed; larger trials not launched"
                )


if __name__ == "__main__":
    main()
