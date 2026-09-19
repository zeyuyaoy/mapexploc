"""Real native parity gate. Missing assets or hardware failures are failures."""

import argparse
import time
from pathlib import Path

import json
import numpy as np
import pandas as pd
import subprocess
from Bio import SeqIO

from mapexploc.contracts import class_decisions
from mapexploc.deeploc import DeepLocAdapter
from mapexploc.execution import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.configuration.read_text())
    proteins = list(SeqIO.parse(args.fasta, "fasta"))
    sequences = [str(p.seq) for p in proteins]
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    descriptor = None
    try:
        with DeepLocAdapter.from_config(config) as adapter:
            descriptor = adapter.descriptor
            original = adapter.predict_proba(sequences)
            reversed_batch = adapter.predict_proba(sequences[::-1])[::-1]
            singles = np.concatenate([adapter.predict_proba([s]) for s in sequences])
            decisions = class_decisions(descriptor, original)
            measurements = adapter.measurements
        subprocess.run(
            [
                config["python"],
                str(Path(__file__).with_name("native_deeploc_oracle.py")),
                "--configuration",
                str(args.configuration.resolve()),
                "--fasta",
                str(args.fasta.resolve()),
                "--output",
                str((args.output / "native").resolve()),
            ],
            check=True,
        )
        native = json.loads((args.output / "native/full-precision.json").read_text())
        full = np.array([native[p.id] for p in proteins])
        csv_path = sorted((args.output / "native").glob("results_*.csv"))[-1]
        csv = pd.read_csv(csv_path).set_index("Protein_ID")
        rounded = csv.loc[[p.id for p in proteins], list(descriptor.classes)].to_numpy(
            float
        )
        native_decisions = [
            str(csv.loc[p.id, "Localizations"]).split("|") for p in proteins
        ]
        errors = {
            "full_precision": float(np.max(np.abs(original - full))),
            "native_csv": float(np.max(np.abs(original - rounded))),
            "order": float(np.max(np.abs(original - reversed_batch))),
            "single_batch": float(np.max(np.abs(original - singles))),
        }
        decision_match = all(
            set(a) == set(b) for a, b in zip(decisions, native_decisions)
        )
        if (
            errors["full_precision"] > 1e-5
            or errors["native_csv"] > 5.1e-5
            or errors["order"] > 1e-5
            or errors["single_batch"] > 1e-5
            or not decision_match
        ):
            raise ValueError(
                f"Material native divergence: {errors}; decisions"
                f" agree={decision_match}"
            )
        evidence = dict(
            status="passed",
            model=descriptor.model_dump(mode="json"),
            protein_ids=[p.id for p in proteins],
            probabilities=original.tolist(),
            errors=errors,
            tolerances={"full_precision": 1e-5, "native_csv": 5.1e-5},
            native_decisions_agree=decision_match,
            measurements=measurements,
            elapsed_seconds=time.monotonic() - started,
        )
        atomic_json(args.output / "parity.json", evidence)
        print(json.dumps({k: v for k, v in evidence.items() if k != "model"}, indent=2))
    except Exception as exc:
        atomic_json(
            args.output / "qualification-failure.json",
            dict(
                status="failed",
                mode=config.get("mode", "fast"),
                model=descriptor.model_dump(mode="json") if descriptor else None,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=time.monotonic() - started,
            ),
        )
        raise


if __name__ == "__main__":
    main()
