"""Run the unmodified native CLI and capture its full precision dataframe.

Execute with the native Python. No MAP-ExPLoc inference or decision functions
are imported. The upstream CLI owns thresholds, fallback, ordering and converter.
"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configuration", type=Path, required=True)
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.configuration.read_text())
    sys.path.insert(0, config["package_root"])
    os.environ["TORCH_HOME"] = config["torch_home"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if config.get("mode", "fast") == "accurate":
        snapshot = Path(config["prott5_snapshot"])
        repository = snapshot.parent.parent
        if snapshot.name != config.get(
            "prott5_revision", "973be27c52ee6474de9c945952a8008aeb2a1a73"
        ):
            raise ValueError("Oracle requires a pinned native Hub snapshot layout")
        (repository / "refs").mkdir(exist_ok=True)
        (repository / "refs/main").write_text(snapshot.name)
        os.environ["HF_HUB_CACHE"] = str(repository.parent)
    import psutil
    import torch

    minimum = 20 if config.get("mode", "fast") == "accurate" else 8
    if psutil.virtual_memory().available < minimum * 1024 ** 3:
        raise MemoryError(
            f"Native oracle needs {minimum} GiB available for conservative cold-load"
            " qualification"
        )
    torch.manual_seed(42)
    torch.set_num_threads(config.get("threads", 4))
    import DeepLoc2.deeploc2 as native

    name = (
        "run_model_prott5"
        if config.get("mode", "fast") == "accurate"
        else "run_model_esm1b"
    )
    original = getattr(native, name)
    captured = {}

    def capture(*a, **kw):
        frame = original(*a, **kw)
        captured.update(
            {row.ACC: row.multilabel[0, 1:11].tolist() for row in frame.itertuples()}
        )
        return frame

    setattr(native, name, capture)
    args.output.mkdir(parents=True, exist_ok=True)
    native.main(
        SimpleNamespace(
            fasta=str(args.fasta),
            output=str(args.output),
            model=config.get("mode", "fast").title(),
            device=config.get("device", "cpu"),
            plot=False,
        )
    )
    (args.output / "full-precision.json").write_text(json.dumps(captured, indent=2))


if __name__ == "__main__":
    main()
