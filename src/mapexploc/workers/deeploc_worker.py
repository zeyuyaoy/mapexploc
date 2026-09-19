"""Offline native JSON-lines worker. Inspection is separate from lazy loading.

No MAP-ExPLoc imports; this file runs under the pinned licensed native runtime.
Native source files and checkpoints are never modified.
"""

import contextlib
import gc
import hashlib
import importlib.metadata
import os
import pickle
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import json

CLASSES = [
    "Cytoplasm",
    "Nucleus",
    "Extracellular",
    "Cell membrane",
    "Mitochondrion",
    "Plastid",
    "Endoplasmic reticulum",
    "Lysosome/Vacuole",
    "Golgi apparatus",
    "Peroxisome",
]
THRESHOLDS = {
    "fast": [
        0.46953125,
        0.52753906,
        0.64638672,
        0.52368164,
        0.63730469,
        0.65859375,
        0.62783203,
        0.56484375,
        0.66777344,
        0.71679688,
    ],
    "accurate": [
        0.47612305,
        0.50136719,
        0.61728516,
        0.56464844,
        0.62197266,
        0.63945312,
        0.60898438,
        0.58476562,
        0.64941406,
        0.73642578,
    ],
}
PROTT5_ID = "Rostlab/prot_t5_xl_uniref50"
PROTT5_REVISION = "973be27c52ee6474de9c945952a8008aeb2a1a73"
PROTT5_WEIGHTS_SHA256 = (
    "c06141d42e93c12b5f6d95c701952156bd4079661f5f2d981d0d2036ba96dae9"
)
PROTT5_FILES = (
    "config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer_config.json",
)


def checksum(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inspect(configuration: dict[str, Any]) -> dict[str, Any]:
    root = Path(configuration["package_root"]).resolve()
    mode = configuration.get("mode", "fast")
    if mode not in THRESHOLDS:
        raise ValueError("Unsupported DeepLoc mode")
    for name, expected in {
        "model.py": "c24e826542eb2741bb056db23847534354900794672ce390ee383c048846cd40",
        "deeploc2.py": (
            "026fc1c88e70f86fdf74c1870574ceb74832d8d429daf2e320aeab3365622ef5"
        ),
    }.items():
        if checksum(root / "DeepLoc2" / name) != expected:
            raise RuntimeError(
                "Unsupported native DeepLoc source version; refusing to guess output"
                " semantics"
            )
    assets = {}
    for path in sorted((root / "DeepLoc2").rglob("*")):
        # Preserve the historical Fast fingerprint.
        include = (
            "models_prott5" not in str(path)
            if mode == "fast"
            else "models_esm1b" not in str(path)
        )
        if path.suffix in {".py", ".ckpt", ".pkl"} and include:
            assets[str(path.relative_to(root))] = checksum(path)
    if mode == "fast":
        for path in sorted(
            (Path(configuration["torch_home"]) / "hub/checkpoints").glob("esm1b*.pt")
        ):
            assets[path.name] = checksum(path)
        if not any(name.startswith("esm1b_t33") for name in assets):
            raise RuntimeError(
                "Missing offline ESM1b checkpoint; install assets before inference"
            )
    else:
        if configuration.get("batch_size", 1) != 1:
            raise ValueError(
                "Accurate requires batch_size=1; native padding changes localization"
                " pooling"
            )
        snapshot = Path(configuration["prott5_snapshot"])
        if configuration.get("prott5_revision") != PROTT5_REVISION:
            raise ValueError(
                "Unqualified ProtT5 revision; use the pinned upstream release"
            )
        for name in PROTT5_FILES:
            assets[f"{PROTT5_ID}/{PROTT5_REVISION}/{name}"] = checksum(snapshot / name)
        if (
            assets[f"{PROTT5_ID}/{PROTT5_REVISION}/pytorch_model.bin"]
            != PROTT5_WEIGHTS_SHA256
        ):
            raise ValueError(
                "ProtT5 checkpoint checksum does not match the upstream pinned weights"
            )
    identity = hashlib.sha256(json.dumps(assets, sort_keys=True).encode()).hexdigest()
    expected_checkpoint = configuration.get("expected_checkpoint_sha256")
    if expected_checkpoint and expected_checkpoint != identity:
        raise ValueError(
            "DeepLoc asset fingerprint differs from the configured checkpoint"
        )
    embedding = "ESM1b" if mode == "fast" else PROTT5_ID
    return {
        "model_id": f"DeepLoc-2.1-{mode.title()}",
        "classes": CLASSES,
        "task_type": "multilabel",
        "preprocessing_id": (
            "DeepLoc2.1:ESM1b:BatchConverter:native-no-truncation"
            if mode == "fast"
            else (
                "DeepLoc2.1:ProtT5:BatchConverterProtT5:single-sequence:"
                "native-no-truncation"
            )
        ),
        "min_length": 10,
        "max_length": 1022 if mode == "fast" else 4000,
        "thresholds": THRESHOLDS[mode],
        "threshold_comparison": "greater",
        "empty_decision_policy": "nearest_threshold",
        "fallback_round_decimals": 4,
        "capabilities": ["region_kernel"],
        "checkpoint_sha256": identity,
        "provenance": {
            "upstream": "https://services.healthtech.dtu.dk/services/DeepLoc-2.1/",
            "upstream_version": "2.1",
            "mode": mode,
            "embedding_model": embedding,
            "embedding_revision": (
                PROTT5_REVISION if mode == "accurate" else "esm1b_t33_650M_UR50S"
            ),
            "assets": assets,
            "device": configuration.get("device", "cpu"),
            "precision": "float32",
            "threads": configuration.get("threads", 4),
            "native_seed": 42,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "adapter_runtime_sha256": checksum(Path(__file__)),
            "numerical_environment": {
                key: os.environ.get(key)
                for key in (
                    "PYTORCH_MPS_FAST_MATH",
                    "PYTORCH_ENABLE_MPS_FALLBACK",
                    "NVIDIA_TF32_OVERRIDE",
                )
            },
            "batching_policy": (
                "single_sequence"
                if mode == "accurate"
                else f"native_batch_{configuration.get('batch_size', 8)}"
            ),
            "runtime": {
                "python": platform.python_version(),
                **{
                    name: importlib.metadata.version(name)
                    for name in [
                        "torch",
                        "pytorch-lightning",
                        "fair-esm",
                        "numpy",
                        "transformers",
                    ]
                },
            },
            "output_mapping": (
                "Native ensemble sigmoid localization output columns 1:11; internal"
                " column 0 omitted. Full precision; native CSV rounds to four decimals."
            ),
            "decision_policy": (
                "Native strict thresholds; if none selected, final argsort of"
                " four-decimal-rounded probability minus threshold."
            ),
            "license": "User-provided licensed standalone package; not redistributed",
        },
    }


def available_memory() -> int:
    import psutil

    return int(psutil.virtual_memory().available)


def check_resources(
    configuration: dict[str, Any],
    length: int = 0,
    loading: bool = False,
    batch_size: int = 1,
) -> None:
    import torch

    device = configuration.get("device", "cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "Requested MPS device is unavailable; select a qualified device explicitly"
        )
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested CUDA device is unavailable; no device fallback is permitted"
        )
    if device != "cpu" and device != "mps" and not device.startswith("cuda"):
        raise ValueError("Device must be cpu, mps or cuda[:index]")
    accurate = configuration.get("mode", "fast") == "accurate"
    # Conservative memory estimates.
    reserve = 2 * 1024 ** 3
    allocation = (18 if accurate else 6) * 1024 ** 3 if loading else 0
    allocation += batch_size * (32 if accurate else 20) * (length + 2) ** 2 * 4 * 4
    free = available_memory()
    if free < allocation + reserve:
        raise MemoryError(
            f"Insufficient available host memory: {free / 1024 ** 3:.1f} GiB; estimated"
            f" allocation plus reserve {(allocation + reserve) / 1024 ** 3:.1f} GiB."
            " Close other workloads or use a larger explicitly selected device. No"
            " precision fallback."
        )
    if device.startswith("cuda"):
        gpu_free, _ = torch.cuda.mem_get_info(device)
        required = (
            (6 if accurate else 3) * 1024 ** 3
            + allocation
            - ((18 if accurate else 6) * 1024 ** 3 if loading else 0)
        )
        if gpu_free < required:
            raise MemoryError(
                "Insufficient CUDA device memory for the selected native float32"
                " workload"
            )


def load_native(configuration: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    from DeepLoc2.data import BatchConverter, BatchConverterProtT5
    from DeepLoc2.model import ESM1bE2E, ProtT5E2E
    from transformers import T5EncoderModel, T5Tokenizer

    check_resources(configuration, loading=True)
    torch.manual_seed(42)
    torch.set_num_threads(configuration.get("threads", 4))
    if configuration.get("mode", "fast") == "fast":
        model = ESM1bE2E().eval()
        with (
            Path(configuration["package_root"]) / "DeepLoc2/models/ESM1b_alphabet.pkl"
        ).open("rb") as handle:
            converter = BatchConverter(pickle.load(handle))
    else:
        loader = T5EncoderModel.from_pretrained
        snapshot = configuration["prott5_snapshot"]

        def pinned_loader(identifier: str, *args: Any, **kwargs: Any) -> Any:
            if identifier != PROTT5_ID:
                raise ValueError("Unexpected native embedding identifier")
            return loader(snapshot, *args, local_files_only=True, **kwargs)

        with patch.object(T5EncoderModel, "from_pretrained", pinned_loader):
            model = ProtT5E2E().eval()
        tokenizer = T5Tokenizer.from_pretrained(
            snapshot, do_lower_case=False, local_files_only=True
        )
        converter = BatchConverterProtT5(tokenizer)
    return model.to(configuration.get("device", "cpu")), converter


def main() -> None:
    configuration = json.loads(sys.stdin.readline())
    descriptor = inspect(configuration)
    sys.path.insert(0, str(Path(configuration["package_root"]).resolve()))
    os.environ["TORCH_HOME"] = configuration["torch_home"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(configuration["torch_home"]) / "matplotlib")
    )
    print(json.dumps({"descriptor": descriptor}), flush=True)
    model: Any = None
    converter: Any = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("operation") == "unload":
                model = converter = None
                gc.collect()
                if "torch" in sys.modules:
                    import torch

                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                print(json.dumps({"unloaded": True}), flush=True)
                continue
            sequences = request["sequences"]
            if not sequences or any(
                not 10 <= len(s) <= descriptor["max_length"]
                or set(s) - set("ACDEFGHIKLMNPQRSTVWY")
                for s in sequences
            ):
                raise ValueError(
                    "DeepLoc input violates its no-truncation sequence contract"
                )
            outputs = []
            started = time.monotonic()
            cold = model is None
            with contextlib.redirect_stdout(sys.stderr):
                import numpy as np
                import torch

                if model is None:
                    model, converter = load_native(configuration)
                with torch.inference_mode():
                    batch_size = configuration.get("batch_size", 8)
                    for start in range(0, len(sequences), batch_size):
                        batch = sequences[start: start + batch_size]
                        check_resources(
                            configuration,
                            length=max(map(len, batch)),
                            batch_size=len(batch),
                        )
                        toks, lengths, mask, _ = converter(
                            [(s, str(i)) for i, s in enumerate(batch)]
                        )
                        outputs.extend(
                            np.asarray(model(toks, lengths, mask)[0])[:, 1:11].tolist()
                        )
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform != "darwin":
                peak *= 1024
            print(
                json.dumps(
                    {
                        "probabilities": outputs,
                        "measurement": {
                            "elapsed_seconds": time.monotonic() - started,
                            "cold_load": cold,
                            "sequences": len(sequences),
                            "peak_host_bytes": peak,
                            "peak_scope": "worker_lifetime_high_water_mark",
                        },
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
        except Exception as exc:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)
            return  # Never reuse a failed native worker.


if __name__ == "__main__":
    if sys.argv[1:] == ["--resources"]:
        configuration = json.loads(sys.stdin.readline())
        try:
            check_resources(configuration, loading=True)
            print(
                json.dumps(
                    {
                        "ready": True,
                        "available_host_bytes": available_memory(),
                        "qualification": (
                            "Preflight only; native parity remains a separate gate"
                        ),
                    }
                )
            )
        except Exception as exc:
            print(
                json.dumps({"ready": False, "reason": f"{type(exc).__name__}: {exc}"})
            )
    else:
        main()
