"""Exercise the built distribution in a fresh environment outside the checkout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import tomllib
import venv
from pathlib import Path

# These are part of the installed package's documented offline interface.
REQUIRED_EXAMPLES = {
    "ATTRIBUTION.txt",
    "human_examples.fasta",
    "proteins.json",
    "smoke.csv",
    "smoke.yml",
}

INSPECT_INSTALL = """
import importlib.metadata
import json
import sys
from importlib.resources import files
from pathlib import Path
import mapexploc

assert Path(mapexploc.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
root = files('mapexploc').joinpath('examples')
for name in json.loads(sys.argv[1]):
    assert root.joinpath(name).is_file(), f'Missing packaged example: {name}'
print(json.dumps({'version': importlib.metadata.version('mapexploc'),
                  'config': str(root.joinpath('smoke.yml')),
                  'data': str(root.joinpath('smoke.csv'))}))
"""


def distributions(directory: Path, version: str) -> Path:
    wheels = list(directory.glob("*.whl"))
    sdists = list(directory.glob("*.tar.gz"))
    if (
        len(wheels) != 1
        or len(sdists) != 1
        or not wheels[0].name.startswith(f"mapexploc-{version}-")
        or sdists[0].name != f"mapexploc-{version}.tar.gz"
    ):
        raise ValueError("Expected exactly one matching mapexploc wheel and sdist")
    return wheels[0].resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    project = tomllib.loads((repository / "pyproject.toml").read_text())
    version = project["project"]["version"]
    wheel = distributions(args.directory, version)
    expected = set(REQUIRED_EXAMPLES)
    package = repository / "src/mapexploc"
    for pattern in project["tool"]["setuptools"]["package-data"]["mapexploc"]:
        expected.update(p.name for p in package.glob(pattern) if p.is_file())
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    with tempfile.TemporaryDirectory(prefix="mapexploc-wheel-") as temporary:
        work = Path(temporary)
        environment = work / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(environment)
        binaries = environment / ("Scripts" if os.name == "nt" else "bin")
        python = binaries / ("python.exe" if os.name == "nt" else "python")
        cli = binaries / ("mapexploc.exe" if os.name == "nt" else "mapexploc")
        env["PATH"] = str(binaries) + os.pathsep + env.get("PATH", "")
        env["MPLCONFIGDIR"] = str(work / "matplotlib")

        def run(*command: str | Path) -> str:
            print("Running:", " ".join(str(part) for part in command[:3]), flush=True)
            result = subprocess.run(
                [str(part) for part in command],
                cwd=work,
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
            )
            if result.returncode:
                print(result.stdout, flush=True)
                result.check_returncode()
            return result.stdout.strip()

        run(python, "-m", "pip", "install", str(wheel))
        run(python, "-m", "pip", "check")
        info = json.loads(
            run(python, "-I", "-c", INSPECT_INSTALL, json.dumps(sorted(expected)))
        )
        if info["version"] != version or run(cli, "--version") != version:
            raise ValueError("Installed version or console entry point is incorrect")
        model = work / "smoke.joblib"
        run(
            cli,
            "train",
            "--config",
            info["config"],
            "--data-path",
            info["data"],
            "--output-model",
            model,
        )
        if not model.is_file():
            raise ValueError("Training did not produce a model")
        sequence = "MKTIIALSYIFCLVFADYKDDDDK"
        prediction = run(cli, "predict", sequence, "--model-path", model)
        if "Prediction:" not in prediction or "Confidence:" not in prediction:
            raise ValueError("Prediction did not return a classification")
        output = work / "explanation"
        run(cli, "explain", sequence, "--model-path", model, "--output-dir", output)
        explanation = json.loads((output / "explanation.json").read_text())
        if not explanation:
            raise ValueError("Explanation output is empty")
    print("Installed wheel: resources, version, train, predict and explain passed.")


if __name__ == "__main__":
    main()
