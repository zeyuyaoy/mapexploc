# First analysis

## Install

Python 3.12+ is supported for source installation. The checked-in scientific artifact should be loaded with the [recorded model versions](model-card.md), or retrained in your environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,docs]"
```

## Start the scientific baseline

From the repository root:

```bash
python -m uvicorn mapexploc.api:app --host 127.0.0.1 --port 8000
```

Repository startup uses `config/default-model.json` to select and checksum-verify the evaluated version 1 model, independently of the working directory. To select another trusted model, set `MAPEXPLOC_MODEL_PATH=/absolute/path/to/model.joblib`on the Python API process. A wheel installation without repository assets requires that override. Invalid overrides fail explicitly; there is no fallback to `model.pkl`.

In another terminal, use Node 24 and pnpm 11.18.0:

```bash
cd ui
pnpm install --frozen-lockfile
pnpm dev
```

Open the URL printed by Vite. Choose **Try an example**, then **Analyze sequences**. Results appear on a separate screen. Switch between Prediction, Sequence and Explanation; use Batch FASTA for several proteins and expand Compare all proteins when needed.

Examples are real held-out human sequences. An example's annotated location is not a guarantee of the model's prediction. Read the [model card](model-card.md) before interpreting results.

## Smoke-test the CLI

This three-row dataset tests integration only and deliberately cannot support cross-validation:

```bash
mapexploc train --config config/default.yml --output-model artifacts/smoke.pkl
mapexploc predict MKTIIALSYIFCLVFADYKDDDDK --model-path artifacts/smoke.pkl
mapexploc explain MKTIIALSYIFCLVFADYKDDDDK --model-path artifacts/smoke.pkl \
  --output-dir artifacts/smoke-explanation
```

Outside a source checkout, small examples are included in the installed package:

```python
from importlib.resources import files

example_root = files("mapexploc").joinpath("examples")
print(example_root.joinpath("smoke.yml"))
print(example_root.joinpath("human_examples.fasta"))
```

Supply the printed configuration path to `mapexploc train --config ...`; omitting `--data-path` uses the packaged smoke CSV directly, regardless of the current directory. The full scientific dataset and model are repository assets, not embedded in the wheel.

Continue with the [Python API](api.md), [HTTP schema](reporting-schema.md), [UI guide](ui.md), or [baseline reproduction](baseline.md).
