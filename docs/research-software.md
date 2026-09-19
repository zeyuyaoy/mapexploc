### Reproduce and extend

Run commands from the repository root. Load only trusted pickle/joblib artifacts. [Methods and evidence](index.md) defines the estimands and qualification limits.

#### Environments and assets

RF/framework reproduction uses Python 3.13.14 and the recorded runtime lock:

```bash
uv venv --python 3.13.14 .venv
uv pip sync environments/mapexploc-runtime.lock --python .venv/bin/python --require-hashes
uv pip install --python .venv/bin/python --no-deps -e .
source .venv/bin/activate
```

DeepLoc requires its licensed standalone package, external weights and a separate worker:

```bash
uv venv --python 3.11.15 artifacts/deeploc-runtime
uv pip sync environments/deeploc-worker-v2x.lock \
  --python artifacts/deeploc-runtime/bin/python --require-hashes
```

Obtain [DeepLoc 2.1](https://services.healthtech.dtu.dk/services/DeepLoc-2.1/). Copy `config/adapters/deeploc.example.json` or `deeploc-accurate.example.json`; set absolute runtime/package/cache paths and device (`cpu`, `mps`, `cuda[:index]`). Retain expected fingerprints: missing or mismatched assets fail without substitution/download.

Fast needs `esm1b_t33_650M_UR50S.pt` and its contact-regression file in `<torch_home>/hub/checkpoints/`. Accurate additionally needs the full float32 ProtT5 checkpoint/tokenizer:

```bash
hf download Rostlab/prot_t5_xl_uniref50 \
  config.json pytorch_model.bin special_tokens_map.json spiece.model tokenizer_config.json \
  --revision 973be27c52ee6474de9c945952a8008aeb2a1a73 --cache-dir artifacts/prott5
```

Set `prott5_snapshot` to the snapshot directory and `batch_size: 1`; the half-precision encoder-only checkpoint is not interchangeable. Asset/runtime identities are recorded in `examples/validation/v2x/{fast-parity,accurate-assets}.json`. Accurate inference remains unqualified.

#### Reports and extension

Generate a complete RF report:

```bash
mapexploc analyze --adapter feature_artifact \
  --adapter-config config/adapters/rf.example.json \
  --fasta src/mapexploc/examples/human_examples.fasta \
  --configuration config/analysis-v2.json --output-dir results/rf
```

For locally configured DeepLoc:

```bash
mapexploc analyze --adapter deeploc2 --adapter-config /path/to/fast.json \
  --model-mode fast --fasta proteins.fasta \
  --configuration config/analysis-v2x.json \
  --cohort-manifest cohort.json --annotations annotations.json \
  --cache-dir results/cache --restart-dir results/fast-restart \
  --output-dir results/fast
```

Repeat the identical invocation after interruption to resume; changed model/runtime, cohort or settings require a new restart directory. Accurate uses its own configuration and `--model-mode accurate`.

Replace placeholder `cohort_id`/`selection_criteria`. A cohort manifest adds `members` containing `protein_id`, `sequence_sha256` and optional `group`, `split`, `evaluation_labels`; membership must exactly match FASTA. Supply biological dependence groups. Labels may stratify evaluation, never determine masks or explained outputs. Annotations are optional; examples are in `examples/validation/v2/`.

No `method_profile` yields historical schema 2; `v2x-legacy` retains that game with schema 3 diagnostics; `v2x-development` permits experimental references/partitions. `config/analysis-v2x.json` enables costly sensitivity/faithfulness runs.

`report.json` is authoritative; HTML/CSV are views. JSON retains sequences, model/reference identities, all contributions, annotations, denominators and diagnostics. Read either schema with `mapexploc.load_report(Path(...))`; obtain contracts from `AnalysisReport.model_json_schema()`, `AnalysisReportV3.model_json_schema()` and `mapexploc.annotations.Annotation`. Completed reports can be imported into the browser without model assets; interactive exports may contain only visited explanations.

```bash
mapexploc compare --left results/fast/report.json \
  --right results/accurate/report.json --output results/comparison.json
```

Comparisons pair protein ID and exact sequence hash and align classes. Attribution correlation requires compatible partitions/reference games; disagreement does not establish biological truth.

External adapters implement `descriptor: AdapterDescriptor` and `predict_proba(sequence_batch)` returning finite `N × C` probabilities in declared class order. Descriptors declare task, classes, input/preprocessing limits, checkpoint identity and decisions. Multiclass rows sum to one; multilabel scores retain thresholds. Register a trusted configuration factory under `mapexploc.adapters` and use `run_analysis(adapter, proteins, configuration, annotations)`. `examples/adapters/sequence_fixture/` is a synthetic contract example.

#### Reproduce the evidence

| Evidence                                                      | Location                                  |
|---------------------------------------------------------------|-------------------------------------------|
| Historical RF, prepared split and predictions                 | `examples/baseline/`, `examples/models/`  |
| Human nested study, audits, folds and final logistic artifact | `examples/experiments/research-revision/` |
| 30-protein signal-peptide experiment                          | `examples/validation/v2/`                 |
| Partial native qualification and frozen continuation cohorts  | `examples/validation/v2x/`                |

Verify study inventories before use:

```python
import hashlib, json
from pathlib import Path

for root in map(Path, ["examples/experiments/research-revision",
                       "examples/validation/v2", "examples/validation/v2x"]):
    for name, expected in json.loads((root / "checksums.json").read_text()).items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
```

Bulk source snapshots, supervision datasets and historical execution snapshots are unavailable. Queries, dates and hashes are recorded; selected positional source records remain. Prepared-data reproduction is possible, but full recuration needs exact source snapshots. Live queries create new cohorts. Compare scientific values within recorded tolerances, not cross-machine pickle bytes; do not resume historical directories with changed code.

```bash
python scripts/ci_verify_assets.py
python scripts/validation/run_v2_cohort.py \
  --inputs examples/validation/v2 --adapter-config /path/to/fast.json \
  --output results/v2-repeat
```

The human nested study requires its separate Python 3.14.6 environment:

```bash
python3.14 -m venv artifacts/research-runtime
source artifacts/research-runtime/bin/activate
python -m pip install -e '.[plots]' \
  -c examples/experiments/research-revision/scientific-environment.txt
python -m mapexploc.research \
  --source examples/experiments/research-revision/dataset.csv \
  --directory results/primary --jobs 2
python scripts/finalize_research_candidate.py \
  --run results/primary --output results/final
```

Repeat into new directories with `--grouping study --single-repeat` or `--cohort note_free --single-repeat` for dependence sensitivities. Finalization removes rejected exact-model calibration. Infer with `mapexploc.research.predict_research(Path("examples/experiments/research-revision/final/model.joblib"), sequences)`; this logistic artifact is not a tree-service model. Class order follows the methods table. Refit the historical RF in its RF environment:

```bash
mapexploc baseline-train --directory examples/baseline \
  --output-model results/rf.joblib --jobs 4
```

Continuation scripts are `scripts/validation/qualify_v2x_mode.py` (native parity), `run_v2x_development.py` (frozen comparison) and `evaluate_v2x_reports.py` (evaluation after freezing method/family eligibility). Use their `--help` with stored manifests; final annotation overlap must not guide selection.

#### Local interface and public deployment

Normal local development uses the RF/framework environment, Node 24 and pnpm 11.18.0:

```bash
python -m uvicorn mapexploc.api:app --host 127.0.0.1 --port 8000
# Another terminal:
pnpm --dir ui install --frozen-lockfile
pnpm --dir ui dev
```

Vite proxies `/api` to port 8000, stripping the prefix. `MAPEXPLOC_API_PROXY` changes that target; `VITE_API_BASE_URL` overrides the browser base at build time. The local API accepts a trusted `MAPEXPLOC_ADAPTER_CATALOG` JSON mapping identifiers to `{"factory":"deeploc2","configuration":{...}}`; `/docs` describes requests and `/v3/models` lists models. Outside a checkout, set `MAPEXPLOC_MODEL_PATH` for the RF.

The public Vercel service uses **Python 3.12**, `app:app` and only the bundled RF approved by `config/default-model.json`. It ignores those research model/adapter overrides. DeepLoc remains local/research-only, subject to upstream licensing; its assets are not hosted. Report import is browser-local and never executes the source model.

`vercel.json` serves Vite and FastAPI at one origin; `/api` routes to the mounted API. `.python-version` and `uv.lock` fix the deployment runtime/dependencies. The bundled manifest/artifact retain checksum and identity validation. To test this factory in an isolated checkout:

```bash
uv sync --frozen --python 3.12 --no-dev
uv run uvicorn app:app --host 127.0.0.1 --port 8000
# Another terminal:
MAPEXPLOC_API_PROXY=http://127.0.0.1:8000/api pnpm --dir ui dev
python scripts/smoke_web.py http://127.0.0.1:8000
```

Import the repository as one Vercel Services project: root `.`, production branch `stable`, Node 24.x, build settings from `vercel.json`. No application secrets are required; leave `VITE_API_BASE_URL` unset. For a linked project:

```bash
npx vercel@59.23.2 deploy
python scripts/smoke_web.py https://YOUR-PREVIEW.vercel.app
npx vercel@59.23.2 promote https://YOUR-PREVIEW.vercel.app
```

Protected previews may require `VERCEL_AUTOMATION_BYPASS_SECRET` in the local smoke process, never a `VITE_` variable. Services is beta; local routing was tested, cloud deployment was not. Estimated Linux dependencies occupy 453 MiB before model/code, near the 500 MB bundle limit; inspect actual cloud build size. Functions allow 60 seconds. Reports use gzip and reject wire payloads above 4 MB; reduce batches or use the CLI if necessary.

| Public operation    | Proteins | Aggregate residues |
|---------------------|---------:|-------------------:|
| Prediction/features |       25 |             50,000 |
| Explanation         |        5 |             10,000 |
| Complete analysis   |       20 |             20,000 |

Requests also cap bodies at 512,000 bytes, annotations at 200 and bootstrap draws at 1,000. Oversized requests return 413. Experimental references/interventions and region KernelSHAP require CLI execution. Offline limits are unchanged. `/api/health` returns HTTP 200 even if unavailable: inspect `model_available`; `/api/model` returns 503 on failure. The smoke script checks availability, model identity, fixed prediction and SHAP reconstruction.

#### Verification

In the Python 3.12 deployment environment, preserve runtime pins when adding test tools:

```bash
source .venv/bin/activate
uv pip install -e '.[dev,docs]' -c environments/mapexploc-runtime.lock
python -m pytest tests/
python scripts/ci_verify_assets.py
python -m mkdocs build --strict
python -m build
python scripts/ci_package_smoke.py dist
pnpm --dir ui check
CI=true pnpm --dir ui test:e2e
```

These checks establish software behavior, not biological validity. Real DeepLoc qualification requires licensed assets, the worker environment and native parity; unavailable assets or rejected resources are not passes.
