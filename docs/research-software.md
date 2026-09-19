### Reproduce and extend

Run from the repository root unless noted. Scientific interpretation and qualification limits are in [Methods and evidence](index.md). Model files are pickle-based: load only trusted artifacts.

#### Environments and assets

Use Python **3.13.14** for the RF/framework environment; the lock records all runtime versions:

```bash
uv venv --python 3.13.14 .venv
uv pip sync environments/mapexploc-runtime.lock --python .venv/bin/python --require-hashes
uv pip install --python .venv/bin/python --no-deps -e .
source .venv/bin/activate
```

DeepLoc runs in a separate Python **3.11.15** environment:

```bash
uv venv --python 3.11.15 artifacts/deeploc-runtime
uv pip sync environments/deeploc-worker-v2x.lock \
  --python artifacts/deeploc-runtime/bin/python --require-hashes
```

Obtain the licensed [DeepLoc 2.1 standalone package](https://services.healthtech.dtu.dk/services/DeepLoc-2.1/) and its pretrained assets. Copy `config/adapters/deeploc.example.json` or `deeploc-accurate.example.json` to an operator configuration and set absolute runtime/package/cache paths and an explicit device (`cpu`, `mps`, `cuda[:index]`). Keep the expected checkpoint fingerprint. Source, tokenizer and weight hashes are checked; missing offline assets fail rather than download or substitute a model.

Fast requires `esm1b_t33_650M_UR50S.pt` and its contact-regression file under `<torch_home>/hub/checkpoints/`. Accurate additionally requires the pinned full ProtT5 checkpoint and tokenizer:

```bash
hf download Rostlab/prot_t5_xl_uniref50 \
  config.json pytorch_model.bin special_tokens_map.json spiece.model tokenizer_config.json \
  --revision 973be27c52ee6474de9c945952a8008aeb2a1a73 \
  --cache-dir artifacts/prott5
```

Set `prott5_snapshot` to that snapshot directory and `batch_size: 1`. Do not substitute the half-precision encoder-only checkpoint. Loading uses float32 with no silent device/precision fallback. Asset hashes and tested runtime metadata are in `examples/validation/v2x/{fast-parity,accurate-assets}.json`; asset availability does not establish Accurate qualification.

#### Generate and inspect a report

For the served RF and packaged human examples:

```bash
mapexploc analyze --adapter feature_artifact \
  --adapter-config config/adapters/rf.example.json \
  --fasta src/mapexploc/examples/human_examples.fasta \
  --configuration config/analysis-v2.json --output-dir results/rf
```

For an external sequence model:

```bash
mapexploc analyze --adapter deeploc2 --adapter-config /path/to/fast.json \
  --model-mode fast --fasta proteins.fasta \
  --configuration config/analysis-v2x.json \
  --cohort-manifest cohort.json --annotations annotations.json \
  --cache-dir results/cache --restart-dir results/fast-restart \
  --output-dir results/fast
```

Use the Accurate configuration with `--model-mode accurate` only within the documented qualification limits. Expensive explanations are CLI-first. Ctrl-C preserves completed proteins; repeat the identical invocation to resume. Changed model/runtime, membership or analysis settings require a new restart directory. Cache identity includes the exact sequence and model/runtime; annotations are assembled separately.

Set actual `cohort_id` and `selection_criteria` in the analysis configuration. A cohort manifest supplies these plus `members` with `protein_id`, `sequence_sha256`, and optional `group`, `split`, `evaluation_labels`; membership must exactly match FASTA. Provide biological dependence groups for group summaries. Labels may define evaluation strata, never the explained output or masks. Without `--cohort-manifest`, configuration supplies the cohort definition; `--annotations` is optional. Frozen examples of both inputs are in `examples/validation/v2/`.

No `method_profile` preserves schema 2 and the historical game. `v2x-legacy` uses the same game with schema 3 diagnostics; `v2x-development` permits experimental references/partitions. `config/analysis-v2x.json` enables additional sensitivity/faithfulness runs and therefore costs more than the historical configuration. Unknown profiles fail.

Each run writes `report.json`, `report.html`, `predictions.csv` and `attributions.csv`. JSON is authoritative and contains full class-major attributions, sequences/hashes, model/asset/preprocessing identity, reference sequences, seeds/budgets, residuals, annotations, cohort denominators and warnings. Schema 3 adds runtime identity, diagnostics and cost measurements. Python readers accept both versions:

```python
from pathlib import Path
from mapexploc import load_report
report = load_report(Path("results/fast/report.json"))
```

Full schemas are available from `AnalysisReport.model_json_schema()` and `AnalysisReportV3.model_json_schema()`; positional annotation fields are defined by `mapexploc.annotations.Annotation`. Completed reports can be imported into React without model assets. Legacy interactive exports may contain only visited explanations; use completed reports for scientific export.

Start the viewer with Node 24/pnpm 11.18.0:

```bash
python -m uvicorn mapexploc.api:app --host 127.0.0.1 --port 8000
# In another terminal:
pnpm --dir ui install --frozen-lockfile
pnpm --dir ui dev
```

For native prediction, set `MAPEXPLOC_ADAPTER_CATALOG` to a trusted JSON file mapping each identifier to `{"factory":"deeploc2","configuration":{...}}`, using the operator configurations above. The API `/docs` provides request schemas. React selects configured modes and exports a bound CLI run specification; native explanations run outside the browser. Outside the checkout, set `MAPEXPLOC_MODEL_PATH` to the RF artifact if serving it.

```bash
mapexploc compare --left results/fast/report.json \
  --right results/accurate/report.json --output results/comparison.json
```

Comparisons pair protein ID **and** exact sequence hash, aligning declared classes. Direct attribution correlation requires compatible partitions/reference games; different-method comparisons are confounded. Disagreement describes models, not biological truth.

#### Reproduce the evidence

`examples/baseline/` and `examples/models/` contain the historical prepared split, RF and predictions. `examples/experiments/research-revision/` contains the human development dataset, protocols, folds, predictions, audits and final logistic model. `examples/validation/v2/` contains the 30-protein Fast study; `v2x/` contains the later partial qualification, cohort manifests and overlap audits. Each study's `checksums.json` inventories its scientific files:

```python
import hashlib, json
from pathlib import Path
root = Path("examples/validation/v2")  # repeat for the other study directories
for name, expected in json.loads((root / "checksums.json").read_text()).items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
```

Bulk source snapshots and DeepLoc supervision datasets are not redistributed. Source queries, retrieval dates and hashes are recorded in the protocols/audits; selected positional source records are retained. Prepared-data reproduction is possible; full recuration requires the exact source snapshots. Live queries produce new cohorts. Historical execution snapshots are also unavailable: compare scientific values within recorded tolerances, not cross-machine pickle bytes, and do not resume historical run directories with changed code.

Reproduce the signal-peptide analysis using the configured native Fast assets:

```bash
python scripts/validation/run_v2_cohort.py \
  --inputs examples/validation/v2 --adapter-config /path/to/fast.json \
  --output results/v2-repeat
```

For the human nested study, use a **separate Python 3.14.6** environment and its matching scikit-learn constraints; this differs from the saved RF environment:

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

Repeat the research command into new directories with `--grouping study --single-repeat` or `--cohort note_free --single-repeat` for the two dependence sensitivities. Finalization rejects exact-model calibration; the intermediate model may still be calibrated. Use `mapexploc.research.predict_research(Path("examples/experiments/research-revision/final/model.joblib"), sequences)` for the final logistic model; it is not a tree-service artifact. Its class order is the scientific label table. Refit the historical RF in its original environment with `mapexploc baseline-train --directory examples/baseline --output-model results/rf.joblib --jobs 4`; repeating that holdout is not new validation.

Continuation uses `scripts/validation/qualify_v2x_mode.py` for native parity, `run_v2x_development.py` for the frozen development comparison, and `evaluate_v2x_reports.py` only after method selection and family eligibility are frozen. Their `--help` options take the stored manifests/configurations. Final evaluation must not guide method selection; the historical 30-protein result remains historical evidence.

#### Extend and verify

An external adapter requires `descriptor: AdapterDescriptor` and `predict_proba(sequence_batch)` returning finite `N × C` probabilities in declared class order. The descriptor specifies task type, unique classes, input limits, preprocessing, model/checkpoint identity and native decisions. Multiclass rows sum to one; multilabel outputs retain thresholds without renormalization. Embeddings are optional. Register a trusted factory accepting a configuration mapping under the `mapexploc.adapters` entry-point group, then call public `run_analysis(adapter, proteins, configuration, annotations)`. The separately installable `examples/adapters/sequence_fixture/` demonstrates this contract with synthetic probabilities, not biological evidence. Unsupported explainer/model combinations must remain explicit errors.

For local regressions, install `.[dev,docs]` without replacing pinned runtime versions, then run `python -m pytest tests/`, `pnpm --dir ui check`, `pnpm --dir ui test:e2e` and `mkdocs build --strict`. `python scripts/ci_verify_assets.py` verifies the historical assets. Build with `python -m build`; `python scripts/ci_package_smoke.py dist` checks installation and the independent fixture outside the checkout. Real-model reproduction additionally requires `scripts/validation/qualify_v2.py` in an installed-wheel environment and mode-specific native parity; missing checkpoints or resource rejection cannot count as passes.
