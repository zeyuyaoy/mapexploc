### Reproduction and extensions of study

Run commands from the repository root unless stated otherwise. Use trusted model files: artifacts are pickle-based. General installation requires Python ≥3.12; scientific reproduction requires the recorded interpreter and dependencies.

#### Evidence files

| Location                                  | Required contents                                                                                        |
|-------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `examples/baseline/`                      | Original prepared CSV, manifest, group assignments and sequence audits                                   |
| `examples/models/`                        | Served forest, original report, individual holdout predictions, scientific environment                   |
| `examples/experiments/research-revision/` | Internal study input, nested folds/predictions, cohort audit and final logistic artifact  |

For the internal study, `primary/`, `study-groups/` and `note-free/` contain protocols, designs, results, `folds/` records and per-protein `predictions.csv` files. Their original completion manifests verify these directly accessible files. `final/finalization.json` and `final/calibration-predictions.npz` record the rejected exact-model calibration. `diagnostics/` contains error tables and feature-reliance analyses; `audit/` records exclusions, annotation notes and provenance.

The study `checksums.json` inventories the retained files. Per-run completion manifests and recorded protocols preserve the original result provenance. Reruns use the current source implementation; the old execution snapshots are no longer distributed, so byte-identical execution or resumption of old runs is not claimed.

```python
import hashlib
import json
from pathlib import Path

root = Path("examples/experiments/research-revision")
for name, expected in json.loads((root / "checksums.json").read_text()).items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
```

The raw 20,431-entry UniProt JSON is not distributed. The retained local copy is `artifacts/source/uniprot-2026_03.json`, with release headers beside it. Its source hash is recorded in the study provenance. Prepared datasets preserve analyzed sequences and structured evidence. Full recuration requires that exact source snapshot; a live download may change the cohort.

#### Reproduce the internal study

Use **Python 3.14.6**, with constraints from `examples/experiments/research-revision/scientific-environment.txt` (NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, SciPy 1.18.1, Biopython 1.88). The constraints record the original environment; unrelated notebook packages need not be installed.

```bash
python3.14 -m venv .venv-research
source .venv-research/bin/activate
python -m pip install -e '.[plots]' \
  -c examples/experiments/research-revision/scientific-environment.txt
python -c 'import platform; assert platform.python_version() == "3.14.6"'

export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR=/tmp/mapexploc-mpl

python -m mapexploc.research \
  --source examples/experiments/research-revision/dataset.csv \
  --directory artifacts/reproduction/primary --jobs 2
python -m mapexploc.research \
  --source examples/experiments/research-revision/dataset.csv \
  --directory artifacts/reproduction/study-groups --grouping study --single-repeat --jobs 2
python -m mapexploc.research \
  --source examples/experiments/research-revision/dataset.csv \
  --directory artifacts/reproduction/note-free --cohort note_free --single-repeat --jobs 2
python scripts/finalize_research_candidate.py \
  --run artifacts/reproduction/primary --output artifacts/reproduction/final
```

Use new output directories. Paths and environment/code hashes are part of each run; do not rewrite an old protocol to make a changed checkout resume. Compare new predictions and metrics with recorded results, not serialized byte hashes across machines. The finalization command rejects calibration for the selected logistic configuration; the intermediate `research-model.joblib` may still be calibrated.

Outer seeds are 20260918/20260919; inner seed is outer seed + zero-based fold + 100. Full-development selection uses split seed 20261018 and estimator seed 20260918. Estimators and BLAS use one worker; `--jobs` controls concurrent candidate fits. Exact memberships are in `design.json`. To reproduce feature/error diagnostics, run `scripts/research_diagnostics.py --run ... --output ...`; the stronger exploratory null uses `scripts/shuffle_sensitivity.py` with the same options and a JSON output path.

#### Historical baseline and new data

The served artifact's environment is `examples/models/scientific-environment.txt` (**Python 3.13.14**, scikit-learn 1.9.0). In a separate matching environment:

```bash
python -m pip install -e . -c examples/models/scientific-environment.txt
mapexploc baseline-train --directory examples/baseline \
  --output-model artifacts/reproduced-human.joblib --jobs 4
```

This preserves the original test partition and writes the fitted model, report and individual predictions. Cross-version regression checks do not establish general pickle compatibility. Repeating the historical holdout does not create new independent validation.

For a new snapshot, install MMseqs2 and use a new directory:

```bash
mapexploc baseline-download --directory artifacts/new-human
mapexploc baseline-prepare --directory artifacts/new-human --threads 4
```

These produce a new benchmark, not reproduction of the frozen source. For new scientific analyses, pass biological dependence groups to training, keep all preprocessing inside training folds, and evaluate selection with untouched groups. `best_cv_score` is a selection score; ungrouped CV and tiny smoke fixtures do not measure transfer to unrelated proteins.

#### Use the fitted models

In the matching research environment, using the source `PYTHONPATH` above:

```python
from pathlib import Path
from Bio import SeqIO
from mapexploc.research import predict_research

sequences = [str(record.seq) for record in SeqIO.parse("proteins.fasta", "fasta")]
probabilities = predict_research(
    Path("examples/experiments/research-revision/final/model.joblib"), sequences
)
```

Columns are Cytoplasm, Membrane, Mitochondrion, Nucleus, Secreted. This logistic artifact is separate from the tree/SHAP service format. The service serves the historical forest selected by `config/default-model.json`; an explicit `MAPEXPLOC_MODEL_PATH` overrides it. Start from the current source:

```bash
PYTHONPATH=src python -m uvicorn mapexploc.api:app --host 127.0.0.1 --port 8000
# Optional interface: Node 24, pnpm 11.18.0
pnpm --dir ui install --frozen-lockfile
pnpm --dir ui dev
```

The API's `/docs` supplies executable request/response schemas. The interface accepts raw sequence or FASTA and exports predictions CSV and analysis JSON. Tree SHAP reports signed engineered-feature contributions to a predicted class; truncated contribution lists need not sum to its probability. Scientific scope and limitations remain those in [Methods and evidence](index.md).
