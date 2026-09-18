# Python API

## Feature extraction

```python
from mapexploc.features import build_feature_matrix

features = build_feature_matrix(["MKTIIALSYIFCLVFADYKDDDDK"])
assert features.shape == (1, 423)
```

The same function accepts a raw sequence string, a sequence batch, or a FASTA path. Optional annotations are joined by
`sequence_id`, `entry_name`, `accession`, or `id` for FASTA input; otherwise a same-length positional join is required.

## Training

```python
import pandas as pd

from mapexploc.features import build_feature_matrix
from mapexploc.models.rf import train_random_forest

data = pd.read_csv("training.csv")
X = build_feature_matrix(data["sequence"])
result = train_random_forest(X, data["label"])
model = result["model"]
```

Hyperparameter search uses stratified cross-validation when every class has at least two samples. SMOTE is fitted inside
each fold and disabled when the fold is too small.

## Model artifacts

```python
from pathlib import Path
from mapexploc.artifacts import load_model_artifact, save_model_artifact

save_model_artifact(model, Path("model.pkl"))
loaded = load_model_artifact(Path("model.pkl"))
```

The loader verifies the feature schema and supports legacy bare estimators that expose a compatible `feature_names_in_`.
Load only trusted artifacts.

## FastAPI application

```python
from pathlib import Path
from mapexploc.api import create_app

app = create_app(model_path=Path("model.pkl"))
```

The module-level `mapexploc.api:app` reads `MAPEXPLOC_MODEL_PATH`, defaulting to `model.pkl`. The file is loaded on the
first health, model-info or inference request. Readiness requires successful loading.

## Generated reference

::: mapexploc
options:
show_root_heading: true
members_order: source

## Reproducible baseline

```python
from pathlib import Path
from mapexploc.baseline import train_baseline

# Uses frozen evidence, checksums, similarity groups and train/test assignments.
report = train_baseline(
    Path("examples/baseline"), Path("artifacts/my-human-baseline.joblib"), jobs=4
)
print(report["evaluation"]["macro_f1"])
```

Unlike the general Random Forest helper's ordinary stratified CV, this workflow uses grouped CV and an untouched test
set. Never duplicate samples to manufacture cross-validation support. See [reproduction](baseline.md) and
the [model card](model-card.md).

The artifact loader retains metadata as `loaded.metadata`. Old artifacts without metadata remain supported, and `/model`
returns `metadata_available: false` for them.
