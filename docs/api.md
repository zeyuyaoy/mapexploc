# Python API

## Feature extraction

```python
from mapexploc.features import build_feature_matrix

features = build_feature_matrix(["MKTIIALSYIFCLVFADYKDDDDK"])
assert features.shape == (1, 423)
```

The same function accepts a raw sequence string, a sequence batch, or a FASTA path. Optional annotations are joined by `sequence_id`, `entry_name`, `accession`, or `id` for FASTA input; otherwise a same-length positional join is required.

## Training

```python
import pandas as pd

from mapexploc.features import build_feature_matrix
from mapexploc.models.rf import train_random_forest

data = pd.read_csv("training.csv")
X = build_feature_matrix(data["sequence"])
result = train_random_forest(
    X, data["label"], groups=data["homology_group"], use_smote=False
)
model = result["model"]
```

`homology_group` must be constructed before model selection using the declared sequence-dependence policy. Grouped CV keeps groups disjoint and requires all classes in every fold. Without `groups`, the helper uses ordinary stratified CV and logs that it cannot establish generalization to unrelated proteins. The default selection score is macro-F1; `best_cv_score` is a model-selection estimate, not an independent performance estimate. RF may fit without CV for a tiny workflow fixture and returns `best_cv_score=None`; such a fit supplies no validation evidence.

SMOTE is **disabled by default**. If explicitly requested, scaling and resampling occur within training folds; SMOTE is disabled when fold support is insufficient. The k-NN helper uses stratified/grouped folds and rejects inadequate class support or neighbor counts exceeding a training fold. Its old `test_features`/`test_targets`arguments are rejected rather than silently ignored; evaluate a frozen model separately.

## Model artifacts

```python
from pathlib import Path
from mapexploc.artifacts import load_model_artifact, save_model_artifact

save_model_artifact(model, Path("model.pkl"))
loaded = load_model_artifact(Path("model.pkl"))
```

The loader verifies the feature schema and supports legacy bare estimators that expose a compatible `feature_names_in_`. Load only trusted artifacts.

## FastAPI application

```python
from pathlib import Path
from mapexploc.api import create_app

app = create_app(model_path=Path("model.pkl"))
```

The module-level `mapexploc.api:app` reads `MAPEXPLOC_MODEL_PATH` first, then the source repository's `config/default-model.json`. The manifest pins the artifact path, SHA-256 and model ID. Resolution does not depend on the working directory and never implicitly loads `model.pkl`. A wheel without reference assets requires an explicit trusted path. The first health, model-info or inference request loads and validates the model; readiness requires success. Invalid overrides or manifests do not fall back.

Explicit `create_app(model=...)` and `create_app(model_path=...)` arguments take precedence. Use `create_app(use_default=True)` to opt into startup resolution; an argument-free factory still creates an unavailable application for embedding/tests. `/model` exposes allowlisted provenance, model family and evaluation status. Legacy artifacts remain supported and report unavailable metadata explicitly.

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

This historical reproduction uses the saved grouped split. Its 364-protein holdout has already been inspected and does not become a new independent test when rerun. Never duplicate samples to manufacture cross-validation support. See [reproduction](baseline.md) and the [model card](model-card.md).

`evaluate_rf` and `evaluate_knn` use the shared fixed-class metric implementation. They preserve the estimator's probability-column order, validate probabilities, report missing-class ROC/AP as null, and include log loss, Brier score and calibration diagnostics. Missing labels are not negatives. These helpers do not establish split independence or perform uncertainty analysis; use the appropriate study protocol.

The frozen logistic candidate has a separate artifact and inference function: `mapexploc.research.predict_research`. Do not pass it to the tree/SHAP service loader.

The artifact loader retains metadata as `loaded.metadata`. Old artifacts without metadata remain supported, and `/model` returns `metadata_available: false` for them.

The [CPU experiment workflow](model-improvement.md) separately compares Random Forest and Extra Trees using grouped validation, historical-group exclusions, and conditional independent confirmation. Its four CLI stages are `experiment-prepare`, `experiment-train`, `experiment-evaluate`, and `model-promote`.
