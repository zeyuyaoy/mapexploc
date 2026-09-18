# Model adapters

The HTTP layer consumes sequence-oriented models: `predict(batch)` and `predict_proba(batch)` both receive protein-sequence strings. A custom adapter may optionally implement `embed(batch)`.

```python
import numpy as np

class SequenceAdapter:
    classes = ("cytosol", "secreted")

    def predict(self, batch):
        return np.array(["cytosol"] * len(batch))

    def predict_proba(self, batch):
        return np.array([[0.8, 0.2]] * len(batch))
```

Pass it directly to `create_app(model=SequenceAdapter())`. Prediction works for any conforming adapter. The built-in `/explain` route deliberately returns 501 for custom sequence models because the repository only implements SHAP for its feature-based tree model; adapters must not receive a fabricated explanation.

Scikit-learn estimators trained on MAP-ExPLoc features can be wrapped with `FeatureModelAdapter`. It applies the same validated 423-feature transform at inference:

```python
from mapexploc.adapter import FeatureModelAdapter

adapter = FeatureModelAdapter(fitted_pipeline)
labels = adapter.predict(["MKTIIALSYIFCLVFADYKDDDDK"])
```

The API automatically applies this wrapper to a fitted scikit-learn pipeline or artifact.
