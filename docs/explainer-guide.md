# SHAP explanations

`ShapExplainer` explains the Random Forest stage of a fitted MAP-ExPLoc pipeline. Before invoking SHAP, it applies
fitted preprocessing steps such as the scaler. This avoids explaining raw values against a model that received
transformed values.

```python
from mapexploc.explainers import ShapExplainer
from mapexploc.features import build_feature_matrix

X = build_feature_matrix(["MKTIIALSYIFCLVFADYKDDDDK"])
report = ShapExplainer(model).explain_predictions(X, top_n=12)[0]
```

Each report contains:

- `prediction`: the class being explained;
- `base_value`: the tree explainer's expected output for that class; and
- `feature_contributions`: ranked feature value and signed SHAP contribution.

A positive contribution supports the predicted class relative to the base value; a negative contribution pulls away from
it. These are model-level feature attributions, not residue-level motif discoveries or causal evidence.

`generate_all_plots(X)` can save a class-level SHAP summary when matplotlib is installed through the `plots` extra. JSON
explanation does not require matplotlib.
