# SHAP explanations

`ShapExplainer` explains the final Random Forest or Extra Trees estimator of a fitted MAP-ExPLoc pipeline, including legacy pipelines whose final step is named `rf`. Before invoking SHAP, it applies fitted preprocessing steps such as the scaler. This avoids explaining raw values against a model that received transformed values.

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

A positive contribution supports the predicted class relative to the base value; a negative contribution pulls away from it. These are model-level feature attributions, not residue-level motif discoveries or causal evidence.

`generate_all_plots(X)` can save a class-level SHAP summary when matplotlib is installed through the `plots` extra. JSON explanation does not require matplotlib.

The frozen logistic research candidate is not supported by this tree/SHAP service path. Its coefficients describe association in standardized engineered features, not causal targeting mechanisms.

For perturbation summaries, `evaluation.aopc` accepts **precomputed signed score drops**, equally weighted across perturbation steps. `mean_absolute_score_change`reports only the magnitude of an output change. The old `insertion_deletion` name is a deprecated alias for that calculation; it does not compute insertion/deletion curve AUC. Neither helper supplies a perturbation design or biological validation.
