import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.pipeline import make_pipeline

from mapexploc.adapter import load_adapter, predict_probabilities
from mapexploc.contracts import AdapterDescriptor
from mapexploc.explainers.shap import ShapExplainer


@pytest.mark.parametrize("classifier", [RandomForestClassifier, ExtraTreesClassifier])
@pytest.mark.parametrize("columns", [["b", "a"], ["a"]])
def test_transformed_identity_and_all_class_probability(classifier, columns):
    x = pd.DataFrame({"a": [0, 0, 1, 1], "b": [0, 0, 0, 0]})
    model = make_pipeline(
        ColumnTransformer([("selection", "passthrough", columns)]),
        classifier(n_estimators=5, random_state=2),
    ).fit(x, ["z", "z", "a", "a"])
    result = ShapExplainer(model).explain_sample(x)
    assert np.max(np.abs(result["shap_values"][:, :, 1])) == 0
    assert np.max(np.abs(result["shap_values"][:, :, 0])) > 0
    np.testing.assert_allclose(
        result["expected_value"] + result["shap_values"].sum(axis=2),
        model.predict_proba(x),
        atol=1e-6,
    )


def test_opaque_projection_and_margin_fail_closed():
    x = pd.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]})
    y = [0, 0, 1, 1]
    model = make_pipeline(PCA(1), RandomForestClassifier(n_estimators=3)).fit(x, y)
    with pytest.raises(TypeError, match="biological feature mapping"):
        ShapExplainer(model).explain_sample(x)
    with pytest.raises(TypeError, match="probability explanations"):
        ShapExplainer(GradientBoostingClassifier().fit(x, y))


def test_malformed_tensor_rejected():
    with pytest.raises(ValueError, match="shape"):
        ShapExplainer._normalise_values(np.zeros((2, 3)), 2, 3, 2)
    with pytest.raises(ValueError, match="Nonfinite"):
        ShapExplainer._normalise_values(np.full((2, 3, 2), np.nan), 2, 3, 2)


def test_legacy_labels_survive_optional_embedding():
    class Legacy:
        classes_ = ["outside", "inside"]

        def predict_proba(self, batch):
            return [[0.3, 0.7]] * len(batch)

        def embed(self, batch):
            return np.zeros((len(batch), 2))

    adapter = load_adapter(Legacy())
    assert adapter.descriptor.classes == ("outside", "inside")
    np.testing.assert_equal(predict_probabilities(adapter, ["ACD"]), [[0.3, 0.7]])


@pytest.mark.parametrize(
    "values", [[[0.9, 0.8]], [[float("nan"), 0.5]], [[-0.1, 1.1]], [[0.2]]]
)
def test_invalid_probabilities(values):
    class Bad:
        descriptor = AdapterDescriptor(
            model_id="fixture", classes=("a", "b"), preprocessing_id="identity"
        )

        def predict_proba(self, batch):
            return values

    with pytest.raises(ValueError):
        predict_probabilities(Bad(), ["ACD"])


def test_multilabel_never_renormalized():
    class Multi:
        descriptor = AdapterDescriptor(
            model_id="multi",
            classes=("a", "b"),
            preprocessing_id="identity",
            task_type="multilabel",
            thresholds=(0.5, 0.6),
        )

        def predict_proba(self, batch):
            return [[0.9, 0.8]] * len(batch)

    np.testing.assert_equal(predict_probabilities(Multi(), ["ACD"]), [[0.9, 0.8]])
