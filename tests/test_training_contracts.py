"""Small synthetic fixtures for general helper contracts, not model validation."""

import numpy as np
import pandas as pd
import pytest

from mapexploc.models.knn import train_knn
from mapexploc.models.rf import train_random_forest


def test_knn_rejects_ignored_test_inputs_and_invalid_class_support() -> None:
    x = pd.DataFrame({"x": np.arange(6)})
    y = pd.Series(["a"] * 3 + ["b"] * 3)
    with pytest.raises(ValueError, match="separately"):
        train_knn(x, y, test_features=x, test_targets=y)
    with pytest.raises(ValueError, match="every class"):
        train_knn(x, pd.Series(["a"] * 5 + ["b"]))
    with pytest.raises(ValueError, match="Neighbor count"):
        train_knn(x, y, param_grid={"knn__n_neighbors": [5]}, cv=3)
    with pytest.raises(ValueError, match="conflicts"):
        train_knn(x.assign(localization="a"), y)
    for train in (train_knn, train_random_forest):
        with pytest.raises(ValueError, match="two folds"):
            train(x, y, cv=1)


def test_knn_small_sample_folds_are_stratified_and_reproducible() -> None:
    x = pd.DataFrame({"x": np.arange(8)})
    y = pd.Series(["a"] * 4 + ["b"] * 4)
    kwargs = {"param_grid": {"knn__n_neighbors": [1]}, "cv": 4, "n_jobs": 1}
    first = train_knn(x, y, **kwargs)["grid_search"]
    second = train_knn(x, y, **kwargs)["grid_search"]
    for (fit, valid), (fit2, valid2) in zip(first.cv, second.cv):
        assert set(y.iloc[fit]) == set(y.iloc[valid]) == {"a", "b"}
        np.testing.assert_array_equal(fit, fit2)
        np.testing.assert_array_equal(valid, valid2)
