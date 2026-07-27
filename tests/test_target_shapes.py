"""Regression tests: a (n, 1) column-vector y must behave like a 1-D y.

fit() documents ``y : array-like, shape (n_samples,) or (n_samples, n_targets)``,
but setup() used ``unsqueeze(dim=1)``, turning an (n, 1) target into a (B, 1, 1)
label tensor. MSELoss then broadcast against the (B, 1) predictions and optimised
an all-pairs objective (R2 collapsed from 0.996 to 0.002), while
BCEWithLogitsLoss refused to broadcast and raised.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from deeptab.configs.core import PreprocessingConfig
from deeptab.models import MLPClassifier, MLPRegressor

QUIET = {
    "accelerator": "cpu",
    "devices": 1,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "logger": False,
}


def _regression_data(n=80, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = 3.0 * X["a"].to_numpy() + 0.1 * rng.normal(size=n)
    return X, y


@pytest.mark.parametrize("as_column_vector", [False, True])
def test_regression_label_tensor_is_2d(as_column_vector):
    X, y = _regression_data()
    target = y.reshape(-1, 1) if as_column_vector else y

    model = MLPRegressor(random_state=0)
    model.fit(X, target, max_epochs=1, batch_size=16, **QUIET)
    model._data_module.setup("fit")  # type: ignore[union-attr]

    labels = model._data_module.train_dataset.labels  # type: ignore[union-attr]
    assert labels.ndim == 2, f"expected (B, 1) labels, got {tuple(labels.shape)}"


def test_column_vector_y_trains_as_well_as_1d():
    X, y = _regression_data()
    scores = {}
    for tag, target in (("1d", y), ("2d", y.reshape(-1, 1))):
        model = MLPRegressor(
            preprocessing_config=PreprocessingConfig(numerical_preprocessing="standardization"),
            random_state=0,
        )
        model.fit(X, target, max_epochs=30, batch_size=16, lr=1e-2, patience=30, **QUIET)
        scores[tag] = r2_score(y, model.predict(X))

    assert scores["1d"] > 0.9, f"1-D baseline did not train: {scores}"
    assert scores["2d"] == pytest.approx(scores["1d"], abs=0.1), f"column vector trained differently: {scores}"


def test_binary_classification_accepts_column_vector_y():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=48), "b": rng.normal(size=48)})
    y = (X["a"].to_numpy() > 0).astype(int)

    model = MLPClassifier(random_state=0)
    model.fit(X, y.reshape(-1, 1), max_epochs=1, batch_size=16, **QUIET)
    assert model.predict(X).shape == (48,)
