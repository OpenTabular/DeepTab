"""
Round-trip save / load tests — Requirement 4.

For each task type (Regressor, Classifier, LSS) we:
  1. Fit a small model.
  2. Record predictions on a held-out set.
  3. Save the model to a temporary file.
  4. Load it back into a fresh object.
  5. Assert the predictions are bit-for-bit identical.

We use the lightest available model (MLP) to keep CI fast.
"""

import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from deeptab.models import MLPLSS, MLPClassifier, MLPRegressor

# ---------------------------------------------------------------------------
# Shared dataset parameters
# ---------------------------------------------------------------------------

N_SAMPLES = 200
N_FEATURES = 6
N_CLASSES = 3
RANDOM_STATE = 7
FIT_KWARGS: dict[str, Any] = {"max_epochs": 2, "batch_size": 64}


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y = X @ rng.standard_normal(N_FEATURES) + rng.standard_normal(N_SAMPLES)
    df = pd.DataFrame({f"f{i}": X[:, i] for i in range(N_FEATURES)})
    return train_test_split(df, y, test_size=0.2, random_state=RANDOM_STATE)


@pytest.fixture(scope="module")
def classification_data():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y_cont = X @ rng.standard_normal(N_FEATURES) + rng.standard_normal(N_SAMPLES)
    y = pd.qcut(y_cont, q=N_CLASSES, labels=False)
    df = pd.DataFrame({f"f{i}": X[:, i] for i in range(N_FEATURES)})
    return train_test_split(df, y, test_size=0.2, random_state=RANDOM_STATE)


# ---------------------------------------------------------------------------
# Regressor round-trip
# ---------------------------------------------------------------------------


def test_regressor_save_load_predictions(regression_data):
    X_train, X_test, y_train, _y_test = regression_data
    model = MLPRegressor()
    model.fit(X_train, y_train, **FIT_KWARGS)

    preds_before = model.predict(X_test)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        model.save(tmp_path)
        loaded = MLPRegressor.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    preds_after = loaded.predict(X_test)

    np.testing.assert_array_equal(
        preds_before,
        preds_after,
        err_msg="MLPRegressor predictions changed after save/load round-trip",
    )


def test_regressor_save_raises_when_unfitted():
    model = MLPRegressor()
    with pytest.raises(ValueError, match="fitted"):
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            model.save(f.name)


# ---------------------------------------------------------------------------
# Classifier round-trip
# ---------------------------------------------------------------------------


def test_classifier_save_load_predictions(classification_data):
    X_train, X_test, y_train, _y_test = classification_data
    model = MLPClassifier()
    model.fit(X_train, y_train, **FIT_KWARGS)

    preds_before = model.predict(X_test)
    proba_before = model.predict_proba(X_test)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        model.save(tmp_path)
        loaded = MLPClassifier.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    preds_after = loaded.predict(X_test)
    proba_after = loaded.predict_proba(X_test)

    np.testing.assert_array_equal(
        preds_before,
        preds_after,
        err_msg="MLPClassifier.predict changed after save/load round-trip",
    )
    np.testing.assert_array_equal(
        proba_before,
        proba_after,
        err_msg="MLPClassifier.predict_proba changed after save/load round-trip",
    )


# ---------------------------------------------------------------------------
# LSS round-trip
# ---------------------------------------------------------------------------


def test_lss_save_load_predictions(regression_data):
    X_train, X_test, y_train, _y_test = regression_data
    model = MLPLSS()
    model.fit(X_train, y_train, family="normal", **FIT_KWARGS)

    preds_before = model.predict(X_test)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        model.save(tmp_path)
        loaded = MLPLSS.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    preds_after = loaded.predict(X_test)

    np.testing.assert_array_equal(
        preds_before,
        preds_after,
        err_msg="MLPLSS predictions changed after save/load round-trip",
    )
