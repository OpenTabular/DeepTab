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
import torch
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
    assert preds_before.shape == (len(X_test),)

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
        bundle = torch.load(tmp_path, weights_only=False)
        loaded = MLPClassifier.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    assert bundle["artifact_metadata"]["format_version"] == 2
    assert bundle["artifact_metadata"]["architecture"]["name"] == "MLP"
    assert bundle["artifact_metadata"]["feature_schema"]["column_order"] == list(X_train.columns)
    assert bundle["artifact_metadata"]["task"]["task"] == "classification"
    assert bundle["artifact_metadata"]["versions"]["packages"]["torch"] is not None
    assert bundle["n_features_in_"] == X_train.shape[1]
    np.testing.assert_array_equal(bundle["feature_names_in_"], np.asarray(X_train.columns, dtype=object))
    np.testing.assert_array_equal(bundle["classes_"], model.classes_)
    assert loaded.input_columns_ == list(X_train.columns)
    assert loaded.n_features_in_ == X_train.shape[1]
    np.testing.assert_array_equal(loaded.feature_names_in_, np.asarray(X_train.columns, dtype=object))
    assert loaded.task_info_["task"] == "classification"
    np.testing.assert_array_equal(loaded.classes_, model.classes_)

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


# ---------------------------------------------------------------------------
# Bundle structure — verifies build_save_bundle produces a consistent artifact
# ---------------------------------------------------------------------------


def test_bundle_structure_regressor(regression_data):
    """build_save_bundle must always produce the required top-level keys."""
    X_train, _X_test, y_train, _y_test = regression_data
    model = MLPRegressor()
    model.fit(X_train, y_train, **FIT_KWARGS)

    from deeptab.core.serialization import build_save_bundle

    bundle = build_save_bundle(model, lss=False, family=None)

    required_keys = {
        "_class",
        "config",
        "config_kwargs",
        "preprocessor",
        "preprocessor_kwargs",
        "feature_info",
        "batch_size",
        "regression",
        "model_class",
        "num_classes",
        "lss",
        "family",
        "optimizer_type",
        "optimizer_kwargs",
        "lr",
        "lr_patience",
        "lr_factor",
        "weight_decay",
        "task_model_state_dict",
        "artifact_metadata",
        "feature_schema",
        "input_columns",
        "task_info",
        "classes_",
        "n_features_in_",
        "feature_names_in_",
        "versions",
    }
    assert required_keys.issubset(bundle.keys()), f"Missing keys: {required_keys - bundle.keys()}"

    meta = bundle["artifact_metadata"]
    assert meta["format_version"] == 2
    assert meta["architecture"]["name"] == "MLP"
    assert meta["task"]["task"] == "regression"
    assert meta["task"]["lss"] is False
    assert meta["task"]["family"] is None
    assert meta["versions"]["packages"]["torch"] is not None
    assert bundle["lss"] is False
    assert bundle["family"] is None
    assert bundle["regression"] is True


def test_bundle_structure_classifier(classification_data):
    """Classifier bundle must record task='classification' and classes_."""
    X_train, _X_test, y_train, _y_test = classification_data
    model = MLPClassifier()
    model.fit(X_train, y_train, **FIT_KWARGS)

    from deeptab.core.serialization import build_save_bundle

    bundle = build_save_bundle(model, lss=False, family=None)

    assert bundle["artifact_metadata"]["task"]["task"] == "classification"
    np.testing.assert_array_equal(bundle["classes_"], model.classes_)
    assert bundle["n_features_in_"] == X_train.shape[1]
    np.testing.assert_array_equal(bundle["feature_names_in_"], np.asarray(X_train.columns, dtype=object))
    assert bundle["input_columns"] == list(X_train.columns)


def test_bundle_raises_when_unfitted():
    """build_save_bundle must raise ValueError if the model is not fitted."""
    from deeptab.core.serialization import build_save_bundle

    model = MLPRegressor()
    with pytest.raises(ValueError, match="fitted"):
        build_save_bundle(model, lss=False, family=None)


def test_restore_base_state(regression_data):
    """restore_base_state must populate all common fields from the bundle."""
    X_train, _X_test, y_train, _y_test = regression_data
    model = MLPRegressor()
    model.fit(X_train, y_train, **FIT_KWARGS)

    from deeptab.core.serialization import _PREPROCESSOR_ARG_NAMES, build_save_bundle, restore_base_state

    bundle = build_save_bundle(model, lss=False, family=None)

    obj = object.__new__(MLPRegressor)
    restore_base_state(obj, bundle)

    assert obj.built is True
    assert obj.is_fitted_ is True
    assert obj.model_config is None
    assert obj.preprocessing_config is None
    assert obj.trainer_config is None
    assert obj.random_state is None
    assert obj.config is bundle["config"]
    assert obj.preprocessor is bundle["preprocessor"]
    assert obj.optimizer_type == bundle["optimizer_type"]
    assert obj.preprocessor_arg_names == list(_PREPROCESSOR_ARG_NAMES)


def test_lss_bundle_structure(regression_data):
    """LSS bundle must set lss=True and record the family name."""
    X_train, _X_test, y_train, _y_test = regression_data
    model = MLPLSS()
    model.fit(X_train, y_train, family="normal", **FIT_KWARGS)

    from deeptab.core.serialization import build_save_bundle

    bundle = build_save_bundle(model, lss=True, family="normal")

    assert bundle["lss"] is True
    assert bundle["family"] == "normal"
    assert bundle["artifact_metadata"]["task"]["lss"] is True
    assert bundle["artifact_metadata"]["task"]["family"] == "normal"
    assert bundle["artifact_metadata"]["task"]["task"] == "distributional_regression"
