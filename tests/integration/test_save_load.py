"""Tests for save load."""

import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from deeptab.core.exceptions import InvalidDeviceError
from deeptab.models import MLPLSS, MLPClassifier, MLPRegressor
from deeptab.training import TaskModel
from deeptab.training.losses import FocalLoss, WeightedBCEWithLogitsLoss, WeightedCrossEntropyLoss

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


@pytest.fixture(scope="module")
def binary_classification_data():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y_cont = X @ rng.standard_normal(N_FEATURES) + rng.standard_normal(N_SAMPLES)
    y = pd.qcut(y_cont, q=2, labels=False)
    df = pd.DataFrame({f"f{i}": X[:, i] for i in range(N_FEATURES)})
    return train_test_split(df, y, test_size=0.2, random_state=RANDOM_STATE)


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
        # device="auto" reproduces fit()'s own (unpinned) accelerator choice, so
        # this stays a same-hardware, bit-exact round-trip check. load()'s
        # actual default ("cpu") is exercised separately in the device tests below.
        loaded = MLPRegressor.load(tmp_path, device="auto")
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
        # device="auto" reproduces fit()'s own (unpinned) accelerator choice, so
        # this stays a same-hardware, bit-exact round-trip check. load()'s
        # actual default ("cpu") is exercised separately in the device tests below.
        loaded = MLPClassifier.load(tmp_path, device="auto")
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


def test_lss_save_load_predictions(regression_data):
    X_train, X_test, y_train, _y_test = regression_data
    model = MLPLSS()
    model.fit(X_train, y_train, family="normal", **FIT_KWARGS)

    preds_before = model.predict(X_test)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        model.save(tmp_path)
        # device="auto" reproduces fit()'s own (unpinned) accelerator choice, so
        # this stays a same-hardware, bit-exact round-trip check. load()'s
        # actual default ("cpu") is exercised separately in the device tests below.
        loaded = MLPLSS.load(tmp_path, device="auto")
    finally:
        os.unlink(tmp_path)

    preds_after = loaded.predict(X_test)

    np.testing.assert_array_equal(
        preds_before,
        preds_after,
        err_msg="MLPLSS predictions changed after save/load round-trip",
    )


def test_preprocessing_metadata_survives_save_load_round_trip(regression_data):
    """spec/fingerprint/feature_widths are unchanged after a save/load round-trip."""
    X_train, _X_test, y_train, _y_test = regression_data
    model = MLPRegressor()
    model.fit(X_train, y_train, **FIT_KWARGS)

    import json

    from deeptab.core.serialization import build_save_bundle

    meta_before = build_save_bundle(model, lss=False, family=None)["preprocessing_metadata"]

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        model.save(tmp_path)
        loaded = MLPRegressor.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    meta_after = loaded.preprocessing_metadata_
    assert meta_after["fingerprint"] == meta_before["fingerprint"]
    assert meta_after["feature_widths"] == meta_before["feature_widths"]
    # Compared via their JSON text rather than `==`: the spec's fitted state can
    # legitimately contain NaN (e.g. sklearn's `SimpleImputer(missing_values=nan)`),
    # and NaN never equals itself, which would make an identical dict compare unequal.
    assert json.dumps(meta_after["spec"], sort_keys=True) == json.dumps(meta_before["spec"], sort_keys=True)


def _save_and_load(model, model_cls):
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        model.save(tmp_path)
        loaded = model_cls.load(tmp_path)
    finally:
        os.unlink(tmp_path)
    return loaded


def _loss_fct_of(model):
    task_model = model._task_model
    assert isinstance(task_model, TaskModel)
    return task_model.loss_fct


def test_classifier_save_load_preserves_plain_binary_loss_type(binary_classification_data):
    """A plain binary classifier (no class_weight) must reload with the loss it
    was actually trained with, not the regression default (MSELoss)."""
    X_train, _X_test, y_train, _y_test = binary_classification_data
    model = MLPClassifier()
    model.fit(X_train, y_train, **FIT_KWARGS)
    assert isinstance(_loss_fct_of(model), nn.BCEWithLogitsLoss)

    loaded = _save_and_load(model, MLPClassifier)

    loaded_loss = _loss_fct_of(loaded)
    assert isinstance(loaded_loss, nn.BCEWithLogitsLoss)
    assert not isinstance(loaded_loss, nn.MSELoss)


def test_classifier_save_load_preserves_class_weight_binary(binary_classification_data):
    """A binary classifier trained with class_weight must reload without error,
    and the reloaded loss must carry the same pos_weight."""
    X_train, _X_test, y_train, _y_test = binary_classification_data
    model = MLPClassifier()
    model.fit(X_train, y_train, class_weight="balanced", **FIT_KWARGS)
    original_loss = _loss_fct_of(model)
    assert isinstance(original_loss, WeightedBCEWithLogitsLoss)
    assert original_loss.pos_weight is not None
    original_pos_weight = original_loss.pos_weight.clone()

    loaded = _save_and_load(model, MLPClassifier)

    loaded_loss = _loss_fct_of(loaded)
    assert isinstance(loaded_loss, WeightedBCEWithLogitsLoss)
    torch.testing.assert_close(loaded_loss.pos_weight, original_pos_weight)


def test_classifier_save_load_preserves_class_weight_multiclass(classification_data):
    """A multiclass classifier trained with class_weight must reload without
    error, and the reloaded loss must carry the same per-class weights."""
    X_train, _X_test, y_train, _y_test = classification_data
    model = MLPClassifier()
    model.fit(X_train, y_train, class_weight="balanced", **FIT_KWARGS)
    original_loss = _loss_fct_of(model)
    assert isinstance(original_loss, WeightedCrossEntropyLoss)
    assert original_loss.weight is not None
    original_weight = original_loss.weight.clone()

    loaded = _save_and_load(model, MLPClassifier)

    loaded_loss = _loss_fct_of(loaded)
    assert isinstance(loaded_loss, WeightedCrossEntropyLoss)
    torch.testing.assert_close(loaded_loss.weight, original_weight)


def test_classifier_save_load_preserves_focal_loss(binary_classification_data):
    """A classifier trained with loss_fct='focal' must reload as FocalLoss,
    not silently fall back to MSELoss."""
    X_train, _X_test, y_train, _y_test = binary_classification_data
    model = MLPClassifier()
    model.fit(X_train, y_train, loss_fct="focal", **FIT_KWARGS)
    assert isinstance(_loss_fct_of(model), FocalLoss)

    loaded = _save_and_load(model, MLPClassifier)

    loaded_loss = _loss_fct_of(loaded)
    assert isinstance(loaded_loss, FocalLoss)
    assert not isinstance(loaded_loss, nn.MSELoss)


def test_regressor_save_load_preserves_custom_loss_fct(regression_data):
    """A regressor trained with a custom loss_fct must reload with that same
    loss, not the implicit MSELoss default."""
    X_train, _X_test, y_train, _y_test = regression_data
    model = MLPRegressor()
    model.fit(X_train, y_train, loss_fct=nn.HuberLoss(delta=1.0), **FIT_KWARGS)
    assert isinstance(_loss_fct_of(model), nn.HuberLoss)

    loaded = _save_and_load(model, MLPRegressor)

    loaded_loss = _loss_fct_of(loaded)
    assert isinstance(loaded_loss, nn.HuberLoss)
    assert loaded_loss.delta == 1.0


def test_load_uses_map_location_matching_resolved_device(regression_data):
    """load() must forward a concrete map_location to torch.load(), so weights
    saved from a different device can be deserialized safely instead of
    crashing when that device isn't available on the loading machine."""
    from unittest.mock import patch

    X_train, _X_test, y_train, _y_test = regression_data
    model = MLPRegressor()
    model.fit(X_train, y_train, **FIT_KWARGS)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        model.save(tmp_path)
        with patch("torch.load", wraps=torch.load) as mock_load:
            MLPRegressor.load(tmp_path, device="cpu")
        assert mock_load.call_args.kwargs["map_location"] == "cpu"
    finally:
        os.unlink(tmp_path)


def test_load_defaults_to_cpu_accelerator(regression_data):
    """load() with no `device` argument must pin the reconstructed trainer to
    CPU, even though fit() itself left the accelerator unpinned ("auto")."""
    X_train, _X_test, y_train, _y_test = regression_data
    model = MLPRegressor()
    model.fit(X_train, y_train, **FIT_KWARGS)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        model.save(tmp_path)
        loaded = MLPRegressor.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    assert type(loaded._trainer.accelerator).__name__ == "CPUAccelerator"


def test_load_device_auto_opts_into_automatic_selection(regression_data):
    """device="auto" must still work and re-enable Lightning's own automatic
    hardware selection (the pre-#465 behavior), as an explicit opt-in."""
    X_train, X_test, y_train, _y_test = regression_data
    model = MLPRegressor()
    model.fit(X_train, y_train, **FIT_KWARGS)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        model.save(tmp_path)
        loaded = MLPRegressor.load(tmp_path, device="auto")
    finally:
        os.unlink(tmp_path)

    preds = loaded.predict(X_test)
    assert preds.shape == (len(X_test),)


def test_load_explicit_cpu_device(regression_data):
    X_train, X_test, y_train, _y_test = regression_data
    model = MLPRegressor()
    model.fit(X_train, y_train, **FIT_KWARGS)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        model.save(tmp_path)
        loaded = MLPRegressor.load(tmp_path, device="cpu")
    finally:
        os.unlink(tmp_path)

    assert type(loaded._trainer.accelerator).__name__ == "CPUAccelerator"
    preds = loaded.predict(X_test)
    assert preds.shape == (len(X_test),)


def test_regressor_explicit_cpu_load_predictions_match(regression_data):
    """fit -> predict -> save -> explicit device="cpu" load -> predict must match within floating-point tolerance."""
    X_train, X_test, y_train, _y_test = regression_data
    model = MLPRegressor()
    model.fit(X_train, y_train, **FIT_KWARGS)
    preds_before = model.predict(X_test)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        model.save(tmp_path)
        loaded = MLPRegressor.load(tmp_path, device="cpu")
    finally:
        os.unlink(tmp_path)

    preds_after = loaded.predict(X_test)
    assert preds_after.shape == preds_before.shape
    # allclose, not exact equality: fit() trains on the auto-selected accelerator
    # (e.g. MPS/CUDA), while an explicit device="cpu" reload runs inference on a
    # different backend, whose floating-point rounding differs at the ULP level.
    np.testing.assert_allclose(
        preds_before,
        preds_after,
        rtol=1e-4,
        atol=1e-6,
        err_msg="MLPRegressor predictions changed after an explicit device='cpu' save/load round-trip",
    )


def test_classifier_explicit_cpu_load_predictions_match(classification_data):
    """fit -> predict -> save -> explicit device="cpu" load -> predict must match within floating-point tolerance."""
    X_train, X_test, y_train, _y_test = classification_data
    model = MLPClassifier()
    model.fit(X_train, y_train, **FIT_KWARGS)
    preds_before = model.predict(X_test)
    proba_before = model.predict_proba(X_test)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        model.save(tmp_path)
        loaded = MLPClassifier.load(tmp_path, device="cpu")
    finally:
        os.unlink(tmp_path)

    preds_after = loaded.predict(X_test)
    proba_after = loaded.predict_proba(X_test)
    assert preds_after.shape == preds_before.shape
    assert proba_after.shape == proba_before.shape
    # allclose, not exact equality: fit() trains on the auto-selected accelerator
    # (e.g. MPS/CUDA), while an explicit device="cpu" reload runs inference on a
    # different backend, whose floating-point rounding differs at the ULP level.
    np.testing.assert_array_equal(
        preds_before,
        preds_after,
        err_msg="MLPClassifier.predict changed after an explicit device='cpu' save/load round-trip",
    )
    np.testing.assert_allclose(
        proba_before,
        proba_after,
        rtol=1e-4,
        atol=1e-6,
        err_msg="MLPClassifier.predict_proba changed after an explicit device='cpu' save/load round-trip",
    )


def test_load_invalid_device_raises(regression_data):
    X_train, _X_test, y_train, _y_test = regression_data
    model = MLPRegressor()
    model.fit(X_train, y_train, **FIT_KWARGS)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        model.save(tmp_path)
        with pytest.raises(InvalidDeviceError, match="device must be one of"):
            MLPRegressor.load(tmp_path, device="tpu")
    finally:
        os.unlink(tmp_path)


def test_lss_load_defaults_to_cpu_accelerator(regression_data):
    X_train, _X_test, y_train, _y_test = regression_data
    model = MLPLSS()
    model.fit(X_train, y_train, family="normal", **FIT_KWARGS)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        model.save(tmp_path)
        loaded = MLPLSS.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    assert type(loaded._trainer.accelerator).__name__ == "CPUAccelerator"


def test_classifier_load_defaults_to_cpu_accelerator(classification_data):
    X_train, _X_test, y_train, _y_test = classification_data
    model = MLPClassifier()
    model.fit(X_train, y_train, **FIT_KWARGS)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        model.save(tmp_path)
        loaded = MLPClassifier.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    assert type(loaded._trainer.accelerator).__name__ == "CPUAccelerator"


def test_ndtf_regressor_save_load_predictions(regression_data):
    from deeptab.models import NDTFRegressor

    X_train, X_test, y_train, _y_test = regression_data
    model = NDTFRegressor()
    model.fit(X_train, y_train, **FIT_KWARGS)
    preds_before = model.predict(X_test)

    with tempfile.NamedTemporaryFile(suffix=".deeptab", delete=False) as f:
        tmp_path = f.name
    try:
        model.save(tmp_path)
        loaded = NDTFRegressor.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    preds_after = loaded.predict(X_test)
    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5, atol=1e-6)


def test_ndtf_classifier_save_load_predictions(classification_data):
    from deeptab.models import NDTFClassifier

    X_train, X_test, y_train, _y_test = classification_data
    model = NDTFClassifier()
    model.fit(X_train, y_train, **FIT_KWARGS)
    preds_before = model.predict_proba(X_test)

    with tempfile.NamedTemporaryFile(suffix=".deeptab", delete=False) as f:
        tmp_path = f.name
    try:
        model.save(tmp_path)
        loaded = NDTFClassifier.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    preds_after = loaded.predict_proba(X_test)
    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5, atol=1e-6)


def test_ndtf_architecture_state_persists_tree_shapes(regression_data):
    """The saved bundle must carry the exact per-tree shapes NDTF generated,
    and load() must rebuild trees using those values rather than new ones."""
    from deeptab.models import NDTFRegressor

    X_train, _X_test, y_train, _y_test = regression_data
    model = NDTFRegressor()
    model.fit(X_train, y_train, **FIT_KWARGS)
    architecture_state = model._estimator.get_architecture_state()

    with tempfile.NamedTemporaryFile(suffix=".deeptab", delete=False) as f:
        tmp_path = f.name
    try:
        model.save(tmp_path)
        loaded = NDTFRegressor.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    assert loaded._estimator.get_architecture_state() == architecture_state
