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
import torch.nn as nn
from sklearn.model_selection import train_test_split

from deeptab.core.exceptions import DeviceUnavailableError, InvalidDeviceError
from deeptab.models import MLPLSS, MLPClassifier, MLPRegressor
from deeptab.training import TaskModel
from deeptab.training.losses import FocalLoss, WeightedBCEWithLogitsLoss, WeightedCrossEntropyLoss

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


@pytest.fixture(scope="module")
def binary_classification_data():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y_cont = X @ rng.standard_normal(N_FEATURES) + rng.standard_normal(N_SAMPLES)
    y = pd.qcut(y_cont, q=2, labels=False)
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

    assert obj._built is True
    assert obj.is_fitted_ is True
    assert obj.model_config is None
    assert obj.preprocessing_config is None
    assert obj.trainer_config is None
    assert obj.random_state is None
    assert obj.config is bundle["config"]
    assert obj._preprocessor is bundle["preprocessor"]
    assert obj._optimizer_type == bundle["optimizer_type"]
    assert obj._preprocessor_arg_names == list(_PREPROCESSOR_ARG_NAMES)


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


# ---------------------------------------------------------------------------
# Preprocessing spec / fingerprint / feature-width persistence
# ---------------------------------------------------------------------------


def test_preprocessing_metadata_includes_spec_fingerprint_and_widths(regression_data):
    """The saved bundle records the PreTab spec, fingerprint, and per-feature widths."""
    X_train, _X_test, y_train, _y_test = regression_data
    model = MLPRegressor()
    model.fit(X_train, y_train, **FIT_KWARGS)

    from deeptab.core.serialization import build_save_bundle

    bundle = build_save_bundle(model, lss=False, family=None)
    meta = bundle["preprocessing_metadata"]

    assert isinstance(meta["spec"], dict)
    assert meta["spec"]["schema_version"] == 1
    assert isinstance(meta["fingerprint"], str) and meta["fingerprint"]
    assert set(meta["feature_widths"].keys()) == set(X_train.columns)
    assert all(isinstance(width, int) and width > 0 for width in meta["feature_widths"].values())


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


def test_preprocessing_metadata_handles_preprocessor_without_spec_support():
    """A preprocessor lacking to_spec()/fingerprint_ still produces a valid metadata block."""
    from deeptab.core.serialization import build_preprocessing_metadata

    class _BarePreprocessor:
        pass

    meta = build_preprocessing_metadata(_BarePreprocessor(), preprocessor_kwargs={})

    assert meta["spec"] is None
    assert meta["fingerprint"] is None
    assert meta["feature_widths"] == {}


def test_preprocessing_metadata_none_preprocessor():
    """A None preprocessor (no fitted state) yields spec=None, fingerprint=None."""
    from deeptab.core.serialization import build_preprocessing_metadata

    meta = build_preprocessing_metadata(None, preprocessor_kwargs=None)

    assert meta["fitted_state_persisted"] is False
    assert meta["spec"] is None
    assert meta["fingerprint"] is None
    assert meta["feature_widths"] == {}


# ---------------------------------------------------------------------------
# Fitted loss persistence (#445) — the loss a model was actually trained with
# must survive a save/load round trip, instead of being re-derived from
# num_classes alone.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# load()'s `device` parameter
#
# load() must not let whatever hardware happens to be visible on the loading
# machine decide how the reconstructed model runs. It defaults to "cpu"
# regardless of the accelerator used during fit(), and even "auto" (which
# opts back into automatic accelerator selection) always pins a single
# device, never distributed multi-device inference.
# ---------------------------------------------------------------------------


def test_resolve_inference_accelerator_valid_devices():
    from deeptab.core.serialization import resolve_inference_accelerator

    assert resolve_inference_accelerator("cpu") == ("cpu", 1, "cpu")

    if torch.cuda.is_available():
        assert resolve_inference_accelerator("cuda") == ("cuda", 1, "cuda")
    else:
        with pytest.raises(DeviceUnavailableError, match="not available"):
            resolve_inference_accelerator("cuda")

    if torch.backends.mps.is_available():
        assert resolve_inference_accelerator("mps") == ("mps", 1, "mps")
    else:
        with pytest.raises(DeviceUnavailableError, match="not available"):
            resolve_inference_accelerator("mps")

    accelerator, devices, map_location = resolve_inference_accelerator("auto")
    assert accelerator == "auto"
    assert devices == 1
    assert map_location in ("cpu", "cuda", "mps")


def test_resolve_inference_accelerator_invalid_device_raises():
    from deeptab.core.serialization import resolve_inference_accelerator

    with pytest.raises(InvalidDeviceError, match="device must be one of"):
        resolve_inference_accelerator("tpu")


def test_resolve_inference_accelerator_unavailable_device_raises(monkeypatch):
    """device="cuda" on a machine without CUDA must raise a clear error."""
    from deeptab.core.serialization import resolve_inference_accelerator

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(DeviceUnavailableError, match="not available"):
        resolve_inference_accelerator("cuda")


def test_resolve_inference_accelerator_auto_pins_single_device(monkeypatch):
    """Even when multiple GPUs would be visible, device="auto" must still
    resolve to a single device, never Lightning's own multi-device
    auto-scaling."""
    from deeptab.core.serialization import resolve_inference_accelerator

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    accelerator, devices, map_location = resolve_inference_accelerator("auto")
    assert accelerator == "auto"
    assert devices == 1
    assert map_location == "cuda"


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


# ---------------------------------------------------------------------------
# NDTF round trip
#
# NDTF randomly picks each tree's input width, depth, and temperature at
# construction time, which sizes that tree's weight tensors. load() must
# reconstruct trees with the exact same shapes the saved weights were
# trained with, instead of drawing a fresh, differently-shaped forest.
# ---------------------------------------------------------------------------


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
