"""Tests for serialization."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.model_selection import train_test_split

from deeptab.configs import MLPConfig, PreprocessingConfig, TrainerConfig
from deeptab.core.exceptions import DeviceUnavailableError, InvalidDeviceError
from deeptab.models import MLPLSS, MLPClassifier, MLPRegressor

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
    assert obj._is_pretrained is False
    assert obj._best_model_path is None
    assert obj.model_config is None
    assert obj.preprocessing_config is None
    assert obj.trainer_config is None
    assert obj.random_state is None
    assert obj.config is bundle["config"]
    assert obj._preprocessor is bundle["preprocessor"]
    assert obj._optimizer_type == bundle["optimizer_type"]
    assert obj._preprocessor_arg_names == list(_PREPROCESSOR_ARG_NAMES)


def test_restore_base_state_recovers_older_bundle_configs():
    from deeptab.core.serialization import restore_base_state

    config = MLPConfig(layer_sizes=[16])
    bundle = {
        "config": config,
        "config_kwargs": config.get_params(),
        "preprocessor_kwargs": {"numerical_method": "standardization", "output_dim": 8},
        "preprocessor": None,
        "optimizer_type": "SGD",
        "optimizer_kwargs": {"momentum": 0.5},
        "batch_size": 16,
        "lr": 0.05,
        "lr_patience": 3,
        "lr_factor": 0.2,
        "weight_decay": 0.01,
    }
    obj = object.__new__(MLPRegressor)
    restore_base_state(obj, bundle)

    assert obj.model_config is config
    assert isinstance(obj.preprocessing_config, PreprocessingConfig)
    assert obj.preprocessing_config.numerical_method == "standardization"
    assert obj.preprocessing_config.output_dim == 8
    assert isinstance(obj.trainer_config, TrainerConfig)
    assert obj.trainer_config.batch_size == 16
    assert obj.trainer_config.lr == 0.05
    assert obj.trainer_config.lr_patience == 3
    assert obj.trainer_config.lr_factor == 0.2
    assert obj.trainer_config.weight_decay == 0.01
    assert obj.trainer_config.optimizer_type == "SGD"
    assert obj.trainer_config.optimizer_kwargs == {"momentum": 0.5}
    assert obj.random_state is None


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
