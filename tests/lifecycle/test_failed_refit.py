"""Failed refits must preserve valid state or prevent subsequent prediction."""

import numpy as np
import pandas as pd
import pytest
import torch
from lightning.pytorch import Trainer

from deeptab.configs import MLPConfig, PreprocessingConfig, TrainerConfig
from deeptab.core.exceptions import DataError, DeepTabError, NotFittedError
from deeptab.data.datamodule import TabularDataModule
from deeptab.models import MLPLSS, MLPClassifier, MLPRegressor
from deeptab.training import TaskModel

pytestmark = pytest.mark.integration


def _data():
    features = np.random.default_rng(17).normal(size=(48, 3)).astype(np.float32)
    labels = (features[:, 0] > 0).astype(int)
    return features[:32], labels[:32], features[32:], labels[32:]


def _model(estimator_cls, tmp_path):
    return estimator_cls(
        model_config=MLPConfig(layer_sizes=[8]),
        preprocessing_config=PreprocessingConfig(numerical_method="standardization", output_dim=7),
        trainer_config=TrainerConfig(
            max_epochs=2, batch_size=16, patience=1, lr_patience=1, checkpoint_path=str(tmp_path)
        ),
        random_state=17,
    )


def _fit_options():
    return {"accelerator": "cpu", "logger": False, "enable_progress_bar": False, "enable_model_summary": False}


@pytest.fixture(params=[MLPClassifier, MLPRegressor, MLPLSS])
def fitted_case(request, tmp_path):
    features, labels, validation_features, validation_labels = _data()
    if request.param is not MLPClassifier:
        labels = features[:, 0].copy()
        validation_labels = validation_features[:, 0].copy()
    model = _model(request.param, tmp_path)
    options = _fit_options()
    if request.param is MLPLSS:
        options["family"] = "normal"
    model.fit(features, labels, X_val=validation_features, y_val=validation_labels, **options)
    return model, features, labels, validation_features, validation_labels, options


def _snapshot(model, validation_features):
    predictions = model.predict(validation_features)
    attributes = vars(model).copy()
    weights = {name: tensor.detach().clone() for name, tensor in model._task_model.state_dict().items()}
    return predictions, attributes, weights


def _assert_preserved(model, validation_features, snapshot):
    predictions, attributes, weights = snapshot
    assert model.is_fitted_ is True
    assert vars(model).keys() == attributes.keys()
    assert all(vars(model)[name] is previous for name, previous in attributes.items())
    for name, previous in weights.items():
        torch.testing.assert_close(model._task_model.state_dict()[name], previous, rtol=0, atol=0)
    np.testing.assert_array_equal(model.predict(validation_features), predictions)


def test_unseen_validation_labels_preserve_fitted_classifier(tmp_path):
    features, labels, validation_features, validation_labels = _data()
    model = _model(MLPClassifier, tmp_path)
    model.fit(features, labels, X_val=validation_features, y_val=validation_labels, **_fit_options())
    classes = model.classes_.copy()
    predictions = model.predict(validation_features)
    runtime = (model._task_model, model._data_module, model._preprocessor, model._trainer)

    with pytest.raises(ValueError, match="not seen during fit"):
        model.fit(
            features,
            labels + 10,
            X_val=validation_features,
            y_val=validation_labels + 12,
            **_fit_options(),
        )

    np.testing.assert_array_equal(model.classes_, classes)
    assert model.is_fitted_ is True
    assert all(
        current is previous
        for current, previous in zip(
            (model._task_model, model._data_module, model._preprocessor, model._trainer), runtime, strict=True
        )
    )
    np.testing.assert_array_equal(model.predict(validation_features), predictions)


@pytest.mark.parametrize(
    "invalid",
    [
        "missing_target",
        "nan_target",
        "infinite_target",
        "length_mismatch",
        "missing_validation_target",
        "missing_validation_features",
        "validation_length",
        "validation_infinity",
        "validation_width",
        "feature_vector",
        "empty_features",
        "validation_nan",
        "validation_target_columns",
    ],
)
def test_invalid_inputs_preserve_fitted_estimator(fitted_case, invalid):
    model, features, labels, validation_features, validation_labels, options = fitted_case
    snapshot = _snapshot(model, validation_features)
    targets = labels.astype(float) + 10
    supplied_features = features
    supplied_validation_features = validation_features
    supplied_validation_labels = validation_labels.astype(float) + 10
    if invalid == "missing_target":
        targets = None
    elif invalid == "nan_target":
        targets[0] = np.nan
    elif invalid == "infinite_target":
        targets[0] = np.inf
    elif invalid == "length_mismatch":
        supplied_features = features[:-1]
    elif invalid == "missing_validation_target":
        supplied_validation_labels = None
    elif invalid == "missing_validation_features":
        supplied_validation_features = None
    elif invalid == "validation_length":
        supplied_validation_labels = supplied_validation_labels[:-1]
    elif invalid == "validation_infinity":
        supplied_validation_labels[0] = -np.inf
    elif invalid == "validation_width":
        supplied_validation_features = validation_features[:, :-1]
    elif invalid == "feature_vector":
        supplied_features = features[:, 0]
    elif invalid == "empty_features":
        supplied_features = features[:0]
    elif invalid == "validation_nan":
        supplied_validation_labels[0] = np.nan
    elif invalid == "validation_target_columns":
        supplied_validation_labels = np.column_stack([supplied_validation_labels] * 2)

    with pytest.raises((DataError, ValueError)):
        model.fit(
            supplied_features,
            targets,
            X_val=supplied_validation_features,
            y_val=supplied_validation_labels,
            **options,
        )

    _assert_preserved(model, validation_features, snapshot)


def test_continuous_targets_preserve_fitted_classifier(tmp_path):
    features, labels, validation_features, validation_labels = _data()
    model = _model(MLPClassifier, tmp_path)
    model.fit(features, labels, X_val=validation_features, y_val=validation_labels, **_fit_options())
    snapshot = _snapshot(model, validation_features)
    with pytest.raises(ValueError, match="continuous"):
        model.fit(features, np.linspace(0.1, 0.9, len(features)), **_fit_options())
    _assert_preserved(model, validation_features, snapshot)


@pytest.mark.parametrize("stage", ["preprocessing", "model_construction", "trainer_construction", "training"])
def test_late_failure_invalidates_prediction_and_refit_recovers(fitted_case, monkeypatch, stage):
    model, features, labels, validation_features, validation_labels, options = fitted_case

    def fail(*args, **kwargs):
        raise RuntimeError("injected fit failure")

    with monkeypatch.context() as patcher:
        if stage == "preprocessing":
            patcher.setattr(TabularDataModule, "preprocess_data", fail)
        elif stage == "model_construction":
            patcher.setattr(TaskModel, "__init__", fail)
        elif stage == "trainer_construction":
            patcher.setattr(Trainer, "__init__", fail)
        else:
            patcher.setattr(Trainer, "fit", fail)
        with pytest.raises(RuntimeError, match="injected fit failure"):
            model.fit(features, labels + 10, X_val=validation_features, y_val=validation_labels + 10, **options)

    assert model.is_fitted_ is False
    with pytest.raises(NotFittedError):
        model.predict(validation_features)
    if isinstance(model, MLPClassifier):
        with pytest.raises(NotFittedError):
            model.predict_proba(validation_features)

    model.fit(features, labels + 10, X_val=validation_features, y_val=validation_labels + 10, **options)
    assert model.is_fitted_ is True
    assert np.isfinite(model.predict(validation_features)).all()
    if isinstance(model, MLPClassifier):
        np.testing.assert_array_equal(model.classes_, [10, 11])


@pytest.mark.parametrize(
    "invalid", ["length", "matrix", "negative", "nan", "infinity", "zero", "text", "loss", "class_weight"]
)
def test_invalid_classifier_options_preserve_state(tmp_path, invalid):
    features, labels, validation_features, validation_labels = _data()
    model = _model(MLPClassifier, tmp_path)
    model.fit(features, labels, X_val=validation_features, y_val=validation_labels, **_fit_options())
    snapshot = _snapshot(model, validation_features)
    weights = np.ones(len(features))
    options = {"sample_weight": weights}
    if invalid == "length":
        options["sample_weight"] = weights[:-1]
    elif invalid == "matrix":
        options["sample_weight"] = weights[:, None]
    elif invalid == "negative":
        weights[0] = -1
    elif invalid == "nan":
        weights[0] = np.nan
    elif invalid == "infinity":
        weights[0] = np.inf
    elif invalid == "zero":
        weights[:] = 0
    elif invalid == "text":
        options = {"sample_weight": ["invalid"] * len(features)}
    elif invalid == "loss":
        options = {"loss_fct": "unknown_loss"}
    else:
        options = {"class_weight": [1, 2, 3]}
    with pytest.raises((DeepTabError, ValueError)):
        model.fit(
            features, labels + 10, X_val=validation_features, y_val=validation_labels + 10, **options, **_fit_options()
        )
    _assert_preserved(model, validation_features, snapshot)


@pytest.mark.parametrize("invalid", ["unknown_family", "unknown_option", "invalid_dirichlet"])
def test_invalid_distribution_configuration_preserves_state(tmp_path, invalid):
    features, _, validation_features, _ = _data()
    model = _model(MLPLSS, tmp_path)
    model.fit(
        features,
        features[:, 0],
        X_val=validation_features,
        y_val=validation_features[:, 0],
        family="normal",
        **_fit_options(),
    )
    snapshot = _snapshot(model, validation_features)
    options = {"family": "unknown_family"}
    if invalid == "unknown_option":
        options = {"family": "normal", "distributional_kwargs": {"unknown_option": 3}}
    elif invalid == "invalid_dirichlet":
        options = {"family": "dirichlet"}
    with pytest.raises((DeepTabError, ValueError, TypeError)):
        model.fit(features, features[:, 0], **options, **_fit_options())
    _assert_preserved(model, validation_features, snapshot)


def test_named_validation_schema_preserves_state(tmp_path):
    features, labels, validation_features, validation_labels = _data()
    features = pd.DataFrame(features, columns=pd.Index(["first", "second", "third"]))
    validation_features = pd.DataFrame(validation_features, columns=features.columns)
    model = _model(MLPClassifier, tmp_path)
    model.fit(features, labels, X_val=validation_features, y_val=validation_labels, **_fit_options())
    snapshot = _snapshot(model, validation_features)
    with pytest.raises(DataError):
        model.fit(
            features,
            labels + 10,
            X_val=validation_features.iloc[:, ::-1],
            y_val=validation_labels + 10,
            **_fit_options(),
        )
    _assert_preserved(model, validation_features, snapshot)


def test_dirichlet_refit_preserves_target_dimensions_and_selected_family(tmp_path):
    features, _, validation_features, _ = _data()
    targets = np.random.default_rng(19).dirichlet([2, 3, 4], size=len(features))
    validation_targets = np.random.default_rng(23).dirichlet([2, 3, 4], size=len(validation_features))
    model = _model(MLPLSS, tmp_path)
    model.fit(
        features, targets, X_val=validation_features, y_val=validation_targets, family="dirichlet", **_fit_options()
    )
    snapshot = _snapshot(model, validation_features)
    invalid_targets = targets.copy()
    invalid_targets[0, 1] = np.inf
    with pytest.raises((DataError, ValueError)):
        model.fit(
            features,
            invalid_targets,
            family=None,
            X_val=validation_features,
            y_val=validation_targets,
            **_fit_options(),
        )
    _assert_preserved(model, validation_features, snapshot)
    model.fit(features, targets, family=None, X_val=validation_features, y_val=validation_targets, **_fit_options())
    predictions = model.predict(validation_features)
    assert model.is_fitted_ is True
    assert model.family_name == "dirichlet"
    assert model.distributional_kwargs_["num_classes"] == 3
    assert predictions.shape == validation_targets.shape
    assert np.isfinite(predictions).all()
