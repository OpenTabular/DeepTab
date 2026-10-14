"""Monitoring configuration boundaries across real estimator fits."""

from copy import deepcopy
from typing import Any

import numpy as np
import pytest
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau

from deeptab.configs import MLPConfig, PreprocessingConfig, TrainerConfig
from deeptab.models import MLPLSS, MLPClassifier, MLPRegressor

pytestmark = pytest.mark.integration


def _case(kind, tmp_path, **config_options):
    features = np.random.default_rng(17).normal(size=(32, 3)).astype(np.float32)
    targets = features[:, 0] if kind != "classifier" else np.arange(32) % 3
    estimator = {"regressor": MLPRegressor, "classifier": MLPClassifier, "lss": MLPLSS}[kind]
    config = TrainerConfig(max_epochs=1, batch_size=8, checkpoint_path=str(tmp_path), **config_options)
    model = estimator(
        model_config=MLPConfig(layer_sizes=[8]),
        preprocessing_config=PreprocessingConfig(numerical_method="standardization"),
        trainer_config=config,
        random_state=17,
    )
    options: dict[str, Any] = {
        "X_val": features[24:],
        "y_val": targets[24:],
        "accelerator": "cpu",
        "logger": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    if kind == "lss":
        options["family"] = "normal"
    return model, features[:24], targets[:24], options


@pytest.mark.parametrize("kind", ["regressor", "classifier", "lss"])
@pytest.mark.parametrize("continuation", [False, True])
@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "mode",
        "scheduler_missing",
        "scheduler_mode",
        "step_validation",
        "validation_cadence",
        "validation_disabled",
    ],
)
def test_invalid_monitoring_fails_before_task_mutation(kind, continuation, invalid, tmp_path):
    model, features, targets, options = _case(kind, tmp_path)
    if continuation:
        model.fit(features, targets, **options)
    task, data, trainer = model._task_model, model._data_module, model._trainer
    fitted = getattr(model, "is_fitted_", False)
    classes = getattr(model, "classes_", None)
    if invalid == "missing":
        options["monitor"] = "val_missing"
    elif invalid == "mode":
        options["mode"] = "sideways"
    elif invalid == "scheduler_missing":
        model.trainer_config.scheduler_monitor = "val_missing"
    elif invalid == "scheduler_mode":
        model.trainer_config.scheduler_kwargs = {"mode": "sideways"}
    elif invalid == "step_validation":
        model.trainer_config.scheduler_interval = "step"
    elif invalid == "validation_cadence":
        options["check_val_every_n_epoch"] = 2
    else:
        options["limit_val_batches"] = 0
    with pytest.raises(ValueError, match=r"monitor|mode|validation|epoch"):
        model.fit(features, targets, rebuild=not continuation, **options)
    assert model._task_model is task and model._data_module is data and model._trainer is trainer
    assert getattr(model, "is_fitted_", False) == fitted
    np.testing.assert_equal(getattr(model, "classes_", None), classes)


def _target_mean(targets, predictions):
    return float(np.mean(targets))


@pytest.mark.parametrize("kind", ["regressor", "classifier", "lss"])
@pytest.mark.parametrize("continuation", [False, True])
@pytest.mark.parametrize(
    "config_options,overrides,callback_expected,scheduler_expected",
    [
        ({}, {}, ("val_loss", "min"), ("val_loss", "min")),
        ({"monitor": "train_loss", "mode": "max"}, {}, ("train_loss", "max"), ("train_loss", "max")),
        ({}, {"monitor": "train_loss", "mode": "max"}, ("train_loss", "max"), ("train_loss", "max")),
        (
            {"scheduler_monitor": "val_loss", "scheduler_kwargs": {"mode": "min"}},
            {"monitor": "train_loss", "mode": "max"},
            ("train_loss", "max"),
            ("val_loss", "min"),
        ),
        (
            {"scheduler_monitor": "val_loss"},
            {"monitor": "train_loss", "mode": "max"},
            ("train_loss", "max"),
            ("val_loss", "max"),
        ),
        (
            {"scheduler_kwargs": {"mode": "min"}},
            {"monitor": "train_loss", "mode": "max"},
            ("train_loss", "max"),
            ("train_loss", "min"),
        ),
        ({}, {"monitor": "val_target_mean", "mode": "max"}, ("val_target_mean", "max"), ("val_target_mean", "max")),
        (
            {},
            {"monitor": "train_target_mean", "mode": "max", "val_check_interval": 0.5},
            ("train_target_mean", "max"),
            ("train_target_mean", "max"),
        ),
        (
            {"scheduler_type": None, "scheduler_monitor": "unused"},
            {"monitor": "train_loss", "mode": "max"},
            ("train_loss", "max"),
            None,
        ),
        ({"scheduler_type": "StepLR", "scheduler_kwargs": {"step_size": 1}}, {}, ("val_loss", "min"), None),
        (
            {"scheduler_frequency": 2},
            {"check_val_every_n_epoch": 2, "max_epochs": 2},
            ("val_loss", "min"),
            ("val_loss", "min"),
        ),
    ],
    ids=[
        "defaults",
        "constructor",
        "fit",
        "both_scheduler_overrides",
        "scheduler_monitor_only",
        "scheduler_mode_only",
        "custom_validation",
        "custom_train",
        "disabled",
        "step_lr",
        "aligned_validation_cadence",
    ],
)
def test_effective_monitoring_policy(
    kind, continuation, config_options, overrides, callback_expected, scheduler_expected, tmp_path
):
    model, features, targets, options = _case(kind, tmp_path, **config_options)
    config = model.trainer_config
    original = deepcopy(vars(config))
    metrics = {"target_mean": _target_mean}
    if continuation:
        model.fit(features, targets, train_metrics=metrics, val_metrics=metrics, **options)
    else:
        options.update(train_metrics=metrics, val_metrics=metrics)
    model.fit(features, targets, rebuild=not continuation, **options, **overrides)
    _assert_monitoring(model, callback_expected, scheduler_expected)
    trainer = model._trainer
    assert isinstance(trainer, Trainer)
    assert float(trainer.callback_metrics["train_target_mean"]) == pytest.approx(float(np.mean(targets)))
    assert float(trainer.callback_metrics["val_target_mean"]) == pytest.approx(float(np.mean(options["y_val"])))
    assert vars(config) == original


def _assert_monitoring(model, callback_expected, scheduler_expected):
    trainer = model._trainer
    assert isinstance(trainer, Trainer)
    callbacks = [
        callback for callback in vars(trainer)["callbacks"] if isinstance(callback, (EarlyStopping, ModelCheckpoint))
    ]
    assert len(callbacks) == 2
    assert all((callback.monitor, callback.mode) == callback_expected for callback in callbacks)
    assert callback_expected[0] in trainer.callback_metrics
    checkpoint = next(callback for callback in callbacks if isinstance(callback, ModelCheckpoint))
    assert checkpoint.best_model_path
    if scheduler_expected is not None:
        scheduler_config = trainer.lr_scheduler_configs[0]
        assert scheduler_config.monitor == scheduler_expected[0]
        assert isinstance(scheduler_config.scheduler, ReduceLROnPlateau)
        assert scheduler_config.scheduler.mode == scheduler_expected[1]
        assert scheduler_config.scheduler.best == pytest.approx(float(trainer.callback_metrics[scheduler_expected[0]]))
    elif model.trainer_config.scheduler_type is None:
        assert trainer.lr_scheduler_configs == []
    else:
        scheduler_config = trainer.lr_scheduler_configs[0]
        assert scheduler_config.monitor is None
        assert not isinstance(scheduler_config.scheduler, ReduceLROnPlateau)


@pytest.mark.parametrize("kind", ["regressor", "classifier", "lss"])
def test_loaded_continuation_resolves_current_fit_overrides(kind, tmp_path):
    model, features, targets, options = _case(kind, tmp_path)
    model.fit(features, targets, **options)
    path = str(tmp_path / "model.zip")
    model.save(path)
    loaded = type(model).load(path)
    loaded.fit(features, targets, rebuild=False, monitor="train_loss", mode="max", **options)
    _assert_monitoring(loaded, ("train_loss", "max"), ("train_loss", "max"))


@pytest.mark.parametrize("kind", ["regressor", "classifier", "lss"])
def test_continuation_does_not_retain_previous_fit_overrides(kind, tmp_path):
    model, features, targets, options = _case(kind, tmp_path)
    model.fit(features, targets, monitor="train_loss", mode="max", **options)
    model.fit(features, targets, rebuild=False, **options)
    _assert_monitoring(model, ("val_loss", "min"), ("val_loss", "min"))


@pytest.mark.parametrize("kind", ["regressor", "classifier", "lss"])
def test_bad_monitor_rebuild_preserves_fitted_model(kind, tmp_path):
    model, features, targets, options = _case(kind, tmp_path)
    model.fit(features, targets, **options)
    task, data, trainer = model._task_model, model._data_module, model._trainer
    classes = getattr(model, "classes_", None)
    predictions = model.predict(features)
    invalid_options = {**options, "y_val": options["y_val"] + 10}
    with pytest.raises(ValueError, match="val_missing"):
        model.fit(features, targets + 10, monitor="val_missing", rebuild=True, **invalid_options)
    assert model._task_model is task and model._data_module is data and model._trainer is trainer
    assert model.is_fitted_
    np.testing.assert_equal(getattr(model, "classes_", None), classes)
    np.testing.assert_array_equal(model.predict(features), predictions)


class _BatchLosses(Callback):
    def __init__(self):
        self.values = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        value = outputs["loss"] if isinstance(outputs, dict) else outputs
        assert isinstance(value, Tensor)
        self.values.append(float(value.detach()))


@pytest.mark.parametrize("kind", ["regressor", "classifier", "lss"])
@pytest.mark.parametrize("mode", ["min", "max"])
def test_step_plateau_uses_available_batch_loss(kind, mode, tmp_path):
    model, features, targets, options = _case(kind, tmp_path, scheduler_interval="step")
    capture = _BatchLosses()
    model.fit(features, targets, monitor="train_loss", mode=mode, callbacks=[capture], **options)
    trainer = model._trainer
    assert isinstance(trainer, Trainer)
    scheduler_config = trainer.lr_scheduler_configs[0]
    assert scheduler_config.monitor == "train_loss"
    scheduler = scheduler_config.scheduler
    assert isinstance(scheduler, ReduceLROnPlateau)
    assert scheduler.mode == mode and scheduler.last_epoch == len(capture.values)
    expected = min(capture.values) if mode == "min" else max(capture.values)
    assert scheduler.best == pytest.approx(expected)


@pytest.mark.parametrize("kind", ["regressor", "classifier", "lss"])
def test_monitor_overrides_without_constructor_trainer_config(kind, tmp_path):
    model, features, targets, options = _case(kind, tmp_path)
    model.trainer_config = None
    model.fit(
        features,
        targets,
        max_epochs=1,
        batch_size=8,
        checkpoint_path=str(tmp_path),
        monitor="train_loss",
        mode="max",
        **options,
    )
    _assert_monitoring(model, ("train_loss", "max"), ("train_loss", "max"))
    assert model.trainer_config is None


@pytest.mark.parametrize("validation_frequency", [0, -1, None, False, 0.5])
def test_invalid_validation_frequency_fails_before_mutation(validation_frequency, tmp_path):
    model, features, targets, options = _case("regressor", tmp_path)
    with pytest.raises(ValueError, match="positive integer check_val_every_n_epoch"):
        model.fit(features, targets, check_val_every_n_epoch=validation_frequency, **options)
    assert model._task_model is None and model._data_module is None and model._trainer is None
