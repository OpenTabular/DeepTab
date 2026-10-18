"""Tests for hpo workflows."""

from copy import deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from deeptab.models._mixins.hpo import _HyperparameterMixin


@dataclass
class _SearchConfig:
    lr: float = 0.01
    head_layer_sizes: list[int] = field(default_factory=list)


class _SearchEstimator(_HyperparameterMixin):
    def __init__(self, losses, *, baseline=-2.0, random_state=17, build_failures=()):
        self.config = _SearchConfig()
        self.random_state = random_state
        self.losses = losses
        self.baseline = baseline
        self.build_failures = build_failures
        self.fits: list[dict[str, Any]] = []
        self.builds: list[_SearchConfig] = []
        self._data_module = None
        self._task_model: Any = None
        self._trainer = SimpleNamespace(validate=lambda *args, **kwargs: [{"val_loss": self.loss}])

    def build_model(self, X, y, **kwargs):
        self.builds.append(deepcopy(self.config))
        if self.config.lr in self.build_failures:
            raise ValueError("Invalid trial architecture")
        self._task_model = SimpleNamespace(
            config=deepcopy(self.config),
            epoch_val_loss_at=lambda epoch: self.loss,
            early_pruning_threshold=None,
            pruning_epoch=0,
        )

    def fit(self, X, y, **kwargs):
        if kwargs.get("rebuild", True):
            self.build_model(X, y)
        callbacks = kwargs.get("callbacks") or []
        if not isinstance(callbacks, (list, tuple)):
            callbacks = [callbacks]
        for callback in callbacks:
            callback.on_fit_start(self._trainer, self._task_model)
        self.fits.append(
            {
                "config": deepcopy(self.config),
                "threshold": self._task_model.early_pruning_threshold,
                "kwargs": kwargs,
            }
        )
        outcome = self.losses.get(self.config.lr, self.baseline)
        if isinstance(outcome, Exception):
            raise outcome
        self.loss = outcome
        return self


def test_real_bayesian_optimizer_handles_failed_negative_loss_trials(monkeypatch):
    from skopt.space import Real

    import deeptab.models._mixins.hpo as hpo_module

    outcomes = SimpleNamespace(get=lambda lr, default: ValueError("Trial failed") if lr > 0.15 else -lr)
    model = _SearchEstimator(outcomes)
    monkeypatch.setattr(hpo_module, "get_search_space", lambda *args, **kwargs: (["lr"], [Real(0.1, 0.2)]))
    best = model.optimize_hparams([[1.0]], [1.0], regression=True, time=11)
    assert 0.1 <= best[0] <= 0.15
    assert model.config.lr == best[0]
    assert model._task_model.config == model.config
    assert model.loss < 0


@pytest.mark.parametrize("option_source", ["explicit", "trainer_config"])
@pytest.mark.parametrize("task", ["regression", "classification", "lss"])
def test_public_search_refits_and_predicts_without_another_fit(monkeypatch, tmp_path, task, option_source):
    from skopt.space import Categorical

    import deeptab.models._mixins.hpo as hpo_module
    from deeptab.configs import MLPConfig, PreprocessingConfig, TrainerConfig
    from deeptab.models import MLPLSS, MLPClassifier, MLPRegressor

    estimator_class = {"regression": MLPRegressor, "classification": MLPClassifier, "lss": MLPLSS}[task]
    model = estimator_class(
        model_config=MLPConfig(d_model=16, dropout=0.0),
        preprocessing_config=PreprocessingConfig(numerical_method="standardization", output_dim=7),
        trainer_config=TrainerConfig(max_epochs=2, patience=1, lr_patience=1, batch_size=16),
        random_state=73,
    )
    random = np.random.default_rng(12)
    features = random.normal(size=(32, 4)).astype(np.float32)
    targets = np.array([0] * 18 + [1] * 6 + [0] * 4 + [1] * 4) if task == "classification" else features[:, 0]
    monkeypatch.setattr(hpo_module, "get_search_space", lambda *args, **kwargs: (["d_model"], [Categorical([16, 32])]))
    observed: dict[str, Any] = {}
    snapshots: list[dict[str, Any]] = []
    original_fit = model.fit

    def fit(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        task_model = model._task_model
        loader = model._data_module.train_dataloader()
        loss = task_model.family if task == "lss" else task_model.loss_fct
        snapshots.append(
            {
                "batch_size": loader.batch_size,
                "loss_type": type(loss),
                "loss_state": deepcopy(loss.state_dict()),
                "weighted_sampler": isinstance(loader.sampler, WeightedRandomSampler),
                "drop_last": loader.drop_last,
                "lr": task_model.lr,
                "threshold": task_model.early_pruning_threshold,
            }
        )
        return result

    monkeypatch.setattr(model, "fit", fit)

    def minimize(objective, space, **kwargs):
        trials = [[16], [32]]
        scores = [objective(trial) for trial in trials]
        observed["winner"] = trials[int(np.argmin(scores))]
        observed["seed"] = kwargs["random_state"]
        observed["trial_task"] = model._task_model
        return SimpleNamespace(x=observed["winner"])

    monkeypatch.setattr(hpo_module, "gp_minimize", minimize)
    fit_options: dict[str, Any] = {
        "max_epochs": 1,
        "prune_epoch": 0,
        "accelerator": "cpu",
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "default_root_dir": str(tmp_path),
    }
    if task == "lss":
        fit_options["family"] = "normal"
    if option_source == "explicit":
        fit_options.update(
            batch_size=8, lr=0.003, shuffle=False, dataloader_kwargs={"num_workers": 0, "drop_last": True}
        )
    if task == "classification":
        fit_options.update(class_weight="balanced", balanced_sampler=True)
        if option_source == "explicit":
            fit_options["loss_fct"] = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))
    best = model.optimize_hparams(
        features[:24], targets[:24], X_val=features[24:], y_val=targets[24:], time=2, **fit_options
    )
    assert best == observed["winner"]
    assert model.config.d_model == best[0]
    assert model._task_model is not observed["trial_task"]
    assert model._task_model.early_pruning_threshold is None
    assert len(model._task_model.val_losses) == 1
    assert observed["seed"] == 73
    assert len(snapshots) == 4
    for snapshot in snapshots:
        assert snapshot["batch_size"] == (8 if option_source == "explicit" else 16)
        assert snapshot["loss_type"] is snapshots[0]["loss_type"]
        assert snapshot["weighted_sampler"] == (task == "classification")
        assert snapshot["drop_last"] == (option_source == "explicit")
        if option_source == "explicit":
            assert snapshot["lr"] == 0.003
        assert snapshot["loss_state"].keys() == snapshots[0]["loss_state"].keys()
        for name, value in snapshot["loss_state"].items():
            torch.testing.assert_close(value, snapshots[0]["loss_state"][name])
    assert snapshots[0]["threshold"] is None
    assert snapshots[-1]["threshold"] is None
    assert all(snapshot["threshold"] is not None for snapshot in snapshots[1:-1])
    predictions = model.predict(features[24:])
    assert predictions.shape[0] == 8
    assert np.isfinite(predictions).all()
    if task == "lss":
        assert model.family_name == "normal"
