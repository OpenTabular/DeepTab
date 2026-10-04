"""Tests for search driver."""

from copy import deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from lightning.pytorch.callbacks import Callback

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


@pytest.fixture
def search_driver(monkeypatch):
    import deeptab.models._mixins.hpo as hpo_module

    def run(model, trials, *, names=None, **kwargs):
        names = ["lr"] if names is None else names
        monkeypatch.setattr(hpo_module, "get_search_space", lambda *args, **kwargs: (names, names))
        observed: dict[str, Any] = {}

        def minimize(objective, space, **options):
            observed.update(options)
            observed["scores"] = [objective(trial) for trial in trials]
            return SimpleNamespace(x=trials[int(np.argmin(observed["scores"]))])

        monkeypatch.setattr(hpo_module, "gp_minimize", minimize)
        result = model.optimize_hparams([[1.0]], [1.0], regression=True, time=len(trials), **kwargs)
        return result, observed

    return run


@pytest.mark.parametrize("baseline", [-2.0, 0.0, 2.0])
@pytest.mark.parametrize("prune_by_epoch", [True, False])
def test_search_failure_penalty_and_pruning_are_sign_safe(search_driver, baseline, prune_by_epoch):
    model = _SearchEstimator({0.1: baseline - 1, 0.2: ValueError("Trial failed")}, baseline=baseline)
    result, observed = search_driver(model, [[0.1], [0.2]], prune_by_epoch=prune_by_epoch)
    assert result == [0.1]
    assert np.isfinite(observed["scores"][1])
    assert observed["scores"][1] > observed["scores"][0]
    assert model.fits[1]["threshold"] > baseline


def test_search_catches_model_construction_failure(search_driver):
    model = _SearchEstimator({0.1: -3.0}, build_failures=(0.2,))
    result, observed = search_driver(model, [[0.1], [0.2]])
    assert result == [0.1]
    assert observed["scores"][1] > observed["scores"][0]


def test_search_explicitly_rebuilds_baseline_trials_and_winner(search_driver):
    model = _SearchEstimator({0.1: -3.0})
    search_driver(model, [[0.1]], rebuild=False)
    assert len(model.fits) == 3
    assert all(fit["kwargs"].get("rebuild") is True for fit in model.fits)


@pytest.mark.parametrize("length", [1, 3, 5])
def test_search_applies_same_head_shape_and_refits_winner(search_driver, length):
    model = _SearchEstimator({0.1: -4.0, 0.2: -3.0})
    names = ["lr", "head_layer_size_length", *[f"head_layer_size_{index}" for index in range(1, 6)]]
    winner = [0.1, np.int64(length), 17, 33, 49, 65, 81]
    result, _ = search_driver(model, [winner, [0.2, 5, 128, 128, 128, 128, 128]], names=names)
    expected = [16, 32, 48, 64, 80][:length]
    assert result == winner
    assert model.config.head_layer_sizes == expected
    assert model.builds[1].head_layer_sizes == expected
    assert model._task_model.config == model.config
    assert len(model.fits) == 4
    assert model.fits[-1]["threshold"] is None
    assert model.fits[-1]["kwargs"].get("rebuild", True)


@pytest.mark.parametrize("seed", [None, 0, 73])
def test_search_honors_estimator_seed(search_driver, seed):
    model = _SearchEstimator({0.1: -3.0}, random_state=seed)
    _, observed = search_driver(model, [[0.1]])
    assert observed["random_state"] == seed


@pytest.mark.parametrize("container", ["single", "list", "tuple"])
def test_search_preserves_user_callbacks_without_leaking_trial_pruning(search_driver, container):
    class _UserCallback(Callback):
        def __init__(self):
            self.calls = 0

        def on_fit_start(self, trainer, pl_module):
            self.calls += 1

    callback = _UserCallback()
    supplied = callback if container == "single" else [callback] if container == "list" else (callback,)
    model = _SearchEstimator({0.1: -3.0, 0.2: -2.5})
    search_driver(model, [[0.1], [0.2]], callbacks=supplied, prune_epoch=2)
    assert callback.calls == 4
    assert model.fits[0]["kwargs"]["callbacks"] is supplied
    assert model.fits[-1]["kwargs"]["callbacks"] is supplied
    for fit in model.fits[1:-1]:
        assert fit["kwargs"]["callbacks"][0] is callback
        assert len(fit["kwargs"]["callbacks"]) == 2
        assert fit["kwargs"]["callbacks"][1].epoch == 2
        assert fit["threshold"] is not None
    assert model.fits[-1]["threshold"] is None
    if isinstance(supplied, (list, tuple)):
        assert len(supplied) == 1


def test_search_with_no_successful_trials_raises(search_driver):
    model = _SearchEstimator({0.1: ValueError("Trial failed")})
    with pytest.raises(RuntimeError, match="No hyperparameter trial completed successfully"):
        search_driver(model, [[0.1]])


def test_search_nonfinite_trial_loss_cannot_win(search_driver):
    model = _SearchEstimator({0.1: -3.0, 0.2: float("nan")})
    result, observed = search_driver(model, [[0.1], [0.2]])
    assert result == [0.1]
    assert all(np.isfinite(observed["scores"]))


def test_search_failure_penalty_remains_finite_for_large_losses(search_driver):
    model = _SearchEstimator({0.1: 1e308, 0.2: ValueError("Trial failed")}, baseline=1e308)
    result, observed = search_driver(model, [[0.1], [0.2]])
    assert result == [0.1]
    assert np.isfinite(observed["scores"][1])
    assert observed["scores"][1] > observed["scores"][0]


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), -float("inf")])
def test_search_rejects_nonfinite_baseline(search_driver, loss):
    model = _SearchEstimator({0.1: -3.0}, baseline=loss)
    with pytest.raises(ValueError, match="Baseline validation loss must be finite"):
        search_driver(model, [[0.1]])
