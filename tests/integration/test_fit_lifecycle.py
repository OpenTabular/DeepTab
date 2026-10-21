"""Tests for fit lifecycle."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from lightning.pytorch import Trainer as LightningTrainer
from lightning.pytorch.callbacks import Callback, ModelSummary

from deeptab.core.exceptions import NotFittedError
from deeptab.models import MLPLSS, MLPRegressor

RANDOM_STATE = 0

FIT_KWARGS: dict[str, Any] = {"max_epochs": 1, "batch_size": 32}

N = 120

N_FEATURES = 4


def _make_reg_data():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N, N_FEATURES))
    y = rng.standard_normal(N)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)]), y  # type: ignore[call-overload]


class TestFailedFitLeavesIsFittedFalse:
    def test_failed_second_fit_sets_is_fitted_false(self, monkeypatch):
        X, y = _make_reg_data()
        reg = MLPRegressor()
        reg.fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)
        assert reg.is_fitted_ is True

        def _raise_mid_fit(self, *args, **kwargs):
            raise RuntimeError("simulated mid-fit crash")

        monkeypatch.setattr(LightningTrainer, "fit", _raise_mid_fit)

        with pytest.raises(RuntimeError, match="simulated mid-fit crash"):
            reg.fit(X, y, random_state=RANDOM_STATE + 1, **FIT_KWARGS)

        assert reg.is_fitted_ is False

    def test_predict_raises_not_fitted_after_failed_refit(self, monkeypatch):
        X, y = _make_reg_data()
        reg = MLPRegressor()
        reg.fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)

        def _raise_mid_fit(self, *args, **kwargs):
            raise RuntimeError("simulated mid-fit crash")

        monkeypatch.setattr(LightningTrainer, "fit", _raise_mid_fit)
        with pytest.raises(RuntimeError):
            reg.fit(X, y, random_state=RANDOM_STATE + 1, **FIT_KWARGS)

        with pytest.raises(NotFittedError):
            reg.predict(X)


class _RecordingCallback(Callback):
    def __init__(self):
        self.epoch_ends = 0

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_ends += 1


class TestFitAcceptsCustomCallbacks:
    def test_callback_list_is_accepted_and_runs(self):
        X, y = _make_reg_data()
        reg = MLPRegressor()
        cb = _RecordingCallback()
        reg.fit(X, y, random_state=RANDOM_STATE, callbacks=[cb], **FIT_KWARGS)
        assert cb.epoch_ends > 0, "Custom callback's on_train_epoch_end must have fired"

    def test_single_callback_instance_is_accepted(self):
        X, y = _make_reg_data()
        reg = MLPRegressor()
        cb = _RecordingCallback()
        reg.fit(X, y, random_state=RANDOM_STATE, callbacks=cb, **FIT_KWARGS)
        assert cb.epoch_ends > 0

    def test_builtin_callbacks_still_present(self):
        """A custom callbacks= must augment, not replace, the built-in ones."""
        X, y = _make_reg_data()
        reg = MLPRegressor()
        cb = _RecordingCallback()
        reg.fit(X, y, random_state=RANDOM_STATE, callbacks=[cb], **FIT_KWARGS)
        assert reg._best_model_path, "ModelCheckpoint callback must still have produced a checkpoint"


def _summary_callbacks(model):
    return [callback for callback in model._trainer.callbacks if isinstance(callback, ModelSummary)]


@pytest.mark.parametrize(
    "model_class, extra", [(MLPRegressor, {}), (MLPLSS, {"family": "normal"})], ids=["regressor", "lss"]
)
class TestModelSummaryFollowsTrainerFlag:
    def test_summary_is_added_by_default(self, model_class, extra):
        X, y = _make_reg_data()
        model = model_class().fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS, **extra)
        assert [callback._max_depth for callback in _summary_callbacks(model)] == [2]

    def test_enable_model_summary_false_disables_summary(self, model_class, extra, capsys):
        X, y = _make_reg_data()
        model = model_class().fit(
            X,
            y,
            random_state=RANDOM_STATE,
            enable_model_summary=False,
            enable_progress_bar=False,
            **FIT_KWARGS,
            **extra,
        )
        assert _summary_callbacks(model) == []
        assert "Trainable params" not in capsys.readouterr().out

    def test_user_summary_callback_replaces_builtin(self, model_class, extra):
        X, y = _make_reg_data()
        summary = ModelSummary(max_depth=1)
        model = model_class().fit(X, y, random_state=RANDOM_STATE, callbacks=[summary], **FIT_KWARGS, **extra)
        assert _summary_callbacks(model) == [summary]
