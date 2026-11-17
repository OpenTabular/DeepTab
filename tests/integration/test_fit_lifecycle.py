"""Tests for fit lifecycle."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from lightning.pytorch import Trainer as LightningTrainer
from lightning.pytorch.callbacks import Callback

from deeptab.core.exceptions import NotFittedError
from deeptab.models import MLPRegressor

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


class TestRegressorScoreUsesCurrentModel:
    """score() must evaluate the in-memory model, like predict(), without
    reloading the training checkpoint from disk."""

    def test_score_works_after_checkpoint_directory_is_removed(self, tmp_path):
        import shutil

        from sklearn.metrics import r2_score

        X, y = _make_reg_data()
        checkpoint_dir = tmp_path / "checkpoints"
        reg = MLPRegressor().fit(X, y, random_state=RANDOM_STATE, checkpoint_path=str(checkpoint_dir), **FIT_KWARGS)
        assert reg._best_model_path
        shutil.rmtree(checkpoint_dir)
        assert reg.score(X, y) == pytest.approx(r2_score(y, reg.predict(X)))

    def test_score_does_not_replace_current_weights(self):
        import torch
        from sklearn.metrics import r2_score

        X, y = _make_reg_data()
        reg = MLPRegressor().fit(X, y, random_state=RANDOM_STATE, **FIT_KWARGS)
        assert reg._task_model is not None
        with torch.no_grad():
            for parameter in reg._task_model.parameters():
                parameter.add_(0.5)
        predictions = reg.predict(X)
        assert reg.score(X, y) == pytest.approx(r2_score(y, predictions))
        np.testing.assert_array_equal(reg.predict(X), predictions)


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
