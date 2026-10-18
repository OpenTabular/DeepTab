"""Tests for factory wiring."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from deeptab.configs import TrainerConfig
from deeptab.core.default_factories import DefaultDataModuleFactory, DefaultTaskModelFactory
from deeptab.models.mlp import MLPClassifier

_FAST_TRAINER = TrainerConfig(max_epochs=2, patience=2, lr_patience=2)


class TestFactoryReplacementSmoke:
    """Verify _build_model delegates to the injected factories."""

    def test_data_module_factory_called_during_build(self):
        """A spy factory confirms _data_module_factory.create() is called during fit."""
        clf = MLPClassifier(trainer_config=_FAST_TRAINER)
        spy = MagicMock(wraps=DefaultDataModuleFactory())
        clf._data_module_factory = spy

        X = np.random.default_rng(0).standard_normal((50, 4))
        y = np.array([0, 1] * 25)

        clf.fit(X, y)

        spy.create.assert_called_once()
        call_kwargs = spy.create.call_args.kwargs
        assert "preprocessor" in call_kwargs
        assert call_kwargs["batch_size"] == _FAST_TRAINER.batch_size
        assert call_kwargs["regression"] is False

    def test_task_model_factory_called_during_build(self):
        """A spy factory confirms _task_model_factory.create() is called during fit."""
        clf = MLPClassifier(trainer_config=_FAST_TRAINER)
        spy = MagicMock(wraps=DefaultTaskModelFactory())
        clf._task_model_factory = spy

        X = np.random.default_rng(0).standard_normal((50, 4))
        y = np.array([0, 1] * 25)

        clf.fit(X, y)

        spy.create.assert_called_once()
        call_kwargs = spy.create.call_args.kwargs
        assert "model_class" in call_kwargs
        assert "config" in call_kwargs
        assert "feature_information" in call_kwargs
