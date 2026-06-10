"""Tests for the Phase 3 dependency-inversion layer.

Verifies:
1. ``IDataModule`` / ``ITaskModel`` Protocol conformance of the concrete classes.
2. ``IDataModuleFactory`` / ``ITaskModelFactory`` conformance of the default factories.
3. ``SklearnBase`` stores injected factories and uses them in ``_build_model``.
4. Replacing the factory with a test double works end-to-end (factory call
   is intercepted without a real model being built).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from deeptab.configs import TrainerConfig
from deeptab.core.default_factories import DefaultDataModuleFactory, DefaultTaskModelFactory
from deeptab.core.interfaces import IDataModule, IDataModuleFactory, ITaskModel, ITaskModelFactory
from deeptab.data.datamodule import TabularDataModule
from deeptab.models.mlp import MLPClassifier
from deeptab.training import TaskModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAST_TRAINER = TrainerConfig(max_epochs=2, patience=2, lr_patience=2)


# ---------------------------------------------------------------------------
# 1. Protocol conformance — concrete classes
# ---------------------------------------------------------------------------


class TestConcreteProtocolConformance:
    """Verify that the production classes satisfy the runtime-checkable Protocols."""

    def test_tabular_data_module_satisfies_idatamodule(self, tmp_path):
        """TabularDataModule is a structural subtype of IDataModule."""
        from pretab.preprocessor import Preprocessor

        dm = TabularDataModule(
            preprocessor=Preprocessor(),
            batch_size=32,
            shuffle=False,
            regression=False,
        )
        assert isinstance(dm, IDataModule)

    def test_task_model_satisfies_itaskmodel(self):
        """TaskModel has all interface members (verified structurally)."""
        # ITaskModel has a data-member (estimator) which prevents issubclass().
        # 'estimator' is set in __init__ (instance attr), so only verify methods here.
        for method in ("train", "eval", "load_state_dict", "parameters"):
            assert hasattr(TaskModel, method), f"TaskModel is missing method '{method}'"


# ---------------------------------------------------------------------------
# 2. Protocol conformance — default factories
# ---------------------------------------------------------------------------


class TestDefaultFactoryConformance:
    """Verify that the default factories satisfy their factory Protocols."""

    def test_default_data_module_factory_satisfies_protocol(self):
        assert isinstance(DefaultDataModuleFactory(), IDataModuleFactory)

    def test_default_task_model_factory_satisfies_protocol(self):
        assert isinstance(DefaultTaskModelFactory(), ITaskModelFactory)


# ---------------------------------------------------------------------------
# 3. SklearnBase stores injected factories
# ---------------------------------------------------------------------------


class TestFactoryInjection:
    """SklearnBase stores the factories; direct attribute assignment replaces them."""

    def test_default_factories_set_when_none_passed(self):
        clf = MLPClassifier(trainer_config=_FAST_TRAINER)
        assert isinstance(clf._data_module_factory, DefaultDataModuleFactory)
        assert isinstance(clf._task_model_factory, DefaultTaskModelFactory)

    def test_custom_data_module_factory_is_stored(self):
        clf = MLPClassifier(trainer_config=_FAST_TRAINER)
        mock_factory = MagicMock(spec=IDataModuleFactory)
        clf._data_module_factory = mock_factory
        assert clf._data_module_factory is mock_factory

    def test_custom_task_model_factory_is_stored(self):
        clf = MLPClassifier(trainer_config=_FAST_TRAINER)
        mock_factory = MagicMock(spec=ITaskModelFactory)
        clf._task_model_factory = mock_factory
        assert clf._task_model_factory is mock_factory

    def test_factories_not_in_get_params(self):
        """Factory kwargs start with '_' and must not leak into get_params()."""
        clf = MLPClassifier(trainer_config=_FAST_TRAINER)
        params = clf.get_params(deep=True)
        assert "_data_module_factory" not in params
        assert "_task_model_factory" not in params

    def test_sklearn_clone_resets_to_default_factories(self):
        """Cloning via sklearn.base.clone always produces fresh default factories."""
        from sklearn.base import clone

        clf = MLPClassifier(trainer_config=_FAST_TRAINER)
        clf._data_module_factory = MagicMock(spec=IDataModuleFactory)
        cloned = clone(clf)
        assert isinstance(cloned._data_module_factory, DefaultDataModuleFactory), (
            "Clone should use DefaultDataModuleFactory, not the replaced mock."
        )


# ---------------------------------------------------------------------------
# 4. Factory replacement smoke test — _build_model calls the factory
# ---------------------------------------------------------------------------


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
