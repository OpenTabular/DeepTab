"""Tests for factory injection."""

from __future__ import annotations

from unittest.mock import MagicMock

from deeptab.configs import TrainerConfig
from deeptab.core.default_factories import DefaultDataModuleFactory, DefaultTaskModelFactory
from deeptab.core.interfaces import IDataModuleFactory, ITaskModelFactory
from deeptab.models.mlp import MLPClassifier

_FAST_TRAINER = TrainerConfig(max_epochs=2, patience=2, lr_patience=2)


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
        assert isinstance(cloned._data_module_factory, DefaultDataModuleFactory), (  # type: ignore[union-attr]
            "Clone should use DefaultDataModuleFactory, not the replaced mock."
        )
