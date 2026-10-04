"""Tests for factory contracts."""

from __future__ import annotations

from deeptab.configs import TrainerConfig
from deeptab.core.default_factories import DefaultDataModuleFactory, DefaultTaskModelFactory
from deeptab.core.interfaces import IDataModule, IDataModuleFactory, ITaskModelFactory
from deeptab.data.datamodule import TabularDataModule
from deeptab.training import TaskModel

_FAST_TRAINER = TrainerConfig(max_epochs=2, patience=2, lr_patience=2)


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


class TestDefaultFactoryConformance:
    """Verify that the default factories satisfy their factory Protocols."""

    def test_default_data_module_factory_satisfies_protocol(self):
        assert isinstance(DefaultDataModuleFactory(), IDataModuleFactory)

    def test_default_task_model_factory_satisfies_protocol(self):
        assert isinstance(DefaultTaskModelFactory(), ITaskModelFactory)
