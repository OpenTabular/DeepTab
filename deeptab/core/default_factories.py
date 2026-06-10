"""Default factory implementations for ``SklearnBase``'s two core collaborators.

These are the production factories used unless a caller replaces them via
direct attribute assignment on an estimator instance::

    clf._data_module_factory = MyDataModuleFactory()

This module is the **only** place in the library that directly imports
``TabularDataModule`` and ``TaskModel``.  All other code depends on the
``IDataModule`` / ``ITaskModel`` Protocols defined in
:mod:`deeptab.core.interfaces`.
"""

from __future__ import annotations

from typing import Any

from deeptab.core.interfaces import IDataModule, ITaskModel
from deeptab.data.datamodule import TabularDataModule
from deeptab.training import TaskModel


class DefaultDataModuleFactory:
    """Production factory for :class:`~deeptab.data.datamodule.TabularDataModule`.

    Used by ``SklearnBase`` unless replaced with a custom implementation.
    Forwards all arguments verbatim to ``TabularDataModule.__init__``.
    """

    def create(
        self,
        preprocessor: Any,
        batch_size: int,
        shuffle: bool,
        regression: bool,
        **kwargs: Any,
    ) -> IDataModule:
        """Construct a ``TabularDataModule``.

        Parameters
        ----------
        preprocessor :
            Fitted or unfitted ``Preprocessor`` instance.
        batch_size : int
            Mini-batch size for the DataLoader.
        shuffle : bool
            Whether to shuffle training samples each epoch.
        regression : bool
            ``True`` for regression tasks, ``False`` for classification.
        **kwargs
            Additional arguments forwarded to ``TabularDataModule``
            (e.g. ``val_size``, ``sampler``, ``random_state``).

        Returns
        -------
        TabularDataModule
        """
        return TabularDataModule(
            preprocessor=preprocessor,
            batch_size=batch_size,
            shuffle=shuffle,
            regression=regression,
            **kwargs,
        )


class DefaultTaskModelFactory:
    """Production factory for :class:`~deeptab.training.TaskModel`.

    Used by ``SklearnBase`` unless replaced with a custom implementation.
    Forwards all arguments verbatim to ``TaskModel.__init__``.
    """

    def create(
        self,
        model_class: Any,
        config: Any,
        feature_information: tuple[dict, dict, dict],
        **kwargs: Any,
    ) -> ITaskModel:
        """Construct a ``TaskModel``.

        Parameters
        ----------
        model_class :
            The backbone ``nn.Module`` class (not an instance).
        config :
            Config dataclass instance for the backbone architecture.
        feature_information : (num_info, cat_info, emb_info)
            Tuple of three dicts describing the feature schema, as produced
            by ``TabularDataModule`` after ``preprocess_data``.
        **kwargs
            Additional arguments forwarded to ``TaskModel``
            (e.g. ``lr``, ``optimizer_type``, ``loss_fct``).

        Returns
        -------
        TaskModel
        """
        return TaskModel(
            model_class=model_class,
            config=config,
            feature_information=feature_information,
            **kwargs,
        )
