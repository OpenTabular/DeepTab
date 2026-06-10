"""Save and load logic for all DeepTab estimators.

The :meth:`save` / :meth:`load` pair is the canonical persistence
mechanism.  Standard :mod:`pickle` is intentionally **not** supported:
``__getstate__`` clears ``task_model`` to avoid serialising Lightning
modules, so a pickled estimator cannot make predictions after
unpickling.  Use :meth:`save` / :meth:`load` for all persistence needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as pl
import torch

from deeptab.core.default_factories import DefaultDataModuleFactory, DefaultTaskModelFactory
from deeptab.core.serialization import _warn_extension, build_save_bundle, restore_base_state, restore_loaded_metadata


class _SerializationMixin:
    """Bundle-based model persistence.

    Provides :meth:`save` and the classmethod :meth:`load` as the
    sole supported persistence mechanism for fitted DeepTab estimators.
    The bundle format is defined by
    :func:`~deeptab.core.serialization.build_save_bundle` and contains
    all state needed for inference: architecture config, neural-network
    weights, fitted preprocessor, feature schema, column order, task
    metadata, and a version snapshot.

    Note
    ----
    :class:`pickle` is **not** supported.  ``__getstate__`` intentionally
    clears ``task_model`` to prevent serialising Lightning modules.  Always
    use :meth:`save` / :meth:`load` instead.
    """

    if TYPE_CHECKING:
        # _emit_event is provided at runtime by _ObservabilityMixin via the MRO.
        # The stub here lets type-checkers resolve the call sites in save/load.
        def _emit_event(self, event: str, **kwargs) -> None: ...

    def save(self, path: str) -> None:
        """Save the fitted model to *path*.

        The bundle written by this method can be restored with
        :meth:`load`.  It contains all state required for inference:
        architecture/config, neural-network weights, fitted preprocessing
        state, feature schema, column order, task metadata, classifier
        classes (when available), and package versions for debugging
        reloads across environments.

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``"model.pt"``).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = MLPClassifier()
        >>> model.fit(X_train, y_train)
        >>> model.save("my_model.deeptab")
        >>> loaded = MLPClassifier.load("my_model.deeptab")
        >>> predictions = loaded.predict(X_test)
        """
        self._emit_event("save_started", path=path)
        _warn_extension(path)
        bundle = build_save_bundle(self, lss=False, family=None)
        torch.save(bundle, path)
        self._emit_event("save_completed", path=path)

    @classmethod
    def load(cls, path: str):
        """Load and return a fitted model from *path*.

        Parameters
        ----------
        path : str
            Path to a file previously written by :meth:`save`.

        Returns
        -------
        estimator
            A fully reconstructed, ready-to-predict estimator of the same
            type that was saved.

        Examples
        --------
        >>> loaded = MLPClassifier.load("my_model.deeptab")
        >>> predictions = loaded.predict(X_test)
        >>> print(loaded.task_info_["task"])
        'classification'
        >>> print(loaded.n_features_in_)
        6
        """
        _warn_extension(path)
        bundle = torch.load(path, weights_only=False)

        obj = bundle["_class"].__new__(bundle["_class"])
        restore_base_state(obj, bundle)

        # load() bypasses __init__, so factories are not yet set.
        # Initialise them to production defaults before using them.
        if not hasattr(obj, "_data_module_factory") or obj._data_module_factory is None:
            obj._data_module_factory = DefaultDataModuleFactory()
        if not hasattr(obj, "_task_model_factory") or obj._task_model_factory is None:
            obj._task_model_factory = DefaultTaskModelFactory()

        obj.data_module = obj._data_module_factory.create(
            preprocessor=bundle["preprocessor"],
            batch_size=bundle["batch_size"],
            shuffle=False,
            regression=bundle["regression"],
        )
        obj.data_module.num_feature_info = bundle["feature_info"]["num"]
        obj.data_module.cat_feature_info = bundle["feature_info"]["cat"]
        obj.data_module.embedding_feature_info = bundle["feature_info"]["emb"]
        obj.data_module.input_columns_ = bundle.get("input_columns")

        obj.task_model = obj._task_model_factory.create(
            model_class=bundle["model_class"],
            config=bundle["config"],
            feature_information=(
                bundle["feature_info"]["num"],
                bundle["feature_info"]["cat"],
                bundle["feature_info"]["emb"],
            ),
            num_classes=bundle["num_classes"],
            lss=bundle["lss"],
            family=bundle["family"],
            optimizer_type=bundle["optimizer_type"],
            optimizer_args=bundle["optimizer_kwargs"],
            lr=bundle["lr"],
            lr_patience=bundle["lr_patience"],
            lr_factor=bundle["lr_factor"],
            weight_decay=bundle["weight_decay"],
        )
        obj.task_model.load_state_dict(bundle["task_model_state_dict"])
        obj.task_model.eval()
        obj.estimator = obj.task_model.estimator

        obj.trainer = pl.Trainer(
            max_epochs=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        restore_loaded_metadata(obj, bundle)
        obj.data_module.input_columns_ = obj.input_columns_

        obj._emit_event("load_completed", path=path)
        return obj
