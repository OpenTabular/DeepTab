"""Bayesian hyperparameter optimisation for all DeepTab estimators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from skopt import gp_minimize

from deeptab.hpo.search_space import activation_mapper, get_search_space, round_to_nearest_16

if TYPE_CHECKING:
    from deeptab.data.datamodule import TabularDataModule
    from deeptab.training.lightning_module import TaskModel


class _HyperparameterMixin:
    # ---------------------------------------------------------------------------
    # Attributes provided by SklearnBase when this mixin is composed.
    # Declared here for static type-checkers only; never initialised in this class.
    # ---------------------------------------------------------------------------
    if TYPE_CHECKING:
        config: Any
        _trainer: Any
        _task_model: TaskModel | None
        _data_module: TabularDataModule | None

        def fit(self, X: Any, y: Any, **kwargs: Any) -> Any: ...
        def _build_model(self, X: Any, y: Any, **kwargs: Any) -> None: ...
        def build_model(self, X: Any, y: Any, **kwargs: Any) -> Any: ...
        def _score(self, X: Any, y: Any, embeddings: Any, metric: Any) -> float: ...

    """Bayesian hyperparameter search via :func:`skopt.gp_minimize`.

    Exposes :meth:`optimize_hparams`, which runs Gaussian-process
    Bayesian optimisation over the search space derived from the model's
    config dataclass, with optional epoch-level pruning to skip
    unpromising configurations early.
    """

    def optimize_hparams(
        self,
        X,
        y,
        regression,
        X_val=None,
        y_val=None,
        embeddings=None,
        embeddings_val=None,
        time=100,
        max_epochs=200,
        prune_by_epoch=True,
        prune_epoch=5,
        fixed_params={
            "pooling_method": "avg",
            "head_skip_layers": False,
            "head_layer_size_length": 0,
            "cat_encoding": "int",
            "head_skip_layer": False,
            "use_cls": False,
        },
        custom_search_space=None,
        **optimize_kwargs,
    ):
        """Optimise hyperparameters using Bayesian optimisation with optional pruning.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like
            Training labels.
        X_val, y_val : array-like, optional
            Validation data and labels.
        time : int
            Number of optimisation trials to run.
        max_epochs : int
            Maximum number of epochs per trial.
        prune_by_epoch : bool
            Whether to prune based on a specific epoch (``True``) or the best
            validation loss (``False``).
        prune_epoch : int
            The epoch at which to evaluate for pruning when ``prune_by_epoch``
            is ``True``.
        fixed_params : dict
            Hyperparameters to hold fixed during the search.
        custom_search_space : list or None, optional
            Override the default search space for this model.
        **optimize_kwargs
            Additional keyword arguments passed to ``fit``.

        Returns
        -------
        best_hparams : list
            Best hyperparameters found during optimisation.
        """
        param_names, param_space = get_search_space(
            self.config,
            fixed_params=fixed_params,
            custom_search_space=custom_search_space,
        )

        # Shared keyword arguments for every fit() call. The task-aware fit()
        # wrapper of each estimator injects ``regression`` (and an LSS ``family``
        # arrives via ``optimize_kwargs``), so neither is forwarded here. Optional
        # external embeddings are only passed when actually supplied, because the
        # LSS fit() signature does not accept them.
        base_fit_kwargs = {"X_val": X_val, "y_val": y_val, **optimize_kwargs}
        if embeddings is not None:
            base_fit_kwargs["embeddings"] = embeddings
        if embeddings_val is not None:
            base_fit_kwargs["embeddings_val"] = embeddings_val

        def _validation_loss():
            """Return the scalar Lightning ``val_loss`` for the current model.

            ``val_loss`` is the training objective itself (MSE for regression,
            cross-entropy for classification, negative log-likelihood for LSS),
            so it is always defined and always lower-is-better. Using it as the
            optimisation target keeps the search direction consistent across
            every task type.
            """
            return float(self._trainer.validate(self._task_model, self._data_module, verbose=False)[0]["val_loss"])

        # Initial fit to establish a baseline validation loss. rebuild=True (the
        # default) means this call also constructs the model; for LSS it sets the
        # distribution family that subsequent build_model() calls reuse.
        self.fit(X, y, max_epochs=max_epochs, **base_fit_kwargs)

        best_val_loss = _validation_loss()
        best_epoch_val_loss = self._task_model.epoch_val_loss_at(  # type: ignore
            prune_epoch
        )

        def _objective(hyperparams):
            nonlocal best_val_loss, best_epoch_val_loss

            head_layer_sizes = []
            head_layer_size_length = None

            for key, param_value in zip(param_names, hyperparams, strict=False):
                if key == "head_layer_size_length":
                    head_layer_size_length = param_value
                elif key.startswith("head_layer_size_"):
                    head_layer_sizes.append(round_to_nearest_16(param_value))
                elif isinstance(param_value, str) and param_value in activation_mapper:
                    # Activation fields are stored as nn.Module instances; the
                    # search space proposes them by name, so map name -> module.
                    setattr(self.config, key, activation_mapper[param_value])
                else:
                    setattr(self.config, key, param_value)

            if head_layer_size_length is not None:
                self.config.head_layer_sizes = head_layer_sizes[:head_layer_size_length]

            # Rebuild the model with the candidate config using the task-aware
            # public build_model(), which selects the correct head (regression,
            # classification, or the LSS distribution family stored on self).
            build_kwargs = {"X_val": X_val, "y_val": y_val, "lr": getattr(self.config, "lr", None)}
            if embeddings is not None:
                build_kwargs["embeddings"] = embeddings
            if embeddings_val is not None:
                build_kwargs["embeddings_val"] = embeddings_val
            self.build_model(X, y, **build_kwargs)

            if prune_by_epoch:
                early_pruning_threshold = best_epoch_val_loss * 1.5
            else:
                early_pruning_threshold = best_val_loss * 1.5  # type: ignore[operator]

            self._task_model.early_pruning_threshold = early_pruning_threshold  # type: ignore
            self._task_model.pruning_epoch = prune_epoch  # type: ignore

            try:
                # rebuild=False trains the model just constructed above so that
                # the pruning thresholds set on it are preserved.
                self.fit(X, y, max_epochs=max_epochs, rebuild=False, **base_fit_kwargs)

                val_loss = _validation_loss()

                epoch_val_loss = self._task_model.epoch_val_loss_at(  # type: ignore
                    prune_epoch
                )

                if prune_by_epoch and epoch_val_loss < best_epoch_val_loss:
                    best_epoch_val_loss = epoch_val_loss
                if val_loss < best_val_loss:  # type: ignore[operator]
                    best_val_loss = val_loss

                return val_loss

            except Exception as e:
                print(f"Error encountered during fit with hyperparameters {hyperparams}: {e}")
                return best_val_loss * 100  # type: ignore[operator]

        result = gp_minimize(_objective, param_space, n_calls=time, random_state=42)

        best_hparams = result.x  # type: ignore
        head_layer_sizes = [] if "head_layer_sizes" in self.config.__dataclass_fields__ else None
        layer_sizes = [] if "layer_sizes" in self.config.__dataclass_fields__ else None

        for key, param_value in zip(param_names, best_hparams, strict=False):
            if key.startswith("head_layer_size_") and head_layer_sizes is not None:
                head_layer_sizes.append(round_to_nearest_16(param_value))
            elif key.startswith("layer_size_") and layer_sizes is not None:
                layer_sizes.append(round_to_nearest_16(param_value))
            elif isinstance(param_value, str) and param_value in activation_mapper:
                setattr(self.config, key, activation_mapper[param_value])
            else:
                setattr(self.config, key, param_value)

        if head_layer_sizes is not None and head_layer_sizes:
            self.config.head_layer_sizes = head_layer_sizes
        if layer_sizes is not None and layer_sizes:
            self.config.layer_sizes = layer_sizes

        print("Best hyperparameters found:", best_hparams)
        return best_hparams
