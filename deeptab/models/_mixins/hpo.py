"""Bayesian hyperparameter optimisation for all DeepTab estimators."""

from __future__ import annotations

import math
from sys import float_info
from typing import TYPE_CHECKING, Any, cast

from lightning.pytorch.callbacks import Callback
from skopt import gp_minimize

from deeptab.hpo.search_space import activation_mapper, get_search_space, round_to_nearest_16

if TYPE_CHECKING:
    from deeptab.data.datamodule import TabularDataModule
    from deeptab.training.lightning_module import TaskModel


class _TrialPruningCallback(Callback):
    def __init__(self, threshold: float, epoch: int):
        self.threshold = threshold
        self.epoch = epoch

    def on_fit_start(self, trainer, pl_module):
        task_model = cast("TaskModel", pl_module)
        task_model.early_pruning_threshold = self.threshold
        task_model.pruning_epoch = self.epoch


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
        random_state: int | None

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
        fixed_params=None,
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
            Zero-based training epoch at which pruning starts. Sanity checks
            and standalone validation do not count as training epochs.
        fixed_params : dict
            Hyperparameters to hold fixed during the search.
        custom_search_space : dict or None, optional
            Map parameter names to replacement search dimensions.
        **optimize_kwargs
            Additional keyword arguments passed to ``fit``.

        Returns
        -------
        best_hparams : list
            Best successful trial's hyperparameters. The estimator is rebuilt
            and fitted with these settings before this method returns.

        Raises
        ------
        ValueError
            If the baseline validation loss is not finite.
        RuntimeError
            If no trial completes with a finite validation loss.
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
        base_fit_kwargs.pop("rebuild", None)
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

        # Initial fit to establish a baseline validation loss. rebuild=True
        # constructs a fresh model; for LSS it sets the
        # distribution family that subsequent build_model() calls reuse.
        self.fit(X, y, max_epochs=max_epochs, rebuild=True, **base_fit_kwargs)

        best_val_loss = _validation_loss()
        if not math.isfinite(best_val_loss):
            raise ValueError("Baseline validation loss must be finite for hyperparameter search")
        best_epoch_val_loss = self._task_model.epoch_val_loss_at(  # type: ignore
            prune_epoch
        )
        best_trial_loss = float("inf")
        best_hparams = None

        def _apply_hyperparams(hyperparams):
            head_layer_sizes = []
            head_layer_size_length = None

            for key, param_value in zip(param_names, hyperparams, strict=False):
                if key == "head_layer_size_length":
                    head_layer_size_length = int(param_value)
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
            elif head_layer_sizes:
                self.config.head_layer_sizes = head_layer_sizes

        def _objective(hyperparams):
            nonlocal best_val_loss, best_epoch_val_loss, best_trial_loss, best_hparams

            try:
                _apply_hyperparams(hyperparams)
                pruning_baseline = best_epoch_val_loss if prune_by_epoch else best_val_loss
                early_pruning_threshold = pruning_baseline + max(abs(pruning_baseline) * 0.5, 1e-12)
                trial_fit_kwargs = dict(base_fit_kwargs)
                callbacks = trial_fit_kwargs.pop("callbacks", None) or []
                if isinstance(callbacks, Callback):
                    callbacks = [callbacks]
                trial_fit_kwargs["callbacks"] = [
                    *callbacks,
                    _TrialPruningCallback(early_pruning_threshold, prune_epoch),
                ]
                self.fit(X, y, max_epochs=max_epochs, rebuild=True, **trial_fit_kwargs)

                val_loss = _validation_loss()
                if not math.isfinite(val_loss):
                    raise ValueError("Trial validation loss must be finite")

                epoch_val_loss = self._task_model.epoch_val_loss_at(  # type: ignore
                    prune_epoch
                )

                if prune_by_epoch and epoch_val_loss < best_epoch_val_loss:
                    best_epoch_val_loss = epoch_val_loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                if val_loss < best_trial_loss:
                    best_trial_loss = val_loss
                    best_hparams = list(hyperparams)

                return val_loss

            except Exception as e:
                print(f"Error encountered during fit with hyperparameters {hyperparams}: {e}")
                return min(abs(best_val_loss) * 100 + 1_000_000, float_info.max)

        gp_minimize(_objective, param_space, n_calls=time, random_state=self.random_state)
        if best_hparams is None:
            raise RuntimeError("No hyperparameter trial completed successfully")

        _apply_hyperparams(best_hparams)
        self.fit(X, y, max_epochs=max_epochs, rebuild=True, **base_fit_kwargs)

        print("Best hyperparameters found:", best_hparams)
        return best_hparams
