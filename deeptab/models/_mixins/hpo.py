"""Bayesian hyperparameter optimisation for all DeepTab estimators."""

from __future__ import annotations

from skopt import gp_minimize

from deeptab.hpo import activation_mapper, get_search_space, round_to_nearest_16


class _HyperparameterMixin:
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

        # Initial fit to establish a baseline validation loss
        self.fit(
            X,
            y,
            regression=regression,
            X_val=X_val,
            y_val=y_val,
            embeddings=embeddings,
            embeddings_val=embeddings_val,
            max_epochs=max_epochs,
        )

        if hasattr(self, "score") and callable(self.score):  # type: ignore[attr-defined]
            if X_val is not None and y_val is not None:
                val_loss = self.score(X_val, y_val)  # type: ignore[attr-defined]
            else:
                val_loss = self.trainer.validate(self.task_model, self.data_module)[0]["val_loss"]
        else:
            raise NotImplementedError("The 'score' method is not implemented in the child class.")

        best_val_loss = val_loss
        best_epoch_val_loss = self.task_model.epoch_val_loss_at(  # type: ignore
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
                else:
                    field_type = self.config.__dataclass_fields__[key].type
                    if field_type == callable and isinstance(param_value, str):
                        if param_value in activation_mapper:
                            setattr(self.config, key, activation_mapper[param_value])
                        else:
                            raise ValueError(f"Unknown activation function: {param_value}")
                    else:
                        setattr(self.config, key, param_value)

            if head_layer_size_length is not None:
                self.config.head_layer_sizes = head_layer_sizes[:head_layer_size_length]

            self._build_model(
                X,
                y,
                regression=regression,
                X_val=X_val,
                y_val=y_val,
                embeddings=embeddings,
                embeddings_val=embeddings_val,
                lr=self.config.lr,
                **optimize_kwargs,
            )

            if prune_by_epoch:
                early_pruning_threshold = best_epoch_val_loss * 1.5
            else:
                early_pruning_threshold = best_val_loss * 1.5  # type: ignore[operator]

            self.task_model.early_pruning_threshold = early_pruning_threshold  # type: ignore
            self.task_model.pruning_epoch = prune_epoch  # type: ignore

            try:
                self.fit(
                    X,
                    y,
                    regression=regression,
                    X_val=X_val,
                    y_val=y_val,
                    max_epochs=max_epochs,
                    rebuild=False,
                )

                if hasattr(self, "score") and callable(self._score):
                    if X_val is not None and y_val is not None:
                        val_loss = self._score(X_val, y_val)  # type: ignore[call-arg]
                    else:
                        val_loss = self.trainer.validate(self.task_model, self.data_module)[0]["val_loss"]
                else:
                    raise NotImplementedError("The 'score' method is not implemented in the child class.")

                epoch_val_loss = self.task_model.epoch_val_loss_at(  # type: ignore
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
            else:
                field_type = self.config.__dataclass_fields__[key].type
                if field_type == callable and isinstance(param_value, str):
                    setattr(self.config, key, activation_mapper[param_value])
                else:
                    setattr(self.config, key, param_value)

        if head_layer_sizes is not None and head_layer_sizes:
            self.config.head_layer_sizes = head_layer_sizes
        if layer_sizes is not None and layer_sizes:
            self.config.layer_sizes = layer_sizes

        print("Best hyperparameters found:", best_hparams)
        return best_hparams
