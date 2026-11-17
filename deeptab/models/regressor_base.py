from collections.abc import Callable

import torch
from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score

from deeptab.core.exceptions import not_fitted_error
from deeptab.metrics import get_default_metrics_dict
from deeptab.models._mixins import _PointEstimatorMixin
from deeptab.models.base import SklearnBase


class SklearnBaseRegressor(RegressorMixin, _PointEstimatorMixin, SklearnBase):
    def __init__(
        self,
        model_config=None,
        preprocessing_config=None,
        trainer_config=None,
        observability_config=None,
        random_state=None,
    ):
        super().__init__(
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            observability_config=observability_config,
            random_state=random_state,
        )

    def build_model(
        self,
        X,
        y,
        val_size: float = 0.2,
        X_val=None,
        y_val=None,
        embeddings=None,
        embeddings_val=None,
        random_state: int = 101,
        batch_size: int = 128,
        shuffle: bool = True,
        lr: float | None = None,
        lr_patience: int | None = None,
        lr_factor: float | None = None,
        weight_decay: float | None = None,
        train_metrics: dict[str, Callable] | None = None,
        val_metrics: dict[str, Callable] | None = None,
        dataloader_kwargs=None,
    ):
        """Builds the model using the provided training data.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, 1)
            The target values (real numbers). Multi-output regression (more than
            one target column) is not currently supported.
        val_size : float, default=0.2
            The proportion of the dataset to include in the validation split if `X_val` is None.
            Ignored if `X_val` is provided.
        X_val : DataFrame or array-like, shape (n_samples, n_features), optional
            The validation input samples. If provided, `X` and `y` are not split and this data is used for validation.
        y_val : array-like, shape (n_samples,) or (n_samples, 1), optional
            The validation target values. Required if `X_val` is provided.
        random_state : int, default=101
            Controls the shuffling applied to the data before applying the split.
        batch_size : int, default=128
            Number of samples per gradient update.
        shuffle : bool, default=True
            Whether to shuffle the training data before each epoch.
        lr : float or None, default=None
            Learning rate for the optimizer. Falls back to the active
            `TrainerConfig`'s value, or 1e-4 when no `TrainerConfig` is set.
        lr_patience : int or None, default=None
            Epochs without improvement before reducing the learning rate.
            ``None`` uses the active ``TrainerConfig`` value.
        lr_factor : float or None, default=None
            Learning-rate reduction factor. ``None`` uses the active
            ``TrainerConfig`` value.
        weight_decay : float or None, default=None
            Weight decay (L2 penalty) coefficient. Falls back to the active
            `TrainerConfig`'s value, or 1e-6 when no `TrainerConfig` is set.
        train_metrics : dict, default=None
            torch.metrics dict to be logged during training.
        val_metrics : dict, default=None
            torch.metrics dict to be logged during validation.
        dataloader_kwargs : dict or None, default=None
            Extra PyTorch DataLoader keyword arguments. ``None`` adds none.



        Returns
        -------
        self : object
            The built regressor.
        """

        return super()._build_model(
            X,
            y,
            regression=True,
            val_size=val_size,
            X_val=X_val,
            y_val=y_val,
            embeddings=embeddings,
            embeddings_val=embeddings_val,
            num_classes=1,
            random_state=random_state,
            batch_size=batch_size,
            shuffle=shuffle,
            lr=lr,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
            weight_decay=weight_decay,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            dataloader_kwargs=dataloader_kwargs,
        )

    def fit(
        self,
        X,
        y,
        val_size: float | None = None,
        X_val=None,
        y_val=None,
        embeddings=None,
        embeddings_val=None,
        max_epochs: int | None = None,
        random_state: int | None = None,
        batch_size: int | None = None,
        shuffle: bool | None = None,
        patience: int | None = None,
        monitor: str | None = None,
        mode: str | None = None,
        lr: float | None = None,
        lr_patience: int | None = None,
        lr_factor: float | None = None,
        weight_decay: float | None = None,
        checkpoint_path: str | None = None,
        dataloader_kwargs=None,
        train_metrics: dict[str, Callable] | None = None,
        val_metrics: dict[str, Callable] | None = None,
        rebuild: bool | None = None,
        **trainer_kwargs,
    ):
        """Trains the regression model using the provided training data. Optionally, a separate validation set can be
        used.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, 1)
            The target values (real numbers). Multi-output regression (more than
            one target column) is not currently supported.
        val_size : float or None, default=None
            The proportion of the dataset to include in the validation split if `X_val` is None.
            Ignored if `X_val` is provided. Falls back to the active `TrainerConfig`'s
            value, or 0.2 when no `TrainerConfig` is set.
        X_val : DataFrame or array-like, shape (n_samples, n_features), optional
            The validation input samples. If provided, `X` and `y` are not split and this data is used for validation.
        y_val : array-like, shape (n_samples,) or (n_samples, 1), optional
            The validation target values. Required if `X_val` is provided.
        max_epochs : int or None, default=None
            Maximum number of epochs for training. Falls back to the active
            `TrainerConfig`'s value, or 100 when no `TrainerConfig` is set.
        random_state : int or None, default=None
            RNG seed for reproducibility. An explicit value here always wins
            over the ``random_state`` fixed at construction time. When both
            are ``None``, no reseeding happens, so any external
            ``set_seed``/``seed_context`` call already in effect is left
            untouched.
        batch_size : int or None, default=None
            Number of samples per gradient update. Falls back to the active
            `TrainerConfig`'s value, or 128 when no `TrainerConfig` is set.
        shuffle : bool or None, default=None
            Whether to shuffle the training data before each epoch. Falls back
            to the active `TrainerConfig`'s value, or `True` when no
            `TrainerConfig` is set.
        patience : int or None, default=None
            Number of epochs with no improvement on the validation loss to wait
            before early stopping. Falls back to the active `TrainerConfig`'s
            value, or 15 when no `TrainerConfig` is set.
        monitor : str or None, default=None
            The metric to monitor for early stopping. Falls back to the active
            `TrainerConfig`'s value, or "val_loss" when no `TrainerConfig` is set.
        mode : str or None, default=None
            Whether the monitored metric should be minimized (`min`) or
            maximized (`max`). Falls back to the active `TrainerConfig`'s value,
            or "min" when no `TrainerConfig` is set.
        lr : float or None, default=None
            Learning rate for the optimizer. Falls back to the active
            `TrainerConfig`'s value, or 1e-4 when no `TrainerConfig` is set.
        lr_patience : int or None, default=None
            Epochs without improvement before reducing the learning rate.
            ``None`` uses the active ``TrainerConfig`` value.
        lr_factor : float or None, default=None
            Learning-rate reduction factor. ``None`` uses the active
            ``TrainerConfig`` value.
        weight_decay : float or None, default=None
            Weight decay (L2 penalty) coefficient. Falls back to the active
            `TrainerConfig`'s value, or 1e-6 when no `TrainerConfig` is set.
        checkpoint_path : str or None, default=None
            Path where the checkpoints are being saved. When given, it is used
            as-is (each run must use a distinct path to avoid collisions
            between concurrent fits). Falls back to the active
            `TrainerConfig`'s value, or "model_checkpoints" when no
            `TrainerConfig` is set; in that case each fit writes to its own
            uniquely named sub-directory so parallel/repeated fits never
            collide.
        dataloader_kwargs : dict or None, default=None
            Extra PyTorch DataLoader keyword arguments. ``None`` adds none.
        train_metrics : dict, default=None
            torch.metrics dict to be logged during training.
        val_metrics : dict, default=None
            torch.metrics dict to be logged during validation.
        rebuild : bool or None, default=None
            Whether to rebuild the model when already built. ``None`` (the
            default) rebuilds unless the current model was warm-started via
            :meth:`pretrain`, in which case it continues training the
            pretrained model instead of discarding it. Pass ``True``/``False``
            explicitly to override that behavior either way.
        **trainer_kwargs : Additional keyword arguments for PyTorch Lightning's Trainer class.


        Returns
        -------
        self : object
            The fitted regressor.

        Notes
        -----
        ``rebuild=False`` retains neural weights, fitted preprocessing, loss,
        and metrics while preparing fresh data and loaders. Each call starts
        a new trainer, optimizer, and scheduler rather than resuming their
        previous state. Explicit epoch, learning-rate, weight-decay, batch,
        and loader overrides are honored. Replacing metrics requires
        ``rebuild=True``.

        Targets must be finite numeric values for a single output. Validation
        inputs must be supplied together with a matching feature schema.
        Input-validation errors preserve the previous fitted estimator. Failures
        after rebuilding or training starts invalidate prediction until a
        successful fit.

        Raises
        ------
        deeptab.core.exceptions.UnsupportedParamError
            If the unsupported ``sample_weight`` keyword is supplied. Rejection
            occurs before input validation or changes to estimator state.
        """

        return super().fit(
            X=X,
            y=y,
            regression=True,
            val_size=val_size,
            X_val=X_val,
            y_val=y_val,
            embeddings=embeddings,
            embeddings_val=embeddings_val,
            num_classes=1,
            max_epochs=max_epochs,
            random_state=random_state,
            batch_size=batch_size,
            shuffle=shuffle,
            patience=patience,
            monitor=monitor,
            mode=mode,
            lr=lr,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
            weight_decay=weight_decay,
            checkpoint_path=checkpoint_path,
            dataloader_kwargs=dataloader_kwargs,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            rebuild=rebuild,
            **trainer_kwargs,
        )

    def predict(self, X, embeddings=None, device=None):
        """Predicts target values for the given input samples.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The input samples for which to predict target values.


        Returns
        -------
        predictions : ndarray, shape (n_samples,)
            The predicted target values.
        """
        X = self._validate_predict_input(X, method="predict")
        if self._task_model is None:
            raise not_fitted_error(type(self).__name__, "predict")

        self._emit_event("predict_started", n_samples=len(X))

        # Preprocess the data using the data module
        self._data_module.assign_predict_dataset(X, embeddings)  # type: ignore[union-attr]

        # Set model to evaluation mode
        self._task_model.eval()

        # Perform inference using PyTorch Lightning's predict function
        predictions_list = self._resolve_predict_trainer(device).predict(self._task_model, self._data_module)  # type: ignore[union-attr, arg-type]

        # Concatenate predictions from all batches
        predictions = torch.cat(predictions_list, dim=0)  # type: ignore

        # Check if ensemble is used
        if getattr(self._task_model.estimator, "returns_ensemble", False):  # If using ensemble
            predictions = predictions.mean(dim=1)  # Average over ensemble dimension

        # Convert predictions to NumPy array and return
        predictions = predictions.cpu().numpy()
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions.ravel()
        self._emit_event("predict_completed")
        return predictions

    def evaluate(self, X, y_true, embeddings=None, metrics=None):
        """Evaluate the model on the given data using specified metrics.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples to predict.
        y_true : array-like of shape (n_samples,) or (n_samples, 1)
            The true target values against which to evaluate the predictions.
        metrics : dict
            A dictionary where keys are metric names and values are the metric functions.


        Notes
        -----
        This method uses the `predict` method to generate predictions and computes each metric.

        Returns
        -------
        scores : dict
            A dictionary with metric names as keys and their corresponding scores as values.
        """
        if metrics is None:
            metrics = get_default_metrics_dict("regression")

        # Generate predictions using the trained model
        predictions = self.predict(X, embeddings=embeddings)

        # Initialize dictionary to store results
        scores = {}

        # Compute each metric
        for metric_name, metric_func in metrics.items():
            scores[metric_name] = metric_func(y_true, predictions)

        return scores

    def score(self, X, y, embeddings=None, metric=r2_score):
        """Calculate the score of the model using the specified metric.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples to predict.
        y : array-like of shape (n_samples,) or (n_samples, 1)
            The true target values against which to evaluate the predictions.
        metric : callable, default=r2_score
            The metric function to use for evaluation. Must be a callable with the
            signature ``metric(y_true, y_pred)``. Defaults to ``r2_score`` to match
            scikit-learn's ``RegressorMixin`` convention (higher is better).

        Returns
        -------
        score : float
            The score calculated using the specified metric.

        Examples
        --------
        >>> from sklearn.metrics import mean_squared_error, mean_absolute_error
        >>> model.score(X_test, y_test)                             # R² (default)
        >>> model.score(X_test, y_test, metric=mean_squared_error)  # MSE
        >>> model.score(X_test, y_test, metric=mean_absolute_error) # MAE
        """
        score = super()._score(X, y, embeddings, metric)
        return score
