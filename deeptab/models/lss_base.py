import warnings
from collections.abc import Callable

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from pretab.preprocessor import Preprocessor
from torch.utils.data import DataLoader
from tqdm import tqdm

from deeptab.core.exceptions import not_fitted_error
from deeptab.core.serialization import _warn_extension, build_save_bundle, restore_base_state, restore_loaded_metadata
from deeptab.core.sklearn_compat import ensure_dataframe, set_input_feature_attributes, validate_input_features
from deeptab.data.datamodule import TabularDataModule
from deeptab.distributions import get_distribution
from deeptab.metrics import get_default_metrics_dict
from deeptab.models.base import SklearnBase, _validate_fit_inputs
from deeptab.training import TaskModel


class SklearnBaseLSS(SklearnBase):
    """Distributional regression base class (LSS variant of SklearnBase).

    Inherits all sklearn compatibility, parameter management, serialization,
    HPO, and observability from ``SklearnBase``. Overrides ``build_model``,
    ``fit``, ``predict``, ``save``, and ``load`` to add LSS-specific concerns:
    distribution family selection, ``lss=True`` flag to ``TaskModel``, and
    distribution-transform post-processing in ``predict``.
    """

    def build_model(
        self,
        X,
        y,
        val_size: float = 0.2,
        X_val=None,
        y_val=None,
        random_state: int = 101,
        batch_size: int = 128,
        shuffle: bool = True,
        lr: float | None = None,
        lr_patience: int | None = None,
        lr_factor: float | None = None,
        weight_decay: float | None = None,
        train_metrics: dict[str, Callable] | None = None,
        val_metrics: dict[str, Callable] | None = None,
        dataloader_kwargs={},
    ):
        """Builds the model using the provided training data.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            The target values (real numbers).
        val_size : float, default=0.2
            The proportion of the dataset to include in the validation split if `X_val` is None.
            Ignored if `X_val` is provided.
        X_val : DataFrame or array-like, shape (n_samples, n_features), optional
            The validation input samples. If provided, `X` and `y` are not split and this data is used for validation.
        y_val : array-like, shape (n_samples,) or (n_samples, n_targets), optional
            The validation target values. Required if `X_val` is provided.
        random_state : int, default=101
            Controls the shuffling applied to the data before applying the split.
        batch_size : int, default=64
            Number of samples per gradient update.
        shuffle : bool, default=True
            Whether to shuffle the training data before each epoch.
        lr : float, default=1e-3
            Learning rate for the optimizer.
        lr_patience : int, default=10
            Number of epochs with no improvement on the validation loss to wait before reducing the learning rate.
        lr_factor : float, default=0.1
            Factor by which the learning rate will be reduced.
        train_metrics : dict, default=None
            torch.metrics dict to be logged during training.
        val_metrics : dict, default=None
            torch.metrics dict to be logged during validation.
        weight_decay : float, default=0.025
            Weight decay (L2 penalty) coefficient.
        dataloader_kwargs: dict, default={}
            The kwargs for the pytorch dataloader class.

        Returns
        -------
        self : object
            The built distributional regressor.
        """
        # When trainer_config is active, resolve lr / scheduler params from it
        if self.trainer_config is not None:
            tc = self.trainer_config
            if lr is None:
                lr = tc.lr
            if lr_patience is None:
                lr_patience = tc.lr_patience
            if lr_factor is None:
                lr_factor = tc.lr_factor
            if weight_decay is None:
                weight_decay = tc.weight_decay

        # Re-sync preprocessor from current preprocessing_config state so that
        # direct mutations (e.g. clf.preprocessing_config.n_bins = 8) are
        # honoured on the next fit(), consistent with set_params() behaviour.
        if self.preprocessing_config is not None:
            self._preprocessor_kwargs = self.preprocessing_config.to_preprocessor_kwargs()
            self._preprocessor = Preprocessor(**self._preprocessor_kwargs)

        X = ensure_dataframe(X)
        set_input_feature_attributes(self, X)
        self.classes_ = np.unique(y) if getattr(self, "family_name", None) == "categorical" else None
        if hasattr(y, "values"):
            y = y.values
        if X_val is not None:
            X_val = ensure_dataframe(X_val)
            if y_val is not None and hasattr(y_val, "values"):
                y_val = y_val.values

        self._data_module = TabularDataModule(
            preprocessor=self._preprocessor,
            batch_size=batch_size,
            shuffle=shuffle,
            X_val=X_val,
            y_val=y_val,
            val_size=val_size,
            random_state=random_state,
            regression=getattr(self, "family_name", None) != "categorical",
            **dataloader_kwargs,
        )
        self._data_module.input_columns_ = self.input_columns_

        self._data_module.preprocess_data(X, y, X_val, y_val, val_size=val_size, random_state=random_state)

        # After the first build, self._estimator holds the model *instance*
        # (assigned below). Resolve back to the class so repeated builds
        # (e.g. HPO trials or a refit) construct a fresh model correctly.
        _model_class = self._estimator if isinstance(self._estimator, type) else type(self._estimator)
        self._task_model = TaskModel(
            model_class=_model_class,  # type: ignore
            num_classes=self.family.param_count,
            family=self.family,
            config=self.config,
            feature_information=(
                self._data_module.num_feature_info,
                self._data_module.cat_feature_info,
                self._data_module.embedding_feature_info,
            ),
            lr=lr if lr is not None else getattr(self.config, "lr", None),
            lr_patience=(lr_patience if lr_patience is not None else getattr(self.config, "lr_patience", None)),
            lr_factor=lr_factor if lr_factor is not None else getattr(self.config, "lr_factor", None),
            weight_decay=(weight_decay if weight_decay is not None else getattr(self.config, "weight_decay", None)),
            lss=True,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            optimizer_type=(  # type: ignore[arg-type]
                self.trainer_config.optimizer_type if self.trainer_config is not None else self._optimizer_type
            ),
            optimizer_args=(
                getattr(self.trainer_config, "optimizer_kwargs", None) or self._optimizer_kwargs
                if self.trainer_config is not None
                else self._optimizer_kwargs
            ),
        )

        self._built = True
        self._estimator = self._task_model.estimator

        return self

    def fit(
        self,
        X,
        y,
        family,
        val_size: float = 0.2,
        X_val=None,
        y_val=None,
        max_epochs: int = 100,
        random_state: int = 101,
        batch_size: int = 128,
        shuffle: bool = True,
        patience: int = 15,
        monitor: str = "val_loss",
        mode: str = "min",
        lr: float | None = None,
        lr_patience: int | None = None,
        lr_factor: float | None = None,
        weight_decay: float | None = None,
        checkpoint_path="model_checkpoints",
        distributional_kwargs=None,
        train_metrics: dict[str, Callable] | None = None,
        val_metrics: dict[str, Callable] | None = None,
        dataloader_kwargs={},
        rebuild=True,
        **trainer_kwargs,
    ):
        """Trains the regression model using the provided training data. Optionally, a separate validation set can be
        used.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            The target values (real numbers).
        family : str
            The name of the distribution family to use for the loss function. Examples include 'normal'
            for regression tasks.
        val_size : float, default=0.2
            The proportion of the dataset to include in the validation split if `X_val` is None.
            Ignored if `X_val` is provided.
        X_val : DataFrame or array-like, shape (n_samples, n_features), optional
            The validation input samples. If provided, `X` and `y` are not split and this data is used for validation.
        y_val : array-like, shape (n_samples,) or (n_samples, n_targets), optional
            The validation target values. Required if `X_val` is provided.
        max_epochs : int, default=100
            Maximum number of epochs for training.
        random_state : int, default=101
            Controls the shuffling applied to the data before applying the split.
        batch_size : int, default=64
            Number of samples per gradient update.
        shuffle : bool, default=True
            Whether to shuffle the training data before each epoch.
        patience : int, default=10
            Number of epochs with no improvement on the validation loss to wait before early stopping.
        monitor : str, default="val_loss"
            The metric to monitor for early stopping.
        mode : str, default="min"
            Whether the monitored metric should be minimized (`min`) or maximized (`max`).
        lr : float, default=1e-3
            Learning rate for the optimizer.
        lr_patience : int, default=10
            Number of epochs with no improvement on the validation loss to wait before reducing the learning rate.
        factor : float, default=0.1
            Factor by which the learning rate will be reduced.
        weight_decay : float, default=0.025
            Weight decay (L2 penalty) coefficient.
        distributional_kwargs : dict, default=None
            any arguments taht are specific for a certain distribution.
        train_metrics : dict, default=None
            torch.metrics dict to be logged during training.
        val_metrics : dict, default=None
            torch.metrics dict to be logged during validation.
        checkpoint_path : str, default="model_checkpoints"
            Path where the checkpoints are being saved.
        dataloader_kwargs: dict, default={}
            The kwargs for the pytorch dataloader class.
        **trainer_kwargs : Additional keyword arguments for PyTorch Lightning's Trainer class.


        Returns
        -------
        self : object
            The fitted regressor.
        """
        # When trainer_config is active, override all training-loop params from it
        if self.trainer_config is not None:
            tc = self.trainer_config
            max_epochs = tc.max_epochs
            batch_size = tc.batch_size
            val_size = tc.val_size
            shuffle = tc.shuffle
            patience = tc.patience
            monitor = tc.monitor
            mode = tc.mode
            checkpoint_path = tc.checkpoint_path

        # Validate inputs before any preprocessing or model construction
        _validate_fit_inputs(X, y, regression=True, family=family)

        # When random_state was fixed at construction time, honour it
        if self.random_state is not None:
            random_state = self.random_state

        if distributional_kwargs is None:
            distributional_kwargs = {}

        self.family = get_distribution(family, **distributional_kwargs)
        self.family_name = family

        if rebuild:
            self.build_model(
                X=X,
                y=y,
                val_size=val_size,
                X_val=X_val,
                y_val=y_val,
                random_state=random_state,
                batch_size=batch_size,
                shuffle=shuffle,
                lr=lr,
                lr_patience=lr_patience,
                lr_factor=lr_factor,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                weight_decay=weight_decay,
                dataloader_kwargs=dataloader_kwargs,
            )

        else:
            if not self._built:
                raise ValueError(
                    "The model must be built before calling the fit method. \
                                 Either call .build_model() or set rebuild=True"
                )

        early_stop_callback = EarlyStopping(
            monitor=monitor, min_delta=0.00, patience=patience, verbose=False, mode=mode
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",  # Adjust according to your validation metric
            mode="min",
            save_top_k=1,
            dirpath=checkpoint_path,  # Specify the directory to save checkpoints
            filename="best_model",
        )

        # Initialize the trainer and train the model
        self._trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[
                early_stop_callback,
                checkpoint_callback,
                ModelSummary(max_depth=2),
            ],
            **trainer_kwargs,
        )
        self._trainer.fit(self._task_model, self._data_module)  # type: ignore

        self._best_model_path = checkpoint_callback.best_model_path
        if self._best_model_path:
            torch.serialization.add_safe_globals([type(self.config)])
            checkpoint = torch.load(self._best_model_path, weights_only=False)
            self._task_model.load_state_dict(checkpoint["state_dict"])  # type: ignore

        self.is_fitted_ = True
        return self

    def predict(self, X, raw=False, device=None):
        """Predicts target values for the given input samples.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The input samples for which to predict target values.


        Returns
        -------
        predictions : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            The predicted target values.
        """
        X = self._validate_predict_input(X)
        if self._task_model is None:
            raise not_fitted_error(type(self).__name__, "predict")

        self._emit_event("predict_started", n_samples=len(X))

        # Preprocess the data using the data module
        self._data_module.assign_predict_dataset(X)  # type: ignore[union-attr]

        # Set model to evaluation mode
        self._task_model.eval()

        # Perform inference using PyTorch Lightning's predict function
        predictions_list = self._trainer.predict(self._task_model, self._data_module)  # type: ignore[union-attr, arg-type]

        # Concatenate predictions from all batches
        predictions = torch.cat(predictions_list, dim=0)  # type: ignore[arg-type]

        # Check if ensemble is used
        if getattr(self._estimator, "returns_ensemble", False):  # If using ensemble
            predictions = predictions.mean(dim=1)  # Average over ensemble dimension

        if not raw:
            result = self._task_model.family(predictions).cpu().numpy()  # type: ignore
        else:
            result = predictions.cpu().numpy()
        self._emit_event("predict_completed")
        return result

    def evaluate(self, X, y_true, metrics=None, distribution_family=None):
        """Evaluate the model on the given data using specified metrics.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples to predict.
        y_true : array-like of shape (n_samples,)
            The true target values.
        metrics : dict, optional
            A ``{name: callable}`` dictionary of metric functions with signature
            ``metric(y_true, y_pred) -> float``.  Each callable may be a
            :class:`~deeptab.metrics.DeepTabMetric` instance or any plain
            callable.  When a metric has ``needs_raw=True``, raw model logits
            are passed instead of transformed distribution parameters.
            If ``None``, the default metrics for the distribution family are
            used (see :func:`deeptab.metrics.get_default_metrics`).
        distribution_family : str, optional
            Distribution family key (e.g. ``"normal"``, ``"gamma"``).  Inferred
            from the fitted model when ``None``.

        Returns
        -------
        scores : dict
            ``{metric_name: score}`` dictionary.
        """
        # Infer distribution family from model settings if not provided
        if distribution_family is None:
            distribution_family = getattr(self._task_model, "distribution_family", "normal")

        # Setup default metrics if none are provided
        if metrics is None:
            metrics = self.get_default_metrics(distribution_family)

        # Obtain both transformed and raw predictions up-front only when needed
        needs_any_raw = any(getattr(fn, "needs_raw", False) for fn in metrics.values())
        predictions_transformed = self.predict(X, raw=False)
        predictions_raw = self.predict(X, raw=True) if needs_any_raw else None

        y_true = np.asarray(y_true)
        scores = {}
        for metric_name, metric_func in metrics.items():
            _needs_raw = getattr(metric_func, "needs_raw", False)
            preds = predictions_raw if (_needs_raw and predictions_raw is not None) else predictions_transformed
            try:
                scores[metric_name] = metric_func(y_true, preds)
            except Exception as exc:
                warnings.warn(f"Metric '{metric_name}' failed: {exc}", RuntimeWarning, stacklevel=2)
                scores[metric_name] = float("nan")

        return scores

    def get_default_metrics(self, distribution_family):
        """Return default evaluation metrics for the given distribution family.

        Delegates to :func:`deeptab.metrics.get_default_metrics_dict`, which
        returns a ``{name: DeepTabMetric}`` dictionary covering all supported
        distribution families.

        Parameters
        ----------
        distribution_family : str
            Distribution family key, e.g. ``"normal"``, ``"gamma"``.

        Returns
        -------
        dict
            ``{metric_name: callable}`` dictionary of metric functions.
        """
        return get_default_metrics_dict("lss", family=distribution_family)

    def score(self, X, y, metric="NLL"):
        """Calculate the score of the model using the specified metric.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples to predict.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The true target values against which to evaluate the predictions.
        metric : str, default="NLL"
            So far, only negative log-likelihood is supported

        Returns
        -------
        score : float
            The score calculated using the specified metric.
        """
        predictions = self.predict(X)
        score = self._task_model.family.evaluate_nll(y, predictions)  # type: ignore
        return score

    def encode(self, X, batch_size=64):
        """
        Encodes input data using the trained model's embedding layer.

        Parameters
        ----------
        X : array-like or DataFrame
            Input data to be encoded.
        batch_size : int, optional, default=64
            Batch size for encoding.

        Returns
        -------
        torch.Tensor
            Encoded representations of the input data.

        Raises
        ------
        ValueError
            If the model or data module is not fitted.
        """
        # Ensure model and data module are initialized
        if self._task_model is None or self._data_module is None:
            raise ValueError("The model or data module has not been fitted yet.")
        if not hasattr(self._task_model.estimator, "embedding_layer"):  # type: ignore[union-attr]
            raise AttributeError(
                f"{type(self._task_model.estimator).__name__} does not have an embedding_layer."  # type: ignore[union-attr]
            )
        encoded_dataset = self._data_module.preprocess_new_data(X)

        data_loader = DataLoader(encoded_dataset, batch_size=batch_size, shuffle=False)

        # Process data in batches
        encoded_outputs = []
        for num_features, cat_features in tqdm(data_loader):
            embeddings = self._task_model.estimator.encode(num_features, cat_features)  # type: ignore[union-attr]  # Call your encode function
            encoded_outputs.append(embeddings)

        # Concatenate all encoded outputs
        encoded_outputs = torch.cat(encoded_outputs, dim=0)

        return encoded_outputs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the fitted model to *path*.

        The bundle written by this method can be restored with
        :meth:`load`.  It contains all state required for inference:
        the architecture/config, neural-network weights, fitted
        preprocessing state, feature schema and column order, task
        metadata, distribution family, classifier classes for
        categorical LSS models, and package versions for debugging
        reloads across environments.

        The bundle is built by :func:`~deeptab.core.serialization.build_save_bundle`,
        which is the single source of truth for artifact structure across all
        model variants.

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
        >>> model = MLPLSS()
        >>> model.fit(X_train, y_train, family="normal")
        >>> model.save("my_lss_model.deeptab")
        >>> loaded = MLPLSS.load("my_lss_model.deeptab")
        >>> predictions = loaded.predict(X_test)
        """
        _warn_extension(path)
        bundle = build_save_bundle(self, lss=True, family=self.family_name)
        torch.save(bundle, path)

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
            A fully reconstructed, ready-to-predict estimator. Exposes
            ``artifact_metadata_``, ``architecture_metadata_``,
            ``feature_schema_``, ``input_columns_``, ``task_info_``,
            ``classes_``, and ``versions_`` attributes after loading.

        Examples
        --------
        >>> loaded = MLPLSS.load("my_lss_model.deeptab")
        >>> predictions = loaded.predict(X_test)
        >>> print(loaded.task_info_[\"family\"])
        'normal'
        """
        _warn_extension(path)
        bundle = torch.load(path, weights_only=False)

        obj = bundle["_class"].__new__(bundle["_class"])
        restore_base_state(obj, bundle)
        obj.family = get_distribution(bundle["family"])
        obj.family_name = bundle["family"]

        obj._data_module = TabularDataModule(
            preprocessor=bundle["preprocessor"],
            batch_size=bundle["batch_size"],
            shuffle=False,
            regression=bundle["regression"],
        )
        obj._data_module.num_feature_info = bundle["feature_info"]["num"]
        obj._data_module.cat_feature_info = bundle["feature_info"]["cat"]
        obj._data_module.embedding_feature_info = bundle["feature_info"]["emb"]
        obj._data_module.input_columns_ = bundle.get("input_columns")

        obj._task_model = TaskModel(
            model_class=bundle["model_class"],
            config=bundle["config"],
            feature_information=(
                bundle["feature_info"]["num"],
                bundle["feature_info"]["cat"],
                bundle["feature_info"]["emb"],
            ),
            num_classes=bundle["num_classes"],
            lss=bundle["lss"],
            family=obj.family,
            optimizer_type=bundle["optimizer_type"],
            optimizer_args=bundle["optimizer_kwargs"],
            lr=bundle["lr"],
            lr_patience=bundle["lr_patience"],
            lr_factor=bundle["lr_factor"],
            weight_decay=bundle["weight_decay"],
        )
        obj._task_model.load_state_dict(bundle["task_model_state_dict"])
        obj._task_model.eval()
        obj._estimator = obj._task_model.estimator

        obj._trainer = pl.Trainer(
            max_epochs=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        restore_loaded_metadata(obj, bundle)
        obj._data_module.input_columns_ = obj.input_columns_

        return obj

    def optimize_hparams(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
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
        """Optimizes hyperparameters using Bayesian optimization with optional pruning.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like
            Training labels.
        X_val, y_val : array-like, optional
            Validation data and labels.
        time : int
            The number of optimization trials to run.
        max_epochs : int
            Maximum number of epochs for training.
        prune_by_epoch : bool
            Whether to prune based on a specific epoch (True) or the best validation loss (False).
        prune_epoch : int
            The specific epoch to prune by when prune_by_epoch is True.
        **optimize_kwargs : dict
            Additional keyword arguments passed to the fit method.

        Returns
        -------
        best_hparams : list
            Best hyperparameters found during optimization.
        """

        return super().optimize_hparams(
            X,
            y,
            regression=False,
            X_val=X_val,
            y_val=y_val,
            time=time,
            max_epochs=max_epochs,
            prune_by_epoch=prune_by_epoch,
            prune_epoch=prune_epoch,
            fixed_params=fixed_params,
            custom_search_space=custom_search_space,
            **optimize_kwargs,
        )
