from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss

from deeptab.core.exceptions import NotFittedError, not_fitted_error
from deeptab.metrics import get_default_metrics_dict
from deeptab.models.base import SklearnBase, _raise_flat_param_error
from deeptab.training.losses import build_classification_loss, compute_class_weights


def _resolve_loss_and_sampler(loss_fct, class_weight, balanced_sampler, sample_weight, y, classes, num_classes):
    """Translate the imbalance-handling arguments into a ``(loss_fct, sampler)`` pair.

    * ``loss_fct`` — an ``nn.Module``, a registered loss name (e.g. ``"focal"``),
      or ``None``. Combined with ``class_weight`` via
      :func:`deeptab.training.losses.build_classification_loss`.
    * ``sampler`` — ``sample_weight`` (explicit per-row weights) takes precedence,
      otherwise ``"balanced"`` when ``balanced_sampler`` is set, otherwise ``None``.
    """
    class_weights = None
    if class_weight is not None:
        class_weights = compute_class_weights(class_weight, y, classes=classes)
    resolved_loss = build_classification_loss(loss_fct, num_classes=num_classes, class_weights=class_weights)

    if sample_weight is not None:
        sampler = sample_weight
    elif balanced_sampler:
        sampler = "balanced"
    else:
        sampler = None
    return resolved_loss, sampler


class SklearnBaseClassifier(SklearnBase):
    def __init__(
        self,
        model,
        config,
        model_config=None,
        preprocessing_config=None,
        trainer_config=None,
        random_state=None,
        **kwargs,
    ):
        if kwargs:
            _raise_flat_param_error(kwargs, type(self).__name__)
        super().__init__(
            model,
            config,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
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
        dataloader_kwargs={},
        class_weight: str | dict | list | np.ndarray | None = None,
        loss_fct=None,
        balanced_sampler: bool = False,
        sample_weight=None,
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
        batch_size : int, default=128
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

        class_weight : {"balanced"}, dict, array-like, or None, default=None
            Weights associated with classes for imbalanced data. ``"balanced"``
            mirrors scikit-learn and uses ``n_samples / (n_classes * bincount(y))``.
            A mapping ``{class_label: weight}`` or an array (ordered like
            ``np.unique(y)``) sets weights explicitly. Ignored when ``loss_fct``
            is an ``nn.Module``.
        loss_fct : nn.Module, str, or None, default=None
            Custom loss. An ``nn.Module`` is used as-is; a registered loss name
            (e.g. ``"focal"``, ``"bce"``, ``"cross_entropy"``) is built and
            combined with ``class_weight``. ``None`` falls back to the default
            (weighted) task loss.
        balanced_sampler : bool, default=False
            If ``True``, draw class-balanced mini-batches with a
            ``WeightedRandomSampler`` (oversamples minority classes).
        sample_weight : array-like, optional
            Explicit per-row sampling weights (length matches ``X``). Takes
            precedence over ``balanced_sampler`` and drives the
            ``WeightedRandomSampler``.

        Returns
        -------
        self : object
            The built classifier.
        """

        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)

        loss_fct, sampler = _resolve_loss_and_sampler(
            loss_fct, class_weight, balanced_sampler, sample_weight, y, self.classes_, num_classes
        )

        return super()._build_model(
            X,
            y,
            regression=False,
            val_size=val_size,
            X_val=X_val,
            y_val=y_val,
            embeddings=embeddings,
            embeddings_val=embeddings_val,
            num_classes=num_classes,
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
            loss_fct=loss_fct,
            sampler=sampler,
        )

    def fit(
        self,
        X,
        y,
        val_size: float = 0.2,
        X_val=None,
        y_val=None,
        embeddings=None,
        embeddings_val=None,
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
        train_metrics: dict[str, Callable] | None = None,
        val_metrics: dict[str, Callable] | None = None,
        dataloader_kwargs={},
        rebuild=True,
        class_weight: str | dict | list | np.ndarray | None = None,
        loss_fct=None,
        balanced_sampler: bool = False,
        sample_weight=None,
        **trainer_kwargs,
    ):
        """Trains the classification model using the provided training data. Optionally, a separate validation set can
        be used.

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
        checkpoint_path : str, default="model_checkpoints"
            Path where the checkpoints are being saved.
        train_metrics : dict, default=None
            torch.metrics dict to be logged during training.
        val_metrics : dict, default=None
            torch.metrics dict to be logged during validation.
        dataloader_kwargs: dict, default={}
            The kwargs for the pytorch dataloader class.
        rebuild: bool, default=True
            Whether to rebuild the model when it already was built.
        class_weight : {"balanced"}, dict, array-like, or None, default=None
            Weights associated with classes for imbalanced data. ``"balanced"``
            mirrors scikit-learn and uses ``n_samples / (n_classes * bincount(y))``
            so under-represented classes contribute more to the loss. A mapping
            ``{class_label: weight}`` or an array (ordered like ``np.unique(y)``)
            sets weights explicitly. For binary targets the weights are converted
            to a ``pos_weight`` for ``BCEWithLogitsLoss``; for multiclass they
            become the ``weight`` of ``CrossEntropyLoss``. Ignored when
            ``loss_fct`` is an ``nn.Module``.
        loss_fct : nn.Module, str, or None, default=None
            Custom loss. An ``nn.Module`` is used as-is; a registered loss name
            (e.g. ``"focal"``, ``"bce"``, ``"cross_entropy"``) is built and
            combined with ``class_weight`` (see
            :func:`deeptab.training.losses.build_classification_loss`). ``None``
            falls back to the default (weighted) task loss.
        balanced_sampler : bool, default=False
            If ``True``, draw class-balanced mini-batches with a
            ``WeightedRandomSampler`` (oversamples minority classes). This
            rebalances the data instead of (or in addition to) reweighting the
            loss.
        sample_weight : array-like, optional
            Explicit per-row sampling weights (length matches ``X``). Takes
            precedence over ``balanced_sampler``; rows are drawn into batches in
            proportion to their weight.
        **trainer_kwargs : Additional keyword arguments for PyTorch Lightning's Trainer class.


        Returns
        -------
        self : object
            The fitted classifier.
        """

        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)

        loss_fct, sampler = _resolve_loss_and_sampler(
            loss_fct, class_weight, balanced_sampler, sample_weight, y, self.classes_, num_classes
        )

        return super().fit(
            X=X,
            y=y,
            regression=False,
            val_size=val_size,
            X_val=X_val,
            y_val=y_val,
            embeddings=embeddings,
            embeddings_val=embeddings_val,
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
            num_classes=num_classes,
            loss_fct=loss_fct,
            sampler=sampler,
            **trainer_kwargs,
        )

    def predict(self, X, embeddings=None, device=None):
        """Predicts target labels for the given input samples.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The input samples for which to predict target values.

        Returns
        -------
        predictions : ndarray, shape (n_samples,)
            The predicted class labels.
        """
        X = self._validate_predict_input(X)
        if self.task_model is None:
            raise not_fitted_error(type(self).__name__, "predict")

        self._emit_event("predict_started", n_samples=len(X))

        # Preprocess the data using the data module
        if self.data_module is None:
            raise not_fitted_error(type(self).__name__, "predict")
        self.data_module.assign_predict_dataset(X, embeddings)

        # Set model to evaluation mode
        self.task_model.eval()

        # Perform inference using PyTorch Lightning's predict function
        if self.trainer is None:
            raise not_fitted_error(type(self).__name__, "predict")
        logits_list = self.trainer.predict(self.task_model, self.data_module)

        # Concatenate predictions from all batches
        logits = torch.cat(logits_list, dim=0)  # type: ignore

        # Check if ensemble is used
        if getattr(self.estimator, "returns_ensemble", False):  # If using ensemble
            logits = logits.mean(dim=1)  # Average over ensemble dimension
            if logits.dim() == 1:  # Ensure correct shape
                logits = logits.unsqueeze(1)

        # Check the shape of the logits to determine binary or multi-class classification
        if logits.shape[1] == 1:
            # Binary classification
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).long().view(-1)
        else:
            # Multi-class classification
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        # Convert predictions to NumPy array and return
        predicted_indices = predictions.cpu().numpy()
        classes = getattr(self, "classes_", None)
        if classes is not None and len(classes) > 0:
            result = classes[predicted_indices]
        else:
            result = predicted_indices
        self._emit_event("predict_completed")
        return result

    def predict_proba(self, X, embeddings=None, device=None):
        """Predicts class probabilities for the given input samples.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The input samples for which to predict class probabilities.

        Returns
        -------
        probabilities : ndarray, shape (n_samples, n_classes)
            The predicted class probabilities.
        """
        X = self._validate_predict_input(X)
        if self.task_model is None:
            raise not_fitted_error(type(self).__name__, "predict_proba")

        # Preprocess the data using the data module
        if self.data_module is None:
            raise not_fitted_error(type(self).__name__, "predict_proba")
        self.data_module.assign_predict_dataset(X, embeddings)

        # Set model to evaluation mode
        self.task_model.eval()

        # Perform inference using PyTorch Lightning's predict function
        if self.trainer is None:
            raise not_fitted_error(type(self).__name__, "predict_proba")
        logits_list = self.trainer.predict(self.task_model, self.data_module)

        # Concatenate predictions from all batches
        logits = torch.cat(logits_list, dim=0)  # type: ignore[arg-type]

        # Check if ensemble is used
        if getattr(self.estimator, "returns_ensemble", False):  # If using ensemble
            logits = logits.mean(dim=1)  # Average over ensemble dimension
            if logits.dim() == 1:  # Ensure correct shape
                logits = logits.unsqueeze(1)

        # Compute probabilities
        if logits.shape[1] > 1:
            probabilities = torch.softmax(logits, dim=1)  # Multi-class classification
        else:
            positive = torch.sigmoid(logits).view(-1, 1)
            probabilities = torch.cat([1.0 - positive, positive], dim=1)

        # Convert probabilities to NumPy array and return
        return probabilities.cpu().numpy()

    def evaluate(self, X, y_true, embeddings=None, metrics=None):
        """Evaluate the model on the given data using specified metrics.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples to predict.
        y_true : array-like of shape (n_samples,)
            The true class labels.
        embeddings : array-like or list, optional
            Embeddings for unstructured data inputs.
        metrics : dict, optional
            A ``{name: callable}`` dictionary where each callable has the
            signature ``metric(y_true, y_pred) -> float``.  Each callable may
            be a :class:`~deeptab.metrics.DeepTabMetric` instance or any plain
            callable.  Metrics that need probability scores (e.g. AUROC, LogLoss)
            should accept the 2-D ``predict_proba`` output as ``y_pred``;
            metrics that need class labels (e.g. Accuracy, F1) should accept
            the 1-D ``predict`` output.

            For :class:`~deeptab.metrics.DeepTabMetric` instances, the method
            inspects the ``name`` attribute to decide which prediction format
            to supply: probability-based metrics (``auroc``, ``auprc``,
            ``log_loss``, ``brier``, ``ece``) receive ``predict_proba`` output;
            all others receive ``predict`` output.

            If ``None``, defaults to the registry defaults for
            ``"classification"`` (Accuracy, AUROC, LogLoss).

        Returns
        -------
        scores : dict
            ``{metric_name: score}`` dictionary.
        """
        if metrics is None:
            metrics = get_default_metrics_dict("classification")

        # Metric names that work on probability scores
        _PROBA_NAMES = {"auroc", "auprc", "log_loss", "brier", "ece"}

        # Determine which prediction types are actually needed
        needs_proba = any((getattr(fn, "name", None) in _PROBA_NAMES) for fn in metrics.values())
        needs_labels = any((getattr(fn, "name", None) not in _PROBA_NAMES) for fn in metrics.values())

        probabilities = self.predict_proba(X, embeddings) if needs_proba else None
        predictions = self.predict(X, embeddings) if needs_labels else None

        scores = {}
        for metric_name, metric_func in metrics.items():
            use_proba = getattr(metric_func, "name", None) in _PROBA_NAMES
            preds = probabilities if use_proba else predictions
            if preds is None:
                scores[metric_name] = float("nan")
                continue
            try:
                scores[metric_name] = metric_func(y_true, preds)
            except Exception as exc:
                warnings.warn(f"Metric '{metric_name}' failed: {exc}", RuntimeWarning, stacklevel=2)
                scores[metric_name] = float("nan")

        return scores

    def score(self, X, y, embeddings=None, metric=None):
        """Calculate the score of the model using the specified metric.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples to predict.
        y : array-like of shape (n_samples,)
            The true class labels against which to evaluate the predictions.
        metric : tuple or callable, optional
            A tuple containing the metric function and a boolean indicating whether
            the metric requires probability scores (True) or class labels (False).
            If omitted, accuracy is used to match scikit-learn classifier behavior.

        Returns
        -------
        score : float
            The score calculated using the specified metric.
        """
        if metric is None:
            return accuracy_score(y, self.predict(X, embeddings))

        if isinstance(metric, tuple):
            metric_func, use_proba = metric
        else:
            metric_func, use_proba = metric, False

        if use_proba:
            probabilities = self.predict_proba(X, embeddings)
            return metric_func(y, probabilities)
        else:
            predictions = self.predict(X, embeddings)
            return metric_func(y, predictions)

    def pretrain(
        self,
        pretrain_epochs=15,
        k_neighbors=10,
        temperature=0.1,
        save_path="pretrained_embeddings.pth",
        lr=1e-3,
        use_positive=True,
        use_negative=False,
        pool_sequence=True,
    ):
        """
        Pretrains the embedding layer of the model using a contrastive learning approach.

        This method performs pretraining by optimizing the embeddings with respect to
        neighborhood structure in the feature space. The embeddings are saved after training.

        Parameters
        ----------
        pretrain_epochs : int, default=15
            Number of epochs to run pretraining.
        k_neighbors : int, default=10
            Number of neighbors used in the contrastive loss computation.
        temperature : float, default=0.1
            Temperature parameter for contrastive loss scaling.
        save_path : str, default="pretrained_embeddings.pth"
            Path to save the pretrained embeddings.
        lr : float, default=1e-3
            Learning rate for the pretraining optimizer.
        use_positive : bool, default=True
            Whether to include positive pairs in contrastive learning.
        use_negative : bool, default=False
            Whether to include negative pairs in contrastive learning.
        pool_sequence : bool, default=True
            Whether to apply sequence pooling before computing contrastive loss.

        Raises
        ------
        ValueError
            If the model has not been built before calling this method.
        ValueError
            If the model does not contain an embedding layer.

        Notes
        -----
        - This function requires that `self.build_model()` has been called beforehand.
        - The pretraining method uses `self.task_model.estimator.embedding_layer`.
        - The method invokes `super()._pretrain()` with regression mode enabled.

        """
        if not self.built:
            raise ValueError("The model has not been built yet. Call model.build_model(**args) first.")

        if not hasattr(self.task_model.estimator, "embedding_layer"):  # type: ignore[union-attr]
            raise ValueError("The model does not have an embedding layer")

        if self.data_module is None:
            raise not_fitted_error(type(self).__name__, "_pretrain")
        self.data_module.setup("fit")

        super()._pretrain(
            self.task_model.estimator,  # type: ignore[union-attr]
            self.data_module,
            pretrain_epochs=pretrain_epochs,
            k_neighbors=k_neighbors,
            temperature=temperature,
            save_path=save_path,
            regression=False,
            lr=lr,
            use_positive=use_positive,
            use_negative=use_negative,
            pool_sequence=pool_sequence,
        )

    def optimize_hparams(
        self,
        X,
        y,
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
            embeddings=embeddings,
            embeddings_val=embeddings_val,
            time=time,
            max_epochs=max_epochs,
            prune_by_epoch=prune_by_epoch,
            prune_epoch=prune_epoch,
            fixed_params=fixed_params,
            custom_search_space=custom_search_space,
            **optimize_kwargs,
        )
