import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss

from .sklearn_parent import SklearnBase


class SklearnBaseClassifier(SklearnBase):
    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)
        # Raise a warning if task is set to 'classification'
        preprocessor_kwargs = {k: v for k, v in kwargs.items() if k in self.preprocessor_arg_names}

        if preprocessor_kwargs.get("task") == "regression":
            warnings.warn(
                "The task is set to 'regression'. The Classifier is designed for classification tasks.",
                UserWarning,
                stacklevel=2,
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



        Returns
        -------
        self : object
            The built classifier.
        """

        num_classes = len(np.unique(y))

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
        **trainer_kwargs : Additional keyword arguments for PyTorch Lightning's Trainer class.


        Returns
        -------
        self : object
            The fitted classifier.
        """

        num_classes = len(np.unique(y))
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
        # Ensure model and data module are initialized
        if self.task_model is None or self.data_module is None:
            raise ValueError("The model or data module has not been fitted yet.")

        # Preprocess the data using the data module
        self.data_module.assign_predict_dataset(X, embeddings)

        # Set model to evaluation mode
        self.task_model.eval()

        # Perform inference using PyTorch Lightning's predict function
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
            predictions = (probabilities > 0.5).long().squeeze()
        else:
            # Multi-class classification
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        # Convert predictions to NumPy array and return
        return predictions.cpu().numpy()

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
        # Ensure model and data module are initialized
        if self.task_model is None or self.data_module is None:
            raise ValueError("The model or data module has not been fitted yet.")

        # Preprocess the data using the data module
        self.data_module.assign_predict_dataset(X, embeddings)

        # Set model to evaluation mode
        self.task_model.eval()

        # Perform inference using PyTorch Lightning's predict function
        logits_list = self.trainer.predict(self.task_model, self.data_module)

        # Concatenate predictions from all batches
        logits = torch.cat(logits_list, dim=0)

        # Check if ensemble is used
        if getattr(self.estimator, "returns_ensemble", False):  # If using ensemble
            logits = logits.mean(dim=1)  # Average over ensemble dimension
            if logits.dim() == 1:  # Ensure correct shape
                logits = logits.unsqueeze(1)

        # Compute probabilities
        if logits.shape[1] > 1:
            probabilities = torch.softmax(logits, dim=1)  # Multi-class classification
        else:
            probabilities = torch.sigmoid(logits)  # Binary classification

        # Convert probabilities to NumPy array and return
        return probabilities.cpu().numpy()

    def evaluate(self, X, y_true, embeddings=None, metrics=None):
        """Evaluate the model on the given data using specified metrics.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples to predict.
        y_true : array-like of shape (n_samples,)
            The true class labels against which to evaluate the predictions.
        embneddings : array-like or list of shape(n_samples, dimension)
            List or array with embeddings for unstructured data inputs
        metrics : dict
            A dictionary where keys are metric names and values are tuples containing the metric function
            and a boolean indicating whether the metric requires probability scores (True) or class labels (False).


        Returns
        -------
        scores : dict
            A dictionary with metric names as keys and their corresponding scores as values.


        Notes
        -----
        This method uses either the `predict` or `predict_proba` method depending on the metric requirements.
        """
        # Ensure input is in the correct format
        if metrics is None:
            metrics = {"Accuracy": (accuracy_score, False)}

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Initialize dictionary to store results
        scores = {}

        # Generate class probabilities if any metric requires them
        if any(use_proba for _, use_proba in metrics.values()):
            probabilities = self.predict_proba(X, embeddings)

        # Generate class labels if any metric requires them
        if any(not use_proba for _, use_proba in metrics.values()):
            predictions = self.predict(X, embeddings)

        # Compute each metric
        for metric_name, (metric_func, use_proba) in metrics.items():
            if use_proba:
                scores[metric_name] = metric_func(y_true, probabilities)  # type: ignore
            else:
                scores[metric_name] = metric_func(y_true, predictions)  # type: ignore

        return scores

    def score(self, X, y, embeddings=None, metric=(log_loss, True)):
        """Calculate the score of the model using the specified metric.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples to predict.
        y : array-like of shape (n_samples,)
            The true class labels against which to evaluate the predictions.
        metric : tuple, default=(log_loss, True)
            A tuple containing the metric function and a boolean indicating whether
            the metric requires probability scores (True) or class labels (False).

        Returns
        -------
        score : float
            The score calculated using the specified metric.
        """
        metric_func, use_proba = metric

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

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

        if not hasattr(self.task_model.estimator, "embedding_layer"):
            raise ValueError("The model does not have an embedding layer")

        self.data_module.setup("fit")

        super()._pretrain(
            self.task_model.estimator,
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
