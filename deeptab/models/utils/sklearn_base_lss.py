import warnings
from collections.abc import Callable

import lightning as pl
import numpy as np
import pandas as pd
import properscoring as ps
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from pretab.preprocessor import Preprocessor
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...base_models.utils.lightning_wrapper import TaskModel
from ...data_utils.datamodule import MambularDataModule
from ...utils.distributional_metrics import (
    beta_brier_score,
    dirichlet_error,
    gamma_deviance,
    inverse_gamma_loss,
    negative_binomial_deviance,
    poisson_deviance,
    student_t_loss,
)
from ...utils.distributions import (
    BetaDistribution,
    CategoricalDistribution,
    DirichletDistribution,
    GammaDistribution,
    InverseGammaDistribution,
    JohnsonSuDistribution,
    NegativeBinomialDistribution,
    NormalDistribution,
    PoissonDistribution,
    Quantile,
    StudentTDistribution,
)


class SklearnBaseLSS(BaseEstimator):
    def __init__(self, model, config, **kwargs):
        self.preprocessor_arg_names = [
            "n_bins",
            "feature_preprocessing",
            "numerical_preprocessing",
            "categorical_preprocessing",
            "use_decision_tree_bins",
            "binning_strategy",
            "task",
            "cat_cutoff",
            "treat_all_integers_as_numerical",
            "degree",
            "scaling_strategy",
            "n_knots",
            "use_decision_tree_knots",
            "knots_strategy",
            "spline_implementation",
        ]

        self.config_kwargs = {
            k: v for k, v in kwargs.items() if k not in self.preprocessor_arg_names and not k.startswith("optimizer")
        }
        self.config = config(**self.config_kwargs)

        preprocessor_kwargs = {k: v for k, v in kwargs.items() if k in self.preprocessor_arg_names}

        self.preprocessor = Preprocessor(**preprocessor_kwargs)
        self.task_model = None
        self.estimator = model
        self.built = False

        # Raise a warning if task is set to 'classification'
        if preprocessor_kwargs.get("task") == "classification":
            warnings.warn(
                "The task is set to 'classification'. Be aware of your preferred distribution,that \
                    this might lead to unsatisfactory results.",
                UserWarning,
                stacklevel=2,
            )

        self.optimizer_type = kwargs.get("optimizer_type", "Adam")

        self.optimizer_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["lr", "weight_decay", "patience", "lr_patience", "optimizer_type"]
            and k.startswith("optimizer_")
        }

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = {}
        params.update(self.config_kwargs)

        if deep:
            preprocessor_params = {"prepro__" + key: value for key, value in self.preprocessor.get_params().items()}  # type: ignore[attr-defined]
            params.update(preprocessor_params)

        return params

    def set_params(self, **parameters):
        """Set the parameters of this estimator.

        Parameters
        ----------
        **parameters : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        config_params = {k: v for k, v in parameters.items() if not k.startswith("prepro__")}
        preprocessor_params = {k.split("__")[1]: v for k, v in parameters.items() if k.startswith("prepro__")}

        if config_params:
            self.config_kwargs.update(config_params)
            if self.config is not None:
                for key, value in config_params.items():
                    setattr(self.config, key, value)
            else:
                self.config = self.config_class(**self.config_kwargs)  # type: ignore

        if preprocessor_params:
            self.preprocessor.set_params(**preprocessor_params)  # type: ignore[attr-defined]

        return self

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
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if isinstance(y, pd.Series):
            y = y.values
        if X_val is not None:
            if not isinstance(X_val, pd.DataFrame):
                X_val = pd.DataFrame(X_val)
            if isinstance(y_val, pd.Series):
                y_val = y_val.values

        self.data_module = MambularDataModule(
            preprocessor=self.preprocessor,
            batch_size=batch_size,
            shuffle=shuffle,
            X_val=X_val,
            y_val=y_val,
            val_size=val_size,
            random_state=random_state,
            regression=False,
            **dataloader_kwargs,
        )

        self.data_module.preprocess_data(X, y, X_val, y_val, val_size=val_size, random_state=random_state)

        self.task_model = TaskModel(
            model_class=self.estimator,  # type: ignore
            num_classes=self.family.param_count,
            family=self.family,
            config=self.config,
            feature_information=(
                self.data_module.num_feature_info,
                self.data_module.cat_feature_info,
                self.data_module.embedding_feature_info,
            ),
            lr=lr if lr is not None else self.config.lr,
            lr_patience=(lr_patience if lr_patience is not None else self.config.lr_patience),
            lr_factor=lr_factor if lr_factor is not None else self.config.lr_factor,
            weight_decay=(weight_decay if weight_decay is not None else self.config.weight_decay),
            lss=True,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            optimizer_type=self.optimizer_type,
            optimizer_args=self.optimizer_kwargs,
        )

        self.built = True
        self.estimator = self.task_model.estimator

        return self

    def get_number_of_params(self, requires_grad=True):
        """Calculate the number of parameters in the model.

        Parameters
        ----------
        requires_grad : bool, optional
            If True, only count the parameters that require gradients (trainable parameters).
            If False, count all parameters. Default is True.

        Returns
        -------
        int
            The total number of parameters in the model.

        Raises
        ------
        ValueError
            If the model has not been built prior to calling this method.
        """
        if not self.built:
            raise ValueError("The model must be built before the number of parameters can be estimated")
        else:
            if requires_grad:
                return sum(p.numel() for p in self.task_model.parameters() if p.requires_grad)  # type: ignore
            else:
                return sum(p.numel() for p in self.task_model.parameters())  # type: ignore

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
        distribution_classes = {
            "normal": NormalDistribution,
            "poisson": PoissonDistribution,
            "gamma": GammaDistribution,
            "beta": BetaDistribution,
            "dirichlet": DirichletDistribution,
            "studentt": StudentTDistribution,
            "negativebinom": NegativeBinomialDistribution,
            "inversegamma": InverseGammaDistribution,
            "categorical": CategoricalDistribution,
            "quantile": Quantile,
            "johnsonsu": JohnsonSuDistribution,
        }

        if distributional_kwargs is None:
            distributional_kwargs = {}

        if family in distribution_classes:
            self.family = distribution_classes[family](**distributional_kwargs)
        else:
            raise ValueError(f"Unsupported family: {family}")

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
            if not self.built:
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
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[
                early_stop_callback,
                checkpoint_callback,
                ModelSummary(max_depth=2),
            ],
            **trainer_kwargs,
        )
        self.trainer.fit(self.task_model, self.data_module)  # type: ignore

        self.best_model_path = checkpoint_callback.best_model_path
        if self.best_model_path:
            torch.serialization.add_safe_globals([type(self.config)])
            checkpoint = torch.load(self.best_model_path, weights_only=False)
            self.task_model.load_state_dict(checkpoint["state_dict"])  # type: ignore

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
        # Ensure model and data module are initialized
        if self.task_model is None or self.data_module is None:
            raise ValueError("The model or data module has not been fitted yet.")

        # Preprocess the data using the data module
        self.data_module.assign_predict_dataset(X)

        # Set model to evaluation mode
        self.task_model.eval()

        # Perform inference using PyTorch Lightning's predict function
        predictions_list = self.trainer.predict(self.task_model, self.data_module)

        # Concatenate predictions from all batches
        predictions = torch.cat(predictions_list, dim=0)  # type: ignore[arg-type]

        # Check if ensemble is used
        if getattr(self.estimator, "returns_ensemble", False):  # If using ensemble
            predictions = predictions.mean(dim=1)  # Average over ensemble dimension

        if not raw:
            result = self.task_model.family(predictions).cpu().numpy()  # type: ignore
            return result
        else:
            return predictions.cpu().numpy()

    def evaluate(self, X, y_true, metrics=None, distribution_family=None):
        """Evaluate the model on the given data using specified metrics.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples to predict.
        y_true : array-like of shape (n_samples,)
            The true class labels against which to evaluate the predictions.
        metrics : dict
            A dictionary where keys are metric names and values are tuples containing the metric function
            and a boolean indicating whether the metric requires probability scores (True) or class labels (False).
        distribution_family : str, optional
            Specifies the distribution family the model is predicting for. If None, it will attempt to infer based
            on the model's settings.


        Returns
        -------
        scores : dict
            A dictionary with metric names as keys and their corresponding scores as values.


        Notes
        -----
        This method uses either the `predict` or `predict_proba` method depending on the metric requirements.
        """
        # Infer distribution family from model settings if not provided
        if distribution_family is None:
            distribution_family = getattr(self.task_model, "distribution_family", "normal")

        # Setup default metrics if none are provided
        if metrics is None:
            metrics = self.get_default_metrics(distribution_family)

        # Make predictions
        predictions = self.predict(X, raw=False)

        # Initialize dictionary to store results
        scores = {}

        # Compute each metric
        for metric_name, metric_func in metrics.items():
            scores[metric_name] = metric_func(y_true, predictions)

        return scores

    def get_default_metrics(self, distribution_family):
        """Provides default metrics based on the distribution family.

        Parameters
        ----------
        distribution_family : str
            The distribution family for which to provide default metrics.


        Returns
        -------
        metrics : dict
            A dictionary of default metric functions.
        """
        default_metrics = {
            "normal": {
                "MSE": lambda y, pred: mean_squared_error(y, pred[:, 0]),
                "CRPS": lambda y, pred: np.mean(
                    [ps.crps_gaussian(y[i], mu=pred[i, 0], sig=np.sqrt(pred[i, 1])) for i in range(len(y))]
                ),
            },
            "poisson": {"Poisson Deviance": poisson_deviance},
            "gamma": {"Gamma Deviance": gamma_deviance},
            "beta": {"Brier Score": beta_brier_score},
            "dirichlet": {"Dirichlet Error": dirichlet_error},
            "studentt": {"Student-T Loss": student_t_loss},
            "negativebinom": {"Negative Binomial Deviance": negative_binomial_deviance},
            "inversegamma": {"Inverse Gamma Loss": inverse_gamma_loss},
            "categorical": {"Accuracy": accuracy_score},
        }
        return default_metrics.get(distribution_family, {})

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
        score = self.task_model.family.evaluate_nll(y, predictions)  # type: ignore
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
        if self.task_model is None or self.data_module is None:
            raise ValueError("The model or data module has not been fitted yet.")
        encoded_dataset = self.data_module.preprocess_new_data(X)

        data_loader = DataLoader(encoded_dataset, batch_size=batch_size, shuffle=False)

        # Process data in batches
        encoded_outputs = []
        for num_features, cat_features in tqdm(data_loader):
            embeddings = self.task_model.estimator.encode(num_features, cat_features)  # type: ignore[union-attr]  # Call your encode function
            encoded_outputs.append(embeddings)

        # Concatenate all encoded outputs
        encoded_outputs = torch.cat(encoded_outputs, dim=0)

        return encoded_outputs

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

        return super().optimize_hparams(  # type: ignore[attr-defined]
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
