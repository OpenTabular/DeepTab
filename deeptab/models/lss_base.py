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

from deeptab.configs.core import PreprocessingConfig, TrainerConfig
from deeptab.core.inspection import InspectionMixin
from deeptab.core.serialization import build_artifact_metadata, restore_loaded_metadata
from deeptab.data.datamodule import TabularDataModule
from deeptab.distributions.base import (
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
from deeptab.distributions.metrics import (
    beta_brier_score,
    dirichlet_error,
    gamma_deviance,
    inverse_gamma_loss,
    negative_binomial_deviance,
    poisson_deviance,
    student_t_loss,
)
from deeptab.training import TaskModel

DISTRIBUTION_CLASSES = {
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


class SklearnBaseLSS(InspectionMixin, BaseEstimator):
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
        self.random_state = random_state
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

        if model_config is not None or preprocessing_config is not None or trainer_config is not None:
            # ---- New split-config path ----
            self.model_config = model_config
            self.preprocessing_config = (
                preprocessing_config if preprocessing_config is not None else PreprocessingConfig()
            )
            self.trainer_config = trainer_config if trainer_config is not None else TrainerConfig()

            if model_config is not None:
                self.config_kwargs = model_config.get_params(deep=False)
                self.config = model_config
            else:
                self.config_kwargs = {}
                self.config = config()

            self.preprocessor_kwargs = self.preprocessing_config.to_preprocessor_kwargs()
            self.preprocessor = Preprocessor(**self.preprocessor_kwargs)

            self.optimizer_type = self.trainer_config.optimizer_type
            self.optimizer_kwargs = {}
        else:
            # ---- Legacy flat-kwargs path (backward compat) ----
            self.model_config = None
            self.preprocessing_config = None
            self.trainer_config = None

            self.config_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in self.preprocessor_arg_names and not k.startswith("optimizer")
            }
            self.config = config(**self.config_kwargs)

            self.preprocessor_kwargs = {k: v for k, v in kwargs.items() if k in self.preprocessor_arg_names}
            self.preprocessor = Preprocessor(**self.preprocessor_kwargs)

            # Raise a warning if task is set to 'classification'
            if self.preprocessor_kwargs.get("task") == "classification":
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

        self.task_model = None
        self.estimator = model
        self.built = False

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
        if self.model_config is not None or self.preprocessing_config is not None or self.trainer_config is not None:
            # New split-config style
            params = {
                "model_config": self.model_config,
                "preprocessing_config": self.preprocessing_config,
                "trainer_config": self.trainer_config,
                "random_state": self.random_state,
            }
            if deep:
                if self.model_config is not None:
                    for k, v in self.model_config.get_params(deep=False).items():
                        params[f"model_config__{k}"] = v
                if self.preprocessing_config is not None:
                    for k, v in self.preprocessing_config.get_params(deep=False).items():
                        params[f"preprocessing_config__{k}"] = v
                if self.trainer_config is not None:
                    for k, v in self.trainer_config.get_params(deep=False).items():
                        params[f"trainer_config__{k}"] = v
            return params

        # Legacy flat-kwargs style
        params = {}
        params.update(self.config_kwargs)

        if deep:
            get_params_fn = getattr(self.preprocessor, "get_params", None)
            if get_params_fn is not None:
                preprocessor_params = {"prepro__" + key: value for key, value in get_params_fn().items()}
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
        if self.model_config is not None or self.preprocessing_config is not None or self.trainer_config is not None:
            # New split-config style
            direct_params = {}
            model_config_params = {}
            preprocessing_config_params = {}
            trainer_config_params = {}

            for k, v in parameters.items():
                if k.startswith("model_config__"):
                    model_config_params[k[len("model_config__") :]] = v
                elif k.startswith("preprocessing_config__"):
                    preprocessing_config_params[k[len("preprocessing_config__") :]] = v
                elif k.startswith("trainer_config__"):
                    trainer_config_params[k[len("trainer_config__") :]] = v
                else:
                    direct_params[k] = v

            for k, v in direct_params.items():
                if k == "model_config":
                    self.model_config = v
                    if v is not None:
                        self.config = v
                        self.config_kwargs = v.get_params(deep=False)
                elif k == "preprocessing_config":
                    self.preprocessing_config = v
                    if v is not None:
                        self.preprocessor_kwargs = v.to_preprocessor_kwargs()
                        self.preprocessor = Preprocessor(**self.preprocessor_kwargs)
                elif k == "trainer_config":
                    self.trainer_config = v
                    if v is not None:
                        self.optimizer_type = v.optimizer_type
                elif k == "random_state":
                    self.random_state = v

            if model_config_params and self.model_config is not None:
                self.model_config.set_params(**model_config_params)
                self.config_kwargs = self.model_config.get_params(deep=False)
            if preprocessing_config_params and self.preprocessing_config is not None:
                self.preprocessing_config.set_params(**preprocessing_config_params)
                self.preprocessor_kwargs = self.preprocessing_config.to_preprocessor_kwargs()
                self.preprocessor = Preprocessor(**self.preprocessor_kwargs)
            if trainer_config_params and self.trainer_config is not None:
                self.trainer_config.set_params(**trainer_config_params)
                self.optimizer_type = self.trainer_config.optimizer_type

            return self

        # Legacy flat-kwargs style
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
            self.preprocessor_kwargs.update(preprocessor_params)
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

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.input_columns_ = list(X.columns)
        self.classes_ = np.unique(y) if getattr(self, "family_name", None) == "categorical" else None
        if isinstance(y, pd.Series):
            y = y.values
        if X_val is not None:
            if not isinstance(X_val, pd.DataFrame):
                X_val = pd.DataFrame(X_val)
            if isinstance(y_val, pd.Series):
                y_val = y_val.values

        self.data_module = TabularDataModule(
            preprocessor=self.preprocessor,
            batch_size=batch_size,
            shuffle=shuffle,
            X_val=X_val,
            y_val=y_val,
            val_size=val_size,
            random_state=random_state,
            regression=getattr(self, "family_name", None) != "categorical",
            **dataloader_kwargs,
        )
        self.data_module.input_columns_ = self.input_columns_

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
            lr=lr if lr is not None else getattr(self.config, "lr", None),
            lr_patience=(lr_patience if lr_patience is not None else getattr(self.config, "lr_patience", None)),
            lr_factor=lr_factor if lr_factor is not None else getattr(self.config, "lr_factor", None),
            weight_decay=(weight_decay if weight_decay is not None else getattr(self.config, "weight_decay", None)),
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

        # When random_state was fixed at construction time, honour it
        if self.random_state is not None:
            random_state = self.random_state

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
            self.family_name = family
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

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``"model.pt"``).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if not getattr(self, "is_fitted_", False):
            raise ValueError("Model must be fitted before saving.")
        if self.task_model is None:
            raise RuntimeError("task_model is unexpectedly None after fitting.")
        task = "classification" if self.family_name == "categorical" else "distributional_regression"
        artifact_metadata = build_artifact_metadata(
            estimator=self,
            model_class=type(self.estimator),
            config=self.config,
            data_module=self.data_module,
            preprocessor=self.preprocessor,
            preprocessor_kwargs=getattr(self, "preprocessor_kwargs", {}),
            task=task,
            regression=self.data_module.regression,
            lss=True,
            family=self.family_name,
            num_classes=self.task_model.num_classes,
            classes_=getattr(self, "classes_", None),
        )
        feature_schema = artifact_metadata["feature_schema"]
        bundle = {
            "_class": type(self),
            "config": self.config,
            "config_kwargs": self.config_kwargs,
            "preprocessor_kwargs": getattr(self, "preprocessor_kwargs", {}),
            "preprocessor": self.preprocessor,
            "feature_info": {
                "num": self.data_module.num_feature_info,
                "cat": self.data_module.cat_feature_info,
                "emb": self.data_module.embedding_feature_info,
            },
            "batch_size": self.data_module.batch_size,
            "regression": self.data_module.regression,
            "model_class": type(self.estimator),
            "num_classes": self.task_model.num_classes,
            "lss": True,
            "family": self.family_name,
            "optimizer_type": self.optimizer_type,
            "optimizer_kwargs": self.optimizer_kwargs,
            "lr": self.task_model.lr,
            "lr_patience": self.task_model.lr_patience,
            "lr_factor": self.task_model.lr_factor,
            "weight_decay": self.task_model.weight_decay,
            "task_model_state_dict": self.task_model.state_dict(),
            "artifact_metadata": artifact_metadata,
            "architecture_metadata": artifact_metadata["architecture"],
            "feature_schema": feature_schema,
            "input_columns": feature_schema["column_order"],
            "preprocessing_metadata": artifact_metadata["preprocessing"],
            "task_info": artifact_metadata["task"],
            "classes_": getattr(self, "classes_", None),
            "versions": artifact_metadata["versions"],
        }
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
            A fully reconstructed, ready-to-predict estimator. Newer
            artifacts also expose ``artifact_metadata_``,
            ``architecture_metadata_``, ``feature_schema_``,
            ``input_columns_``, ``task_info_``, ``classes_``, and
            ``versions_`` attributes after loading.
        """
        bundle = torch.load(path, weights_only=False)

        obj = bundle["_class"].__new__(bundle["_class"])
        obj.config = bundle["config"]
        obj.config_kwargs = bundle["config_kwargs"]
        obj.preprocessor_kwargs = bundle.get("preprocessor_kwargs", {})
        obj.preprocessor = bundle["preprocessor"]
        obj.optimizer_type = bundle["optimizer_type"]
        obj.optimizer_kwargs = bundle["optimizer_kwargs"]
        obj.built = True
        obj.is_fitted_ = True
        obj.model_config = None
        obj.preprocessing_config = None
        obj.trainer_config = None
        obj.random_state = None
        obj.family = DISTRIBUTION_CLASSES[bundle["family"]]()
        obj.family_name = bundle["family"]
        obj.preprocessor_arg_names = [
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

        obj.data_module = TabularDataModule(
            preprocessor=bundle["preprocessor"],
            batch_size=bundle["batch_size"],
            shuffle=False,
            regression=bundle["regression"],
        )
        obj.data_module.num_feature_info = bundle["feature_info"]["num"]
        obj.data_module.cat_feature_info = bundle["feature_info"]["cat"]
        obj.data_module.embedding_feature_info = bundle["feature_info"]["emb"]
        obj.data_module.input_columns_ = bundle.get("input_columns")

        obj.task_model = TaskModel(
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
