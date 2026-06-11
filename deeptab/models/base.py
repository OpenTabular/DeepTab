from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import lightning as pl
import numpy as np
from pretab.preprocessor import Preprocessor
from sklearn.base import BaseEstimator

from deeptab.configs.core import PreprocessingConfig, TrainerConfig
from deeptab.core.default_factories import DefaultDataModuleFactory, DefaultTaskModelFactory
from deeptab.core.exceptions import DataError, target_nan_error, target_range_error, warn_data, xy_length_mismatch_error
from deeptab.core.inspection import InspectionMixin
from deeptab.core.interfaces import IDataModule, IDataModuleFactory, ITaskModel, ITaskModelFactory
from deeptab.models._mixins import (
    _FitMixin,
    _HyperparameterMixin,
    _ObservabilityMixin,
    _PredictMixin,
    _SerializationMixin,
)

if TYPE_CHECKING:
    from deeptab.core.observability import ObservabilityConfig


def _validate_fit_inputs(
    X,
    y,
    regression: bool,
    family: str | None = None,
) -> None:
    """Validate X and y before any preprocessing or model building.

    Raises
    ------
    EmptyDataError
        If X is empty (caught later by ensure_dataframe).
    DataError
        If len(X) != len(y), y contains NaN, or y violates the distribution
        family's range constraint.
    """
    n_X = len(X)
    n_y = len(y)
    if n_X != n_y:
        raise xy_length_mismatch_error(n_X, n_y)

    if hasattr(X, "ndim") and X.ndim == 1:
        raise ValueError(
            "Expected a 2D array for X, got a 1D array instead. "
            "Reshape your data using X.reshape(-1, 1) for a single feature."
        )

    y_arr = np.asarray(y)
    if y_arr.ndim <= 2 and np.issubdtype(y_arr.dtype, np.floating) and np.isnan(y_arr).any():
        raise target_nan_error()

    # Distribution family range constraints
    if family is not None:
        family_lower = family.lower()
        if family_lower in {"poisson", "negativebinom"} and (y_arr < 0).any():
            raise target_range_error(family, "non-negative")
        if family_lower in {"gamma", "inversegaussian"} and (y_arr <= 0).any():
            raise target_range_error(family, "strictly positive")
        if family_lower == "binomial" and not np.all((y_arr == 0) | (y_arr == 1)):
            raise target_range_error(family, "binary (0 or 1)")

    # Warn about high-NaN columns
    if hasattr(X, "isna"):
        nan_rate = X.isna().mean()
        high_nan = nan_rate[nan_rate > 0.5].index.tolist()
        if high_nan:
            warn_data(
                f"Columns with >50% missing values: {[str(c) for c in high_nan]}. "
                "Consider dropping or imputing them before calling fit().",
                stacklevel=5,
            )


def _raise_flat_param_error(kwargs: dict, estimator_name: str) -> None:
    """Raise a helpful TypeError when flat kwargs are passed to a split-config estimator.

    DeepTab 2.0 no longer accepts flat model/training/preprocessing parameters in
    Classifier and Regressor constructors.  Pass them via the dedicated config objects.
    """
    param_list = ", ".join(f"'{k}'" for k in sorted(kwargs))
    # Infer the model-config class name from the estimator name.
    # e.g.  MLPClassifier → MLPConfig,  FTTransformerRegressor → FTTransformerConfig
    config_name = estimator_name
    for suffix in ("Classifier", "Regressor"):
        if config_name.endswith(suffix):
            config_name = config_name[: -len(suffix)] + "Config"
            break
    raise TypeError(
        f"{estimator_name}() received unexpected keyword arguments: {param_list}.\n"
        f"\n"
        f"DeepTab 2.0 no longer accepts flat model/training/preprocessing parameters.\n"
        f"Pass them through the split-config API instead:\n"
        f"\n"
        f"  from deeptab.configs import {config_name}, PreprocessingConfig, TrainerConfig\n"
        f"  model = {estimator_name}(\n"
        f"      model_config={config_name}(...),\n"
        f"      preprocessing_config=PreprocessingConfig(...),  # optional\n"
        f"      trainer_config=TrainerConfig(max_epochs=100, lr=1e-4),\n"
        f"  )\n"
    )


class SklearnBase(
    _ObservabilityMixin,
    _FitMixin,
    _PredictMixin,
    _SerializationMixin,
    _HyperparameterMixin,
    InspectionMixin,
    BaseEstimator,
):
    """Thin coordinator — all behaviour lives in the mixins.

    MRO:
        _ObservabilityMixin  →  _FitMixin  →  _PredictMixin
        → _SerializationMixin  →  _HyperparameterMixin
        → InspectionMixin  →  BaseEstimator
    """

    def __init__(
        self,
        model,
        config,
        model_config=None,
        preprocessing_config=None,
        trainer_config=None,
        random_state=None,
        observability_config: ObservabilityConfig | None = None,
        **kwargs,
    ):
        self.random_state = random_state
        self._preprocessor_arg_names = [
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

            if model_config is not None and hasattr(model_config, "get_params"):
                self._config_kwargs = model_config.get_params(deep=False)
                self.config = model_config
            else:
                self._config_kwargs = {}
                self.config = config()

            if hasattr(self.preprocessing_config, "to_preprocessor_kwargs"):
                self._preprocessor_kwargs = self.preprocessing_config.to_preprocessor_kwargs()
            else:
                self._preprocessor_kwargs = {}
            self._preprocessor = Preprocessor(**self._preprocessor_kwargs)

            self._optimizer_type = getattr(self.trainer_config, "optimizer_type", "Adam")
            self._optimizer_kwargs = {}
        else:
            # ---- Legacy flat-kwargs path (backward compat) ----
            self.model_config = None
            self.preprocessing_config = None
            self.trainer_config = None

            self._config_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in self._preprocessor_arg_names and not k.startswith("optimizer")
            }
            self.config = config(**self._config_kwargs)

            self._preprocessor_kwargs = {k: v for k, v in kwargs.items() if k in self._preprocessor_arg_names}
            self._preprocessor = Preprocessor(**self._preprocessor_kwargs)

            self._optimizer_type = kwargs.get("optimizer_type", "Adam")
            self._optimizer_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ["lr", "weight_decay", "patience", "lr_patience", "optimizer_type"]
                and k.startswith("optimizer_")
            }

        self._estimator = model
        self._task_model = None
        self._built = False
        # Fitted attributes (_data_module, _trainer, _best_model_path) are
        # initialised here so fit() never *adds* new public attributes.
        # input_columns_ is a proper fitted attribute (trailing _) set only
        # in fit() via set_input_feature_attributes(); not initialised here.
        self._data_module: IDataModule | None = None
        self._trainer: pl.Trainer | None = None
        self._best_model_path: str | None = None
        # Dependency-inversion factories (underscore-prefixed: ignored by
        # sklearn's get_params/set_params; clones always get fresh defaults).
        # Set via direct attribute assignment to inject test doubles:
        #   estimator._data_module_factory = MyFactory()
        self._data_module_factory: IDataModuleFactory = DefaultDataModuleFactory()
        self._task_model_factory: ITaskModelFactory = DefaultTaskModelFactory()
        # Observability — wire up backends if a config was provided.
        # Underscore-prefix: hidden from sklearn get_params/set_params/clone.
        self._observability_config: ObservabilityConfig | None = observability_config
        if observability_config is not None:
            self.configure_observability(observability_config)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        if self.model_config is not None or self.preprocessing_config is not None or self.trainer_config is not None:
            # New split-config style
            params = {
                "model_config": self.model_config,
                "preprocessing_config": self.preprocessing_config,
                "trainer_config": self.trainer_config,
                "random_state": self.random_state,
            }
            if deep:
                if self.model_config is not None and hasattr(self.model_config, "get_params"):
                    for k, v in self.model_config.get_params(deep=False).items():
                        params[f"model_config__{k}"] = v
                if self.preprocessing_config is not None and hasattr(self.preprocessing_config, "get_params"):
                    for k, v in self.preprocessing_config.get_params(deep=False).items():
                        params[f"preprocessing_config__{k}"] = v
                if self.trainer_config is not None and hasattr(self.trainer_config, "get_params"):
                    for k, v in self.trainer_config.get_params(deep=False).items():
                        params[f"trainer_config__{k}"] = v
            return params

        # Legacy flat-kwargs style
        params = {}
        params.update(self._config_kwargs)
        params.update(self._preprocessor_kwargs)
        if deep:
            get_params_fn = getattr(self._preprocessor, "get_params", None)
            if get_params_fn is not None:
                preprocessor_params = {
                    key: value for key, value in get_params_fn().items() if key in self._preprocessor_arg_names
                }
                params.update(preprocessor_params)
        return params

    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
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
                    if v is not None and hasattr(v, "get_params"):
                        self.config = v
                        self._config_kwargs = v.get_params(deep=False)
                elif k == "preprocessing_config":
                    self.preprocessing_config = v
                    if v is not None and hasattr(v, "to_preprocessor_kwargs"):
                        self._preprocessor_kwargs = v.to_preprocessor_kwargs()
                        self._preprocessor = Preprocessor(**self._preprocessor_kwargs)
                elif k == "trainer_config":
                    self.trainer_config = v
                    if v is not None and hasattr(v, "optimizer_type"):
                        self._optimizer_type = v.optimizer_type
                elif k == "random_state":
                    self.random_state = v

            if model_config_params and self.model_config is not None and hasattr(self.model_config, "set_params"):
                self.model_config.set_params(**model_config_params)
                self._config_kwargs = self.model_config.get_params(deep=False)
            if (
                preprocessing_config_params
                and self.preprocessing_config is not None
                and hasattr(self.preprocessing_config, "set_params")
            ):
                self.preprocessing_config.set_params(**preprocessing_config_params)
                self._preprocessor_kwargs = self.preprocessing_config.to_preprocessor_kwargs()
                self._preprocessor = Preprocessor(**self._preprocessor_kwargs)
            if trainer_config_params and self.trainer_config is not None and hasattr(self.trainer_config, "set_params"):
                self.trainer_config.set_params(**trainer_config_params)
                self._optimizer_type = self.trainer_config.optimizer_type

            return self

        # Legacy flat-kwargs style
        config_params = {k: v for k, v in parameters.items() if k not in self._preprocessor_arg_names}
        preprocessor_params = {k: v for k, v in parameters.items() if k in self._preprocessor_arg_names}

        if config_params:
            self._config_kwargs.update(config_params)

        if preprocessor_params:
            self._preprocessor_kwargs.update(preprocessor_params)
            self._preprocessor.set_params(**self._preprocessor_kwargs)  # type: ignore[attr-defined]

        return self

    def __sklearn_is_fitted__(self) -> bool:
        """sklearn hook: return True only after fit() has completed.

        Declaring this method prevents sklearn's ``check_is_fitted`` from
        inspecting attributes ending with ``_`` (e.g. ``input_columns_``,
        ``n_features_in_``) which exist even on unfitted estimators.
        """
        return bool(getattr(self, "is_fitted_", False))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["task_model"] = None  # Avoid serializing the task model
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._task_model = None  # Reinitialize task model
