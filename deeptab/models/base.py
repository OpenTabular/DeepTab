from __future__ import annotations

import warnings
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from typing import TYPE_CHECKING, ClassVar, cast

import lightning as pl
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import clone as sklearn_clone

from deeptab.configs.core import BaseModelConfig, PreprocessingConfig, TrainerConfig
from deeptab.core.default_factories import DefaultDataModuleFactory, DefaultTaskModelFactory
from deeptab.core.exceptions import (
    DataError,
    target_nan_error,
    target_range_error,
    warn_config,
    warn_data,
    xy_length_mismatch_error,
)
from deeptab.core.inspection import InspectionMixin
from deeptab.core.interfaces import IDataModule, IDataModuleFactory, ITaskModel, ITaskModelFactory
from deeptab.core.preprocessing import build_preprocessor
from deeptab.models._mixins import (
    _FitMixin,
    _HyperparameterMixin,
    _ObservabilityMixin,
    _PredictMixin,
    _SerializationMixin,
)

if TYPE_CHECKING:
    from deeptab.core.observability import ObservabilityConfig


def _warn_on_misplaced_configs(model_config, preprocessing_config, trainer_config) -> None:
    """Warn when a config object is passed to the wrong constructor slot.

    The constructor duck-types each config, so a misplaced config (for example a
    ``TrainerConfig`` handed to ``model_config``) would otherwise be accepted
    silently and then quietly ignored. This emits an advisory ``ConfigWarning``
    when a recognisably wrong config type is detected, without raising, so that
    legitimate duck-typed test doubles keep working.
    """
    slots = (
        ("model_config", model_config, BaseModelConfig),
        ("preprocessing_config", preprocessing_config, PreprocessingConfig),
        ("trainer_config", trainer_config, TrainerConfig),
    )
    known_config_types = (BaseModelConfig, PreprocessingConfig, TrainerConfig)
    for slot_name, value, expected_cls in slots:
        if value is None:
            continue
        # Only warn when the value is clearly *another* known config type;
        # unknown duck-typed objects (mocks, test doubles) are left alone.
        if isinstance(value, known_config_types) and not isinstance(value, expected_cls):
            warn_config(
                f"{type(value).__name__} was passed as '{slot_name}', but '{slot_name}' "
                f"expects a {expected_cls.__name__}. Configs are not reordered for you, "
                f"so this one will be misused or silently ignored. Pass it as its matching "
                f"argument instead.",
                stacklevel=4,
            )


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

    Concrete estimators declare the architecture and its default config class
    via the ``_model_cls`` and ``_config_cls`` class attributes instead of
    passing them through ``__init__``. This keeps the constructor signature
    limited to the public, sklearn-introspectable parameters.
    """

    # Set by concrete estimator subclasses (e.g. ``_model_cls = MLP``).
    _model_cls: ClassVar[type | None] = None
    _config_cls: ClassVar[type | None] = None
    # Task resolved from the estimator's own type, passed to build_preprocessor()
    # rather than trusted from user input. Regression is the default so LSS
    # estimators (which subclass SklearnBase directly) resolve sensibly before
    # fit() knows the distribution family; SklearnBaseClassifier overrides this.
    _task: ClassVar[str] = "regression"

    def __init__(
        self,
        model_config=None,
        preprocessing_config=None,
        trainer_config=None,
        observability_config: ObservabilityConfig | None = None,
        random_state=None,
    ):
        model_cls = type(self)._model_cls
        config_cls = type(self)._config_cls
        if model_cls is None or config_cls is None:
            raise TypeError(
                f"{type(self).__name__} must define the '_model_cls' and "
                "'_config_cls' class attributes (the architecture class and its "
                "default config class)."
            )
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
            _warn_on_misplaced_configs(model_config, preprocessing_config, trainer_config)
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
                self.config = config_cls()

            if hasattr(self.preprocessing_config, "to_preprocessor_kwargs"):
                self._preprocessor_kwargs = self.preprocessing_config.to_preprocessor_kwargs()
            else:
                self._preprocessor_kwargs = {}
            self._preprocessor = build_preprocessor(
                self.preprocessing_config,
                task=type(self)._task,
                random_state=random_state,
                observability_config=observability_config,
            )

            self._optimizer_type = getattr(self.trainer_config, "optimizer_type", "Adam")
            self._optimizer_kwargs = {}
        else:
            # ---- No configs provided: fall back to defaults ----
            self.model_config = None
            self.preprocessing_config = None
            self.trainer_config = None

            self._config_kwargs = {}
            self.config = config_cls()

            self._preprocessor_kwargs = {}
            self._preprocessor = build_preprocessor(
                None, task=type(self)._task, random_state=random_state, observability_config=observability_config
            )

            self._optimizer_type = "Adam"
            self._optimizer_kwargs = {}

        self._estimator = model_cls
        self._task_model = None
        self._built = False
        # Set by pretrain(); makes the next fit() continue from the pretrained
        # weights by default instead of rebuilding a fresh model (#446).
        self._is_pretrained = False
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
        # Only wire up for a genuine ObservabilityConfig; like the model and
        # preprocessing configs above, an unexpected value is stored as-is and
        # validation is deferred rather than raising inside __init__.
        self._apply_observability_config(observability_config)

    def _apply_observability_config(self, observability_config: ObservabilityConfig | None) -> None:
        """Store and wire up an observability config (shared by __init__ and set_params).

        Only wires up backends for a genuine ObservabilityConfig; like the model
        and preprocessing configs, an unexpected value is stored as-is and
        validation is deferred rather than raising here.
        """
        self._observability_config: ObservabilityConfig | None = observability_config
        if observability_config is not None and hasattr(observability_config, "structured_logging"):
            self.configure_observability(observability_config)

    @property
    def config(self):
        """The instantiated model config object backing this estimator.

        Stored on the private ``_config`` attribute so it stays out of
        sklearn's ``get_params``/``__init__`` introspection (it is derived
        from ``model_config``/``_model_cls`` rather than a constructor
        parameter), while remaining readable and settable as ``estimator.config``.
        """
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        if self.model_config is not None or self.preprocessing_config is not None or self.trainer_config is not None:
            # New split-config style
            params = {
                "model_config": self.model_config,
                "preprocessing_config": self.preprocessing_config,
                "trainer_config": self.trainer_config,
                "observability_config": getattr(self, "_observability_config", None),
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

        # No configs were supplied at construction time. `__init__` only ever
        # accepts `model_config`/`preprocessing_config`/`trainer_config`/
        # `random_state` as constructor arguments, so any additional overrides
        # applied via set_params() (e.g. `layer_sizes`) are surfaced here too
        # instead of being silently invisible to get_params() (GH #410).
        # __sklearn_clone__ (below) replays these through set_params() rather
        # than the constructor, so reporting them here does not break clone().
        return {
            "model_config": self.model_config,
            "preprocessing_config": self.preprocessing_config,
            "trainer_config": self.trainer_config,
            "observability_config": getattr(self, "_observability_config", None),
            "random_state": self.random_state,
            **self._config_kwargs,
            **self._preprocessor_kwargs,
        }

    def __sklearn_clone__(self):
        """Clone by replaying set_params() rather than calling cls(**get_params()).

        get_params() can report parameter overrides (e.g. ``layer_sizes`` applied
        through set_params()) that ``__init__`` does not accept as keyword
        arguments. scikit-learn's default clone builds the copy via
        ``type(self)(**self.get_params(deep=False))``, which would raise a
        ``TypeError`` in that case. Rebuilding from the real constructor
        arguments and replaying everything else through set_params() keeps
        clone() reliable for both the split-config and flat-kwargs styles
        (GH #410).
        """
        observability_config = cast(
            "ObservabilityConfig | None", sklearn_clone(getattr(self, "_observability_config", None), safe=False)
        )
        if self.model_config is not None or self.preprocessing_config is not None or self.trainer_config is not None:
            return type(self)(
                model_config=sklearn_clone(self.model_config) if self.model_config is not None else None,
                preprocessing_config=(
                    sklearn_clone(self.preprocessing_config) if self.preprocessing_config is not None else None
                ),
                trainer_config=sklearn_clone(self.trainer_config) if self.trainer_config is not None else None,
                observability_config=observability_config,
                random_state=self.random_state,
            )

        new_object = type(self)(
            observability_config=observability_config,
            random_state=self.random_state,
        )
        extra_params = {**self._config_kwargs, **self._preprocessor_kwargs}
        if extra_params:
            new_object.set_params(**extra_params)
        return new_object

    def set_params(self, **parameters):
        """Set the parameters of this estimator.

        Raises
        ------
        ValueError
            If a key in ``parameters`` does not name a recognised parameter.
        """
        if not parameters:
            return self

        if self.model_config is not None or self.preprocessing_config is not None or self.trainer_config is not None:
            # New split-config style
            direct_params = {}
            model_config_params = {}
            preprocessing_config_params = {}
            trainer_config_params = {}
            invalid_keys = []

            for k, v in parameters.items():
                if k.startswith("model_config__"):
                    model_config_params[k[len("model_config__") :]] = v
                elif k.startswith("preprocessing_config__"):
                    preprocessing_config_params[k[len("preprocessing_config__") :]] = v
                elif k.startswith("trainer_config__"):
                    trainer_config_params[k[len("trainer_config__") :]] = v
                elif k in (
                    "model_config",
                    "preprocessing_config",
                    "trainer_config",
                    "random_state",
                    "observability_config",
                ):
                    direct_params[k] = v
                else:
                    invalid_keys.append(k)

            if invalid_keys:
                raise ValueError(
                    f"Invalid parameter(s) {sorted(invalid_keys)} for estimator "
                    f"{type(self).__name__}. Check the list of available parameters "
                    "with `estimator.get_params().keys()`."
                )

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
                        self._preprocessor = build_preprocessor(
                            v,
                            task=type(self)._task,
                            random_state=self.random_state,
                            observability_config=getattr(self, "_observability_config", None),
                        )
                elif k == "trainer_config":
                    self.trainer_config = v
                    if v is not None and hasattr(v, "optimizer_type"):
                        self._optimizer_type = v.optimizer_type
                elif k == "random_state":
                    self.random_state = v
                elif k == "observability_config":
                    self._apply_observability_config(v)

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
                self._preprocessor = build_preprocessor(
                    self.preprocessing_config,
                    task=type(self)._task,
                    random_state=self.random_state,
                    observability_config=getattr(self, "_observability_config", None),
                )
            if trainer_config_params and self.trainer_config is not None and hasattr(self.trainer_config, "set_params"):
                self.trainer_config.set_params(**trainer_config_params)
                self._optimizer_type = self.trainer_config.optimizer_type

            return self

        # Legacy flat-kwargs style. Unlike the split-config style above, these
        # keys map directly onto `self.config`'s own fields (there is no
        # nested config object to delegate to), so they are validated against
        # its dataclass fields and applied onto it directly. Previously they
        # only ever touched `_config_kwargs`, a bookkeeping dict that fit()
        # never reads, so set_params() silently had no effect on the model
        # actually built (GH #410).
        direct_keys = {"model_config", "preprocessing_config", "trainer_config", "random_state"}
        valid_config_fields = {f.name for f in dataclass_fields(self.config)} if is_dataclass(self.config) else set()
        valid_keys = valid_config_fields | set(self._preprocessor_arg_names) | direct_keys | {"observability_config"}
        invalid_keys = set(parameters) - valid_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid parameter(s) {sorted(invalid_keys)} for estimator "
                f"{type(self).__name__}. Check the list of available parameters "
                "with `estimator.get_params().keys()`."
            )

        if "observability_config" in parameters:
            self._apply_observability_config(parameters["observability_config"])

        for k in direct_keys:
            if k in parameters:
                setattr(self, k, parameters[k])

        config_params = {k: v for k, v in parameters.items() if k in valid_config_fields}
        preprocessor_params = {k: v for k, v in parameters.items() if k in self._preprocessor_arg_names}

        if config_params:
            self._config_kwargs.update(config_params)
            for k, v in config_params.items():
                setattr(self.config, k, v)

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
