from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn
from sklearn.base import BaseEstimator

from deeptab.core.exceptions import (
    ConfigWarning,
    IncompatibleParamsError,
    InvalidParamError,
    incompatible_params_error,
    invalid_param_error,
    warn_config,
)

# Valid choices for PreprocessingConfig fields (mirrors pretab.Preprocessor).
# Confirmed against Preprocessor(numerical_method=...) directly (not just the
# broader representation registry, which also lists standalone-only
# transformers PreTab does not accept here); see dev/debug/pretab_numerical_methods.py.
_VALID_NUMERICAL_PREPROCESSING: frozenset[str | None] = frozenset(
    {
        "box-cox",
        "bspline",
        "cubicspline",
        "custombin",
        "fourier",
        "ispline",
        "minmax",
        "mspline",
        "naturalspline",
        "none",
        "ple",
        "polynomial",
        "pspline",
        "quantile",
        "rbf",
        "relu",
        "robust",
        "sigmoid",
        "standardization",
        "tanh",
        "yeo-johnson",
        None,
    }
)
# Numerical methods that always require y (never fit without a target), vs.
# methods that never consume y, per pretab.list_representations(supervised=...).
_NUMERICAL_METHODS_ALWAYS_TARGET_AWARE: frozenset[str] = frozenset({"ple"})
_NUMERICAL_METHODS_NEVER_TARGET_AWARE: frozenset[str] = frozenset(
    {
        "box-cox",
        "custombin",
        "fourier",
        "minmax",
        "none",
        "polynomial",
        "pspline",
        "quantile",
        "robust",
        "standardization",
        "yeo-johnson",
    }
)
_VALID_SCALING_STRATEGY: frozenset[str | None] = frozenset({"minmax", "standardization", "robust", None})
_VALID_BINNING_STRATEGY: frozenset[str | None] = frozenset({"uniform", "quantile", "kmeans", None})
_VALID_CATEGORICAL_METHOD: frozenset[str | None] = frozenset({"int", "one-hot", "pretrained", "none", None})
# placement_strategy validity is gated by target_aware, not by numerical_method:
# target_aware=False requires "uniform"/"quantile"; target_aware=True requires
# "cart" or "lightgbm". "lightgbm" is deferred for now (needs pretab[lightgbm]).
_VALID_PLACEMENT_STRATEGY: frozenset[str | None] = frozenset({"uniform", "quantile", "cart", None})
_PLACEMENT_STRATEGIES_ALWAYS_TARGET_AWARE: frozenset[str] = frozenset({"cart"})
_PLACEMENT_STRATEGIES_NEVER_TARGET_AWARE: frozenset[str] = frozenset({"uniform", "quantile"})
_VALID_CAT_ENCODING: frozenset[str] = frozenset({"int", "one-hot", "linear"})
_VALID_MONITOR_MODE: frozenset[str] = frozenset({"min", "max"})

__all__ = [
    "BaseModelConfig",
    "PreprocessingConfig",
    "TrainerConfig",
]


@dataclass
class BaseModelConfig(BaseEstimator):
    """Shared architecture hyperparameters for all DeepTab models.

    This class contains only architectural / structural configuration.
    Training-related parameters (``lr``, ``weight_decay``, ``max_epochs``, …)
    belong in :class:`~deeptab.configs.trainer_config.TrainerConfig`.
    Preprocessing parameters belong in
    :class:`~deeptab.configs.preprocessing_config.PreprocessingConfig`.

    Parameters
    ----------
    use_embeddings : bool, default=False
        Whether to use embedding layers for numerical/categorical features.
    embedding_activation : Callable, default=nn.Identity()
        Activation function applied to embeddings.
    embedding_type : str, default="linear"
        Type of embedding (``"linear"``, ``"plr"``, etc.).
    embedding_bias : bool, default=False
        Whether to add a bias term to embedding layers.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalisation after the embedding layer.
    d_model : int, default=32
        Embedding / model dimensionality.
    plr_lite : bool, default=False
        Whether to use the lightweight PLR embedding variant.
    n_frequencies : int, default=48
        Number of frequency components for PLR embeddings.
    frequencies_init_scale : float, default=0.01
        Initial scale for PLR frequency components.
    embedding_projection : bool, default=True
        Whether to apply a linear projection after embeddings.
    batch_norm : bool, default=False
        Whether to use batch normalisation in the model body.
    layer_norm : bool, default=False
        Whether to use layer normalisation in the model body.
    layer_norm_eps : float, default=1e-5
        Epsilon for layer normalisation numerical stability.
    activation : Callable, default=nn.ReLU()
        Activation function used throughout the model body.
    cat_encoding : str, default="int"
        How categorical features are encoded at the model input
        (``"int"``, ``"one-hot"``, ``"linear"``).
    """

    # Embedding parameters
    use_embeddings: bool = False
    embedding_activation: Callable = nn.Identity()  # noqa: RUF009
    embedding_type: str = "linear"
    embedding_bias: bool = False
    layer_norm_after_embedding: bool = False
    d_model: int = 32
    plr_lite: bool = False
    n_frequencies: int = 48
    frequencies_init_scale: float = 0.01
    embedding_projection: bool = True

    # Architecture parameters
    batch_norm: bool = False
    layer_norm: bool = False
    layer_norm_eps: float = 1e-05
    activation: Callable = nn.ReLU()  # noqa: RUF009
    cat_encoding: str = "int"

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.d_model < 1:
            raise invalid_param_error(type(self).__name__, "d_model", self.d_model, "must be >= 1")
        if self.cat_encoding not in _VALID_CAT_ENCODING:
            raise invalid_param_error(
                type(self).__name__,
                "cat_encoding",
                self.cat_encoding,
                "must be one of the known encoding strategies",
                sorted(_VALID_CAT_ENCODING),
            )
        # --- Common optional fields present on many model configs ---
        cls_name = type(self).__name__
        n_layers = getattr(self, "n_layers", None)
        if n_layers is not None and n_layers < 1:
            raise invalid_param_error(cls_name, "n_layers", n_layers, "must be >= 1")

        n_heads = getattr(self, "n_heads", None)
        if n_heads is not None:
            if n_heads < 1:
                raise invalid_param_error(cls_name, "n_heads", n_heads, "must be >= 1")
            if self.d_model % n_heads != 0:
                raise incompatible_params_error(
                    cls_name,
                    f"d_model ({self.d_model}) must be divisible by n_heads ({n_heads}).",
                )

        for dropout_field in ("dropout", "attn_dropout", "ff_dropout", "head_dropout", "rnn_dropout"):
            val = getattr(self, dropout_field, None)
            if val is not None and not (0.0 <= val < 1.0):
                raise invalid_param_error(
                    cls_name,
                    dropout_field,
                    val,
                    "must be in [0, 1)",
                )

        # --- Embedding / frequency fields on BaseModelConfig itself ---
        if self.n_frequencies < 1:
            raise invalid_param_error(cls_name, "n_frequencies", self.n_frequencies, "must be >= 1")
        if self.frequencies_init_scale <= 0:
            raise invalid_param_error(cls_name, "frequencies_init_scale", self.frequencies_init_scale, "must be > 0")
        if self.layer_norm_eps <= 0:
            raise invalid_param_error(cls_name, "layer_norm_eps", self.layer_norm_eps, "must be > 0")

        # --- Cross-field: conflicting normalisation ---
        if self.batch_norm and self.layer_norm:
            warn_config(
                f"{cls_name}: both batch_norm=True and layer_norm=True are set. "
                "Using both simultaneously is unusual and may produce unexpected results. "
                "Consider enabling only one.",
                stacklevel=3,
            )

        # --- Mamba / RNN / Transformer optional integer fields ---
        for int_field in ("expand_factor", "d_conv", "d_state", "dim_feedforward", "transformer_dim_feedforward"):
            val = getattr(self, int_field, None)
            if val is not None and val < 1:
                raise invalid_param_error(cls_name, int_field, val, "must be >= 1")


def _resolve_legacy_alias(
    legacy_name: str,
    legacy_value: Any,
    canonical_name: str,
    canonical_value: Any,
) -> Any:
    """Resolve a deprecated field/canonical field pair to a single value.

    Returns *canonical_value* unchanged when the legacy field was not set.
    When the legacy field was set, it is honored (with a deprecation
    warning) as long as it does not conflict with an explicitly set
    canonical value; a conflict raises ``IncompatibleParamsError``.
    """
    if legacy_value is None:
        return canonical_value
    if canonical_value is not None and canonical_value != legacy_value:
        raise incompatible_params_error(
            "PreprocessingConfig",
            f"'{legacy_name}'={legacy_value!r} and '{canonical_name}'={canonical_value!r} disagree. "
            f"Set only '{canonical_name}'.",
        )
    warn_config(
        f"PreprocessingConfig.{legacy_name} is deprecated; use '{canonical_name}' instead. "
        f"'{legacy_name}={legacy_value!r}' is equivalent to '{canonical_name}={legacy_value!r}'.",
        stacklevel=4,
    )
    return legacy_value


@dataclass
class PreprocessingConfig(BaseEstimator):
    """Configuration for input feature preprocessing.

    Canonical fields map directly to arguments accepted by ``pretab.Preprocessor``.
    Legacy fields are deprecated aliases kept for backward compatibility; using one
    emits a ``DeprecationWarning`` and is resolved to its canonical equivalent.
    Using ``None`` for any field leaves the preprocessor default in effect.

    Parameters
    ----------
    numerical_preprocessing : str or None, default=None
        Deprecated alias for ``numerical_method``; setting this emits a
        ``DeprecationWarning`` and behaves exactly as if ``numerical_method``
        were set to the same value.
    categorical_preprocessing : str or None, default=None
        Deprecated alias for ``categorical_method``; setting this emits a
        ``DeprecationWarning`` and behaves exactly as if ``categorical_method``
        were set to the same value.
    n_bins : int or None, default=None
        Deprecated alias for ``output_dim`` (numerical binning width); setting
        this emits a ``DeprecationWarning`` and behaves exactly as if
        ``output_dim`` were set to the same value.
    feature_preprocessing : str or None, default=None
        General feature-level preprocessing override.
    use_decision_tree_bins : bool or None, default=None
        Deprecated alias for ``target_aware``, honored only when it does not
        conflict with ``use_decision_tree_knots``; setting it emits a
        ``DeprecationWarning``.
    binning_strategy : str or None, default=None
        Deprecated alias for ``placement_strategy``, honored only when it does
        not conflict with ``knots_strategy``; setting it emits a
        ``DeprecationWarning``.
    task : str or None, default=None
        Task type passed to the preprocessor for task-aware transformations
        (e.g. ``"regression"``, ``"classification"``).
    cat_cutoff : float or None, default=None
        Threshold for treating integer columns as categorical.
    treat_all_integers_as_numerical : bool or None, default=None
        When ``True``, integer columns are never converted to categorical.
    degree : int or None, default=None
        Polynomial / spline degree for numerical feature expansion.
    scaling_strategy : str or None, default=None
        Deprecated alias for ``scaling``; setting this emits a
        ``DeprecationWarning`` and behaves exactly as if ``scaling`` were set
        to the same value.
    n_knots : int or None, default=None
        Deprecated alias for ``output_dim`` (spline knot count); setting this
        emits a ``DeprecationWarning`` and behaves exactly as if ``output_dim``
        were set to the same value.
    use_decision_tree_knots : bool or None, default=None
        Deprecated alias for ``target_aware``, honored only when it does not
        conflict with ``use_decision_tree_bins``; setting it emits a
        ``DeprecationWarning``.
    knots_strategy : str or None, default=None
        Deprecated alias for ``placement_strategy``, honored only when it does
        not conflict with ``binning_strategy``; setting it emits a
        ``DeprecationWarning``.
    spline_implementation : str or None, default=None
        Removed in PreTab 1.0. Setting this to any value raises
        ``IncompatibleParamsError``; PreTab now selects its spline backend
        automatically.
    numerical_method : str or None, default=None
        Strategy for transforming numerical features (e.g. ``"ple"``,
        ``"bspline"``, ``"quantile"``). ``None`` uses the preprocessor's
        built-in default. Canonical replacement for ``numerical_preprocessing``.
    categorical_method : str or None, default=None
        Strategy for transforming categorical features (``"int"``,
        ``"one-hot"``, or ``"pretrained"``). ``None`` uses the preprocessor's
        built-in default. Canonical replacement for ``categorical_preprocessing``.
    output_dim : int or None, default=None
        Output width for numerical representations (bins, knots, or expansion
        dimensions). ``None`` uses the preprocessor's built-in default.
        Canonical replacement for ``n_bins`` and ``n_knots``.
    target_aware : bool or None, default=None
        Whether numerical placement uses the target. ``False`` requires
        ``placement_strategy`` in ``{"uniform", "quantile"}``; ``True`` requires
        ``placement_strategy="cart"`` (``"lightgbm"`` is not yet supported).
        Canonical replacement for ``use_decision_tree_bins`` and
        ``use_decision_tree_knots``.
    placement_strategy : str or None, default=None
        Strategy for placing bin edges or knots. Its valid values depend on
        ``target_aware``, not on ``numerical_method``: ``"uniform"`` or
        ``"quantile"`` when ``target_aware=False``; ``"cart"`` when
        ``target_aware=True`` (PreTab's ``"lightgbm"`` option is not yet
        supported, since it requires the optional ``pretab[lightgbm]``
        dependency). Canonical replacement for ``binning_strategy`` and
        ``knots_strategy``.
    scaling : str or None, default=None
        Scaling method applied to numerical features (e.g. ``"standardization"``,
        ``"minmax"``, ``"robust"``). Canonical replacement for
        ``scaling_strategy``.
    """

    numerical_preprocessing: str | None = None
    categorical_preprocessing: str | None = None
    n_bins: int | None = None
    feature_preprocessing: str | None = None
    use_decision_tree_bins: bool | None = None
    binning_strategy: str | None = None
    task: str | None = None
    cat_cutoff: float | None = None
    treat_all_integers_as_numerical: bool | None = None
    degree: int | None = None
    scaling_strategy: str | None = None
    n_knots: int | None = None
    use_decision_tree_knots: bool | None = None
    knots_strategy: str | None = None
    spline_implementation: str | None = None
    numerical_method: str | None = None
    categorical_method: str | None = None
    output_dim: int | None = None
    target_aware: bool | None = None
    placement_strategy: str | None = None
    scaling: str | None = None

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.numerical_preprocessing not in _VALID_NUMERICAL_PREPROCESSING:
            raise invalid_param_error(
                "PreprocessingConfig",
                "numerical_preprocessing",
                self.numerical_preprocessing,
                "must be one of the known preprocessing methods",
                sorted(x for x in _VALID_NUMERICAL_PREPROCESSING if x is not None),
            )
        if self.n_bins is not None and self.n_bins < 2:
            raise invalid_param_error("PreprocessingConfig", "n_bins", self.n_bins, "must be >= 2")
        if self.n_knots is not None and self.n_knots < 2:
            raise invalid_param_error("PreprocessingConfig", "n_knots", self.n_knots, "must be >= 2")
        if self.scaling_strategy not in _VALID_SCALING_STRATEGY:
            raise invalid_param_error(
                "PreprocessingConfig",
                "scaling_strategy",
                self.scaling_strategy,
                "must be one of the known scaling strategies",
                sorted(x for x in _VALID_SCALING_STRATEGY if x is not None),
            )
        if self.binning_strategy not in _VALID_BINNING_STRATEGY:
            raise invalid_param_error(
                "PreprocessingConfig",
                "binning_strategy",
                self.binning_strategy,
                "must be one of the known binning strategies",
                sorted(x for x in _VALID_BINNING_STRATEGY if x is not None),
            )
        if self.cat_cutoff is not None and not (0.0 < self.cat_cutoff < 1.0):
            raise invalid_param_error(
                "PreprocessingConfig",
                "cat_cutoff",
                self.cat_cutoff,
                "must be in the open interval (0, 1)",
            )
        if self.degree is not None and self.degree < 1:
            raise invalid_param_error("PreprocessingConfig", "degree", self.degree, "must be >= 1")
        if self.categorical_method not in _VALID_CATEGORICAL_METHOD:
            raise invalid_param_error(
                "PreprocessingConfig",
                "categorical_method",
                self.categorical_method,
                "must be one of the known categorical methods",
                sorted(x for x in _VALID_CATEGORICAL_METHOD if x is not None),
            )
        if self.output_dim is not None and self.output_dim < 1:
            raise invalid_param_error("PreprocessingConfig", "output_dim", self.output_dim, "must be >= 1")
        if self.placement_strategy not in _VALID_PLACEMENT_STRATEGY:
            raise invalid_param_error(
                "PreprocessingConfig",
                "placement_strategy",
                self.placement_strategy,
                "must be one of the known placement strategies",
                sorted(x for x in _VALID_PLACEMENT_STRATEGY if x is not None),
            )

        # PreTab 1.0 removed this option outright; there is no equivalent to
        # fall back to, so a non-default value is always rejected.
        if self.spline_implementation is not None:
            raise incompatible_params_error(
                "PreprocessingConfig",
                "'spline_implementation' has no equivalent in PreTab 1.0 and is not "
                "accepted. Remove it; PreTab now selects its spline backend automatically.",
            )

        # Resolve each legacy/canonical field pair once, so `to_preprocessor_kwargs`
        # only ever emits canonical PreTab 1.0 names.
        self._resolved_numerical_method = _resolve_legacy_alias(
            "numerical_preprocessing", self.numerical_preprocessing, "numerical_method", self.numerical_method
        )
        self._resolved_categorical_method = _resolve_legacy_alias(
            "categorical_preprocessing", self.categorical_preprocessing, "categorical_method", self.categorical_method
        )
        self._resolved_scaling = _resolve_legacy_alias(
            "scaling_strategy", self.scaling_strategy, "scaling", self.scaling
        )
        self._resolved_output_dim = self._resolve_output_dim()
        self._resolved_target_aware = self._resolve_target_aware()
        self._resolved_placement_strategy = self._resolve_placement_strategy()

        if (
            self._resolved_numerical_method in _NUMERICAL_METHODS_NEVER_TARGET_AWARE
            and self._resolved_target_aware is True
        ):
            raise incompatible_params_error(
                "PreprocessingConfig",
                f"numerical_method={self._resolved_numerical_method!r} does not support target-aware "
                "placement, but target_aware=True was requested. Remove target_aware or choose a "
                "target-aware-capable method.",
            )
        if (
            self._resolved_numerical_method in _NUMERICAL_METHODS_ALWAYS_TARGET_AWARE
            and self._resolved_target_aware is False
        ):
            raise incompatible_params_error(
                "PreprocessingConfig",
                f"numerical_method={self._resolved_numerical_method!r} always requires the target to fit "
                "(it cannot run unsupervised), but target_aware=False was requested. Remove target_aware "
                "or choose a different method.",
            )
        if (
            self._resolved_placement_strategy in _PLACEMENT_STRATEGIES_NEVER_TARGET_AWARE
            and self._resolved_target_aware is True
        ):
            raise incompatible_params_error(
                "PreprocessingConfig",
                f"placement_strategy={self._resolved_placement_strategy!r} requires target_aware=False, "
                "but target_aware=True was requested. Use 'cart' for target-aware placement instead.",
            )
        if (
            self._resolved_placement_strategy in _PLACEMENT_STRATEGIES_ALWAYS_TARGET_AWARE
            and self._resolved_target_aware is False
        ):
            raise incompatible_params_error(
                "PreprocessingConfig",
                f"placement_strategy={self._resolved_placement_strategy!r} requires target_aware=True, "
                "but target_aware=False was requested. Use 'uniform' or 'quantile' for unsupervised "
                "placement instead.",
            )

    def _resolve_output_dim(self) -> int | None:
        """Resolve `n_bins`/`n_knots` (Category A) into `output_dim`."""
        legacy_values = {v for v in (self.n_bins, self.n_knots) if v is not None}
        if not legacy_values:
            return self.output_dim
        if len(legacy_values) > 1:
            raise incompatible_params_error(
                "PreprocessingConfig",
                f"'n_bins'={self.n_bins!r} and 'n_knots'={self.n_knots!r} disagree. Set only 'output_dim'.",
            )
        (legacy_value,) = legacy_values
        legacy_name = "n_bins" if self.n_bins is not None else "n_knots"
        return _resolve_legacy_alias(legacy_name, legacy_value, "output_dim", self.output_dim)

    def _resolve_target_aware(self) -> bool | None:
        """Resolve the decision-tree flags (Category B) into `target_aware`."""
        legacy_values = {v for v in (self.use_decision_tree_bins, self.use_decision_tree_knots) if v is not None}
        if not legacy_values:
            return self.target_aware
        if len(legacy_values) > 1:
            raise incompatible_params_error(
                "PreprocessingConfig",
                f"'use_decision_tree_bins'={self.use_decision_tree_bins!r} and "
                f"'use_decision_tree_knots'={self.use_decision_tree_knots!r} disagree; there is no "
                "unambiguous 'target_aware' equivalent. Set only 'target_aware' instead.",
            )
        (legacy_value,) = legacy_values
        legacy_name = "use_decision_tree_bins" if self.use_decision_tree_bins is not None else "use_decision_tree_knots"
        return _resolve_legacy_alias(legacy_name, legacy_value, "target_aware", self.target_aware)

    def _resolve_placement_strategy(self) -> str | None:
        """Resolve `binning_strategy`/`knots_strategy` (Category B) into `placement_strategy`."""
        legacy_values = {v for v in (self.binning_strategy, self.knots_strategy) if v is not None}
        if not legacy_values:
            return self.placement_strategy
        if len(legacy_values) > 1:
            raise incompatible_params_error(
                "PreprocessingConfig",
                f"'binning_strategy'={self.binning_strategy!r} and 'knots_strategy'={self.knots_strategy!r} "
                "disagree. Set only 'placement_strategy'.",
            )
        (legacy_value,) = legacy_values
        legacy_name = "binning_strategy" if self.binning_strategy is not None else "knots_strategy"
        if legacy_value not in _VALID_PLACEMENT_STRATEGY:
            raise incompatible_params_error(
                "PreprocessingConfig",
                f"'{legacy_name}'={legacy_value!r} has no supported 'placement_strategy' equivalent. "
                f"Use 'placement_strategy' with one of "
                f"{sorted(x for x in _VALID_PLACEMENT_STRATEGY if x is not None)}.",
            )
        return _resolve_legacy_alias(legacy_name, legacy_value, "placement_strategy", self.placement_strategy)

    def to_preprocessor_kwargs(self) -> dict:
        """Return canonical kwargs for ``pretab.Preprocessor(**...)``.

        Legacy field names are resolved to their canonical PreTab 1.0
        equivalents in ``__post_init__``; only canonical names are ever
        returned here, since PreTab 1.0 no longer accepts the legacy ones.

        Returns
        -------
        dict
            Mapping of canonical PreTab field name → value for every resolved
            field that is not ``None``.
        """
        kwargs = {
            "numerical_method": self._resolved_numerical_method,
            "categorical_method": self._resolved_categorical_method,
            "output_dim": self._resolved_output_dim,
            "target_aware": self._resolved_target_aware,
            "placement_strategy": self._resolved_placement_strategy,
            "scaling": self._resolved_scaling,
            "feature_preprocessing": self.feature_preprocessing,
            "task": self.task,
            "cat_cutoff": self.cat_cutoff,
            "treat_all_integers_as_numerical": self.treat_all_integers_as_numerical,
            "degree": self.degree,
        }
        return {k: v for k, v in kwargs.items() if v is not None}


@dataclass
class TrainerConfig(BaseEstimator):
    """Configuration for training loop, optimizer, and runtime execution.

    These settings are entirely separate from model architecture.  They control
    *how* a model is trained and executed, not *what* the model is.

    Parameters
    ----------
    max_epochs : int, default=100
        Maximum number of training epochs.
    batch_size : int, default=128
        Number of samples per gradient update.
    val_size : float, default=0.2
        Fraction of the training data held out for validation when no explicit
        validation set is provided.
    shuffle : bool, default=True
        Whether to shuffle training data before each epoch.
    stratify : bool, default=True
        Whether to stratify the validation split on ``y`` for classification
        tasks, so the train and validation sets keep the same class
        proportions. Has no effect on regression, where a continuous target
        cannot be stratified. Set to ``False`` to draw a purely random split.
    patience : int, default=15
        Number of epochs with no improvement on ``monitor`` before early stopping
        is triggered.
    monitor : str, default="val_loss"
        Metric name to monitor for early stopping and checkpoint selection.
    mode : str, default="min"
        Whether the monitored metric should be minimised (``"min"``) or
        maximised (``"max"``).
    lr : float, default=1e-4
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement before the learning rate is reduced
        by ``lr_factor``.
    lr_factor : float, default=0.1
        Multiplicative factor applied to the learning rate when patience is
        exceeded.
    weight_decay : float, default=1e-6
        L2 regularisation coefficient (weight decay) for the optimizer.
    optimizer_type : str, default="Adam"
        Optimizer class name.  Must be a valid ``torch.optim`` class name or a
        name registered in the project's optimizer registry.
    optimizer_kwargs : dict or None, default=None
        Extra keyword arguments forwarded to the optimizer constructor.
    scheduler_type : str or None, default="ReduceLROnPlateau"
        LR-scheduler class name (case-insensitive), or ``None`` / ``"none"`` to
        disable the scheduler entirely.
    scheduler_kwargs : dict or None, default=None
        Extra keyword arguments forwarded to the scheduler constructor.
        ``factor`` and ``patience`` are synthesised from ``lr_factor`` and
        ``lr_patience`` for ``ReduceLROnPlateau`` when absent here.
    scheduler_monitor : str or None, default=None
        Metric name for the scheduler to monitor.  Falls back to the value of
        ``monitor`` when ``None``.
    scheduler_interval : str, default="epoch"
        Lightning scheduling granularity: ``"epoch"`` or ``"step"``.
    scheduler_frequency : int, default=1
        How often the scheduler steps at the given interval.
    no_weight_decay_for_bias_and_norm : bool, default=False
        When ``True``, bias vectors and normalisation-layer scale/shift
        parameters receive zero weight decay.  Recommended for transformer-
        style models with ``LayerNorm``.
    checkpoint_path : str, default="model_checkpoints"
        Directory where PyTorch Lightning model checkpoints are saved.
    """

    max_epochs: int = 100
    batch_size: int = 128
    val_size: float = 0.2
    shuffle: bool = True
    stratify: bool = True
    patience: int = 15
    monitor: str = "val_loss"
    mode: str = "min"
    lr: float = 1e-4
    lr_patience: int = 10
    lr_factor: float = 0.1
    weight_decay: float = 1e-6
    optimizer_type: str = "Adam"
    optimizer_kwargs: dict | None = None
    scheduler_type: str | None = "ReduceLROnPlateau"
    scheduler_kwargs: dict | None = None
    scheduler_monitor: str | None = None
    scheduler_interval: str = "epoch"
    scheduler_frequency: int = 1
    no_weight_decay_for_bias_and_norm: bool = False
    checkpoint_path: str = "model_checkpoints"

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.max_epochs < 1:
            raise invalid_param_error("TrainerConfig", "max_epochs", self.max_epochs, "must be >= 1")
        if self.batch_size < 1:
            raise invalid_param_error("TrainerConfig", "batch_size", self.batch_size, "must be >= 1")
        if self.lr <= 0:
            raise invalid_param_error("TrainerConfig", "lr", self.lr, "must be > 0")
        if self.weight_decay < 0:
            raise invalid_param_error("TrainerConfig", "weight_decay", self.weight_decay, "must be >= 0")
        if not (0.0 < self.val_size < 1.0):
            raise invalid_param_error(
                "TrainerConfig",
                "val_size",
                self.val_size,
                "must be in the open interval (0, 1)",
            )
        if self.mode not in _VALID_MONITOR_MODE:
            raise invalid_param_error(
                "TrainerConfig",
                "mode",
                self.mode,
                "must be 'min' or 'max'",
                ["min", "max"],
            )
        if self.lr_patience < 1:
            raise invalid_param_error("TrainerConfig", "lr_patience", self.lr_patience, "must be >= 1")
        if not (0.0 < self.lr_factor < 1.0):
            raise invalid_param_error(
                "TrainerConfig",
                "lr_factor",
                self.lr_factor,
                "must be in the open interval (0, 1)",
            )
        if self.patience >= self.max_epochs:
            warn_config(
                f"TrainerConfig: patience={self.patience} >= "
                f"max_epochs={self.max_epochs}. "
                "Early stopping will never trigger before training ends. "
                "Consider reducing patience or increasing max_epochs.",
                stacklevel=3,
            )
        if self.lr_patience >= self.max_epochs:
            warn_config(
                f"TrainerConfig: lr_patience={self.lr_patience} >= "
                f"max_epochs={self.max_epochs}. "
                "The learning rate scheduler will never reduce the LR before training ends. "
                "Consider reducing lr_patience or increasing max_epochs.",
                stacklevel=3,
            )
        if self.scheduler_interval not in {"epoch", "step"}:
            raise invalid_param_error(
                "TrainerConfig",
                "scheduler_interval",
                self.scheduler_interval,
                "must be 'epoch' or 'step'",
                ["epoch", "step"],
            )
        if self.scheduler_frequency < 1:
            raise invalid_param_error(
                "TrainerConfig",
                "scheduler_frequency",
                self.scheduler_frequency,
                "must be >= 1",
            )
