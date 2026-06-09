from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

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

# Valid choices for PreprocessingConfig fields (mirrors pretab.Preprocessor)
_VALID_NUMERICAL_PREPROCESSING: frozenset[str | None] = frozenset(
    {
        "ple",
        "quantile",
        "splines",
        "standardization",
        "minmax",
        "robust",
        "box-cox",
        "yeo-johnson",
        None,
    }
)
_VALID_SCALING_STRATEGY: frozenset[str | None] = frozenset({"minmax", "standardization", "robust", None})
_VALID_BINNING_STRATEGY: frozenset[str | None] = frozenset({"uniform", "quantile", "kmeans", None})
_VALID_CAT_ENCODING: frozenset[str] = frozenset({"int", "one-hot", "linear"})
_VALID_MONITOR_MODE: frozenset[str] = frozenset({"min", "max"})

__all__ = [
    "BaseConfig",
    "BaseModelConfig",
    "PreprocessingConfig",
    "SplitConfig",
    "TrainerConfig",
]


@dataclass
class BaseConfig(BaseEstimator):
    """
    Base configuration class with shared hyperparameters for models.

    This configuration class provides common hyperparameters for optimization,
    embeddings, and categorical encoding, which can be inherited by specific
    model configurations.

    Parameters
    ----------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement before reducing the learning rate.
    weight_decay : float, default=1e-06
        L2 regularization parameter for weight decay in the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate is reduced when patience is exceeded.
    activation : Callable, default=nn.ReLU()
        Activation function to use in the model's layers.
    cat_encoding : str, default="int"
        Method for encoding categorical features ('int', 'one-hot', or 'linear').

    Embedding Parameters
    --------------------
    use_embeddings : bool, default=False
        Whether to use embeddings for categorical or numerical features.
    embedding_activation : Callable, default=nn.Identity()
        Activation function applied to embeddings.
    embedding_type : str, default="linear"
        Type of embedding to use ('linear', 'plr', etc.).
    embedding_bias : bool, default=False
        Whether to use bias in embedding layers.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after embedding layers.
    d_model : int, default=32
        Dimensionality of embeddings or model representations.
    plr_lite : bool, default=False
        Whether to use a lightweight version of Piecewise Linear Regression (PLR).
    n_frequencies : int, default=48
        Number of frequency components for embeddings.
    frequencies_init_scale : float, default=0.01
        Initial scale for frequency components in embeddings.
    embedding_projection : bool, default=True
        Whether to apply a projection layer after embeddings.

    Notes
    -----
    - This base class is meant to be inherited by other configurations.
    - Provides default values that can be overridden in derived configurations.

    """

    # Training Parameters
    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1

    # Embedding Parameters
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

    # Architecture Parameters
    batch_norm: bool = False
    layer_norm: bool = False
    layer_norm_eps: float = 1e-05
    activation: Callable = nn.ReLU()  # noqa: RUF009
    cat_encoding: str = "int"


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

        for dropout_field in ("dropout", "attn_dropout", "ff_dropout", "head_dropout"):
            val = getattr(self, dropout_field, None)
            if val is not None and not (0.0 <= val < 1.0):
                raise invalid_param_error(
                    cls_name,
                    dropout_field,
                    val,
                    "must be in [0, 1)",
                )


@dataclass
class PreprocessingConfig(BaseEstimator):
    """Configuration for input feature preprocessing.

    All fields map directly to arguments accepted by ``pretab.preprocessor.Preprocessor``.
    Using ``None`` for any field leaves the preprocessor default in effect.

    Parameters
    ----------
    numerical_preprocessing : str or None, default=None
        Strategy for transforming numerical features (e.g. ``"ple"``, ``"quantile"``,
        ``"standard"``).  ``None`` uses the preprocessor's built-in default.
    categorical_preprocessing : str or None, default=None
        Strategy for transforming categorical features (e.g. ``"int"``, ``"one-hot"``).
        ``None`` uses the preprocessor's built-in default.
    n_bins : int or None, default=None
        Number of bins for numerical binning.  ``None`` uses the preprocessor default.
    feature_preprocessing : str or None, default=None
        General feature-level preprocessing override.
    use_decision_tree_bins : bool or None, default=None
        Whether to use decision-tree-derived bin edges.
    binning_strategy : str or None, default=None
        Strategy for choosing bin edges (e.g. ``"uniform"``, ``"quantile"``).
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
        Scaling method applied to numerical features (e.g. ``"standard"``,
        ``"minmax"``, ``"robust"``).
    n_knots : int or None, default=None
        Number of knots for spline preprocessing.
    use_decision_tree_knots : bool or None, default=None
        Whether to use decision-tree-derived knot positions.
    knots_strategy : str or None, default=None
        Strategy for knot placement.
    spline_implementation : str or None, default=None
        Backend used for spline transformations.
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

    def to_preprocessor_kwargs(self) -> dict:
        """Return a dict of non-None fields suitable for passing to ``Preprocessor(**...)``.

        Returns
        -------
        dict
            Mapping of field name → value for every field that is not ``None``.
        """
        return {k: v for k, v in self.get_params(deep=False).items() if v is not None}


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
        if self.patience >= self.max_epochs:
            warn_config(
                f"TrainerConfig: patience={self.patience} >= "
                f"max_epochs={self.max_epochs}. "
                "Early stopping will never trigger before training ends. "
                "Consider reducing patience or increasing max_epochs.",
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


@dataclass
class SplitConfig(BaseEstimator):
    """Configuration for train/validation data splitting.

    Controls how the training data is split into training and validation sets
    when no explicit validation set is provided.

    Parameters
    ----------
    val_size : float, default=0.2
        Fraction of the training data held out for validation when no explicit
        validation set is provided. Must be between 0 and 1.
    random_state : int, default=101
        Random seed for reproducibility in data splitting. Controls the
        shuffling applied before the split.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting. If False, the split
        is deterministic based on order.
    stratify : bool, default=False
        Whether to preserve class proportions in classification splits.
        Only applies to classification tasks.
    """

    val_size: float = 0.2
    random_state: int = 101
    shuffle: bool = True
    stratify: bool = False
