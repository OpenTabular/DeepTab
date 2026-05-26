from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn
from sklearn.base import BaseEstimator


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
    checkpoint_path: str = "model_checkpoints"


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
