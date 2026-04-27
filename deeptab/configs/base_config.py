from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn


@dataclass
class BaseConfig:
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
