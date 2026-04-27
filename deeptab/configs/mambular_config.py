from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn

from .base_config import BaseConfig


@dataclass
class DefaultMambularConfig(BaseConfig):
    """Configuration class for the Default Mambular model with predefined hyperparameters.

    Parameters
    ----------
    d_model : int, default=64
        Dimensionality of the model.
    n_layers : int, default=4
        Number of layers in the model.
    expand_factor : int, default=2
        Expansion factor for the feed-forward layers.
    bias : bool, default=False
        Whether to use bias in the linear layers.
    dropout : float, default=0.0
        Dropout rate for regularization.
    d_conv : int, default=4
        Size of convolution over columns.
    dilation : int, default=1
        Dilation factor for the convolution.
    dt_rank : str, default="auto"
        Rank of the decision tree used in the model.
    d_state : int, default=128
        Dimensionality of the state in recurrent layers.
    dt_scale : float, default=1.0
        Scaling factor for decision tree parameters.
    dt_init : str, default="random"
        Initialization method for decision tree parameters.
    dt_max : float, default=0.1
        Maximum value for decision tree initialization.
    dt_min : float, default=1e-04
        Minimum value for decision tree initialization.
    dt_init_floor : float, default=1e-04
        Floor value for decision tree initialization.
    norm : str, default="RMSNorm"
        Type of normalization used ('RMSNorm', etc.).
    activation : callable, default=nn.SiLU()
        Activation function for the model.
    shuffle_embeddings : bool, default=False
        Whether to shuffle embeddings before being passed to Mamba layers.
    head_layer_sizes : list, default=()
        Sizes of the layers in the model's head.
    head_dropout : float, default=0.5
        Dropout rate for the head layers.
    head_skip_layers : bool, default=False
        Whether to skip layers in the head.
    head_activation : callable, default=nn.SELU()
        Activation function for the head layers.
    head_use_batch_norm : bool, default=False
        Whether to use batch normalization in the head layers.
    pooling_method : str, default="avg"
        Pooling method to use ('avg', 'max', etc.).
    bidirectional : bool, default=False
        Whether to process data bidirectionally.
    use_learnable_interaction : bool, default=False
        Whether to use learnable feature interactions before passing through Mamba blocks.
    use_cls : bool, default=False
        Whether to append a CLS token to the input sequences.
    use_pscan : bool, default=False
        Whether to use PSCAN for the state-space model.
    mamba_version : str, default="mamba-torch"
        Version of the Mamba model to use ('mamba-torch', 'mamba1', 'mamba2').
    conv_bias : bool, default=False
        Whether to use a bias in the 1D convolution before each mamba block
    AD_weight_decay: bool = True
        Whether to use weight decay als for the A and D matrices in Mamba
    BC_layer_norm: bool = False
        Whether to use layer norm on the B and C matrices
    """

    # Architecture Parameters
    d_model: int = 64
    n_layers: int = 4
    d_conv: int = 4
    dilation: int = 1
    expand_factor: int = 2
    bias: bool = False
    dropout: float = 0.0
    dt_rank: str = "auto"
    d_state: int = 128
    dt_scale: float = 1.0
    dt_init: str = "random"
    dt_max: float = 0.1
    dt_min: float = 1e-04
    dt_init_floor: float = 1e-04
    norm: str = "RMSNorm"
    activation: Callable = nn.SiLU()  # noqa: RUF009
    conv_bias: bool = False
    AD_weight_decay: bool = True
    BC_layer_norm: bool = False

    # Embedding Parameters
    shuffle_embeddings: bool = False

    # Head Parameters
    head_layer_sizes: list = field(default_factory=list)
    head_dropout: float = 0.5
    head_skip_layers: bool = False
    head_activation: Callable = nn.SELU()  # noqa: RUF009
    head_use_batch_norm: bool = False

    # Additional Features
    pooling_method: str = "avg"
    bidirectional: bool = False
    use_learnable_interaction: bool = False
    use_cls: bool = False
    use_pscan: bool = False

    # Mamba Version
    mamba_version: str = "mamba-torch"
