from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn

from .base_config import BaseConfig


@dataclass
class DefaultMambAttentionConfig(BaseConfig):
    """Configuration class for the Default Mambular Attention model with predefined hyperparameters.

    Parameters
    ----------
    d_model : int, default=64
        Dimensionality of the model.
    n_layers : int, default=4
        Number of layers in the model.
    expand_factor : int, default=2
        Expansion factor for the feed-forward layers.
    n_heads : int, default=8
        Number of attention heads in the model.
    last_layer : str, default="attn"
        Type of the last layer (e.g., 'attn').
    n_mamba_per_attention : int, default=1
        Number of Mamba blocks per attention layer.
    bias : bool, default=False
        Whether to use bias in the linear layers.
    d_conv : int, default=4
        Dimensionality of the convolutional layers.
    conv_bias : bool, default=True
        Whether to use bias in the convolutional layers.
    dropout : float, default=0.0
        Dropout rate for regularization.
    attn_dropout : float, default=0.2
        Dropout rate for the attention mechanism.
    dt_rank : str, default="auto"
        Rank of the decision tree.
    d_state : int, default=128
        Dimensionality of the state in recurrent layers.
    dt_scale : float, default=1.0
        Scaling factor for the decision tree.
    dt_init : str, default="random"
        Initialization method for the decision tree.
    dt_max : float, default=0.1
        Maximum value for decision tree initialization.
    dt_min : float, default=1e-04
        Minimum value for decision tree initialization.
    dt_init_floor : float, default=1e-04
        Floor value for decision tree initialization.
    norm : str, default="LayerNorm"
        Type of normalization used in the model.
    activation : callable, default=nn.SiLU()
        Activation function for the model.
    head_layer_sizes : list, default=()
        Sizes of the fully connected layers in the model's head.
    head_dropout : float, default=0.5
        Dropout rate for the head layers.
    head_skip_layers : bool, default=False
        Whether to use skip connections in the head layers.
    head_activation : callable, default=nn.SELU()
        Activation function for the head layers.
    head_use_batch_norm : bool, default=False
        Whether to use batch normalization in the head layers.
    pooling_method : str, default="avg"
        Pooling method to be used ('avg', 'max', etc.).
    bidirectional : bool, default=False
        Whether to process input sequences bidirectionally.
    use_learnable_interaction : bool, default=False
        Whether to use learnable feature interactions before passing through Mamba blocks.
    use_cls : bool, default=False
        Whether to append a CLS token for sequence pooling.
    shuffle_embeddings : bool, default=False
        Whether to shuffle embeddings before passing to Mamba layers.
    cat_encoding : str, default="int"
        Encoding method for categorical features ('int', 'one-hot', etc.).
    AD_weight_decay : bool, default=True
        Whether weight decay is applied to A-D matrices.
    BC_layer_norm : bool, default=False
        Whether to apply layer normalization to B-C matrices.
    use_pscan : bool, default=False
        Whether to use PSCAN for the state-space model.
    n_attention_layers : int, default=1
        Number of attention layers in the model.
    """

    # Architecture Parameters
    d_model: int = 64
    n_layers: int = 4
    expand_factor: int = 2
    n_heads: int = 8
    last_layer: str = "attn"
    n_mamba_per_attention: int = 1
    bias: bool = False
    d_conv: int = 4
    conv_bias: bool = True
    dropout: float = 0.0
    attn_dropout: float = 0.2
    dt_rank: str = "auto"
    d_state: int = 128
    dt_scale: float = 1.0
    dt_init: str = "random"
    dt_max: float = 0.1
    dt_min: float = 1e-04
    dt_init_floor: float = 1e-04
    norm: str = "LayerNorm"
    activation: Callable = nn.SiLU()  # noqa: RUF009

    # Head Parameters
    head_layer_sizes: list = field(default_factory=list)
    head_dropout: float = 0.5
    head_skip_layers: bool = False
    head_activation: Callable = nn.SELU()  # noqa: RUF009
    head_use_batch_norm: bool = False

    # Pooling and Categorical Encoding
    pooling_method: str = "avg"
    bidirectional: bool = False
    use_learnable_interaction: bool = False
    use_cls: bool = False
    shuffle_embeddings: bool = False
    cat_encoding: str = "int"

    # Additional Features
    AD_weight_decay: bool = True
    BC_layer_norm: bool = False
    use_pscan: bool = False
    n_attention_layers: int = 1
