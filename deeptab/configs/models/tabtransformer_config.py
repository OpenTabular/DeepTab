from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn

from deeptab.nn.blocks.transformer import ReGLU

from ..core import BaseModelConfig


@dataclass
class TabTransformerConfig(BaseModelConfig):
    """Architecture-only configuration for TabTransformer models (DeepTab 2.0 API).

    Parameters
    ----------
    d_model : int, default=128
        Dimensionality of embeddings or model representations.
    activation : Callable, default=nn.SELU()
        Activation function for the transformer layers.
    n_layers : int, default=4
        Number of layers in the transformer.
    n_heads : int, default=8
        Number of attention heads in the transformer.
    attn_dropout : float, default=0.2
        Dropout rate for the attention mechanism.
    ff_dropout : float, default=0.1
        Dropout rate for the feed-forward layers.
    norm : str, default='LayerNorm'
        Normalization method to be used.
    transformer_activation : Callable, default=ReGLU()
        Activation function for the transformer layers.
    transformer_dim_feedforward : int, default=512
        Dimensionality of the feed-forward layers in the transformer.
    norm_first : bool, default=True
        Whether to apply normalization before other operations in each
        transformer block.
    bias : bool, default=True
        Whether to use bias in the linear layers.
    head_layer_sizes : list, default=field(default_factory=list
        Sizes of the layers in the model's head.
    head_dropout : float, default=0.5
        Dropout rate for the head layers.
    head_skip_layers : bool, default=False
        Whether to skip layers in the head.
    head_activation : Callable, default=nn.SELU()
        Activation function for the head layers.
    head_use_batch_norm : bool, default=False
        Whether to use batch normalization in the head layers.
    pooling_method : str, default='avg'
        Pooling method to be used ('cls', 'avg', etc.).
    """

    # Override parent defaults
    d_model: int = 128
    activation: Callable = nn.SELU()  # noqa: RUF009

    # Transformer-specific architecture
    n_layers: int = 4
    n_heads: int = 8
    attn_dropout: float = 0.2
    ff_dropout: float = 0.1
    norm: str = "LayerNorm"
    transformer_activation: Callable = ReGLU()  # noqa: RUF009
    transformer_dim_feedforward: int = 512
    norm_first: bool = True
    bias: bool = True

    # Head
    head_layer_sizes: list = field(default_factory=list)
    head_dropout: float = 0.5
    head_skip_layers: bool = False
    head_activation: Callable = nn.SELU()  # noqa: RUF009
    head_use_batch_norm: bool = False

    # Pooling
    pooling_method: str = "avg"
