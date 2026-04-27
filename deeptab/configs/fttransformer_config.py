from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn

from ..arch_utils.transformer_utils import ReGLU
from .base_config import BaseConfig


@dataclass
class DefaultFTTransformerConfig(BaseConfig):
    """Configuration class for the FT Transformer model with predefined hyperparameters.

    Parameters
    ----------
    d_model : int, default=128
        Dimensionality of the transformer model.
    n_layers : int, default=4
        Number of transformer layers.
    n_heads : int, default=8
        Number of attention heads in the transformer.
    attn_dropout : float, default=0.2
        Dropout rate for the attention mechanism.
    ff_dropout : float, default=0.1
        Dropout rate for the feed-forward layers.
    norm : str, default="LayerNorm"
        Type of normalization to be used ('LayerNorm', 'RMSNorm', etc.).
    activation : callable, default=nn.SELU()
        Activation function for the transformer layers.
    transformer_activation : callable, default=ReGLU()
        Activation function for the transformer feed-forward layers.
    transformer_dim_feedforward : int, default=256
        Dimensionality of the feed-forward layers in the transformer.
    layer_norm_eps : float, default=1e-05
        Epsilon value for layer normalization to improve numerical stability.
    norm_first : bool, default=False
        Whether to apply normalization before other operations in each transformer block.
    bias : bool, default=True
        Whether to use bias in linear layers.
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
        Pooling method to be used ('cls', 'avg', etc.).
    use_cls : bool, default=False
        Whether to use a CLS token for pooling.
    cat_encoding : str, default="int"
        Method for encoding categorical features ('int', 'one-hot', or 'linear').
    """

    # Architecture Parameters
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8
    attn_dropout: float = 0.2
    ff_dropout: float = 0.1
    norm: str = "LayerNorm"
    activation: Callable = nn.SELU()  # noqa: RUF009
    transformer_activation: Callable = ReGLU()  # noqa: RUF009
    transformer_dim_feedforward: int = 256
    layer_norm_eps: float = 1e-05
    norm_first: bool = False
    bias: bool = True

    # Head Parameters
    head_layer_sizes: list = field(default_factory=list)
    head_dropout: float = 0.5
    head_skip_layers: bool = False
    head_activation: Callable = nn.SELU()  # noqa: RUF009
    head_use_batch_norm: bool = False

    # Pooling and Categorical Encoding
    pooling_method: str = "avg"
    use_cls: bool = False
    cat_encoding: str = "int"
