from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn

from ..arch_utils.transformer_utils import ReGLU
from .base_config import BaseConfig


@dataclass
class DefaultAutoIntConfig(BaseConfig):
    """Configuration class for the AutoInt model with predefined hyperparameters.

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
    transformer_dim_feedforward : int, default=256
        Dimensionality of the feed-forward layers in the transformer.
    prenorm : bool, default=False
        Whether to apply normalization before last layer.
    bias : bool, default=True
        Whether to use bias in linear layers.
    cat_encoding : str, default="int"
        Method for encoding categorical features ('int', 'one-hot', or 'linear').
    kv_compression : float, default=0.5
        Compression ratio for key-value pairs.
    kv_compression_sharing : str, default='key-value'
        Sharing strategy for key-value compression ('headwise', or 'key-value').
    """

    # Architecture Parameters
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8
    attn_dropout: float = 0.2
    fprenorm: bool = False
    transformer_dim_feedforward: int = 256
    bias: bool = True

    use_cls: bool = False
    cat_encoding: str = "int"

    kv_compression: float = 0.5
    kv_compression_sharing: str = "key-value"
