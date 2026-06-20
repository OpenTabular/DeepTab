from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn

from deeptab.nn.blocks.transformer import ReGLU

from ..core import BaseModelConfig


@dataclass
class AutoIntConfig(BaseModelConfig):
    """Architecture-only configuration for AutoInt models (DeepTab 2.0 API).

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
    fprenorm : bool, default=False
        Whether to apply pre-normalization in attention layers.
    bias : bool, default=True
        Whether to use bias in linear layers.
    use_cls : bool, default=False
        Whether to use a CLS token for pooling instead of averaging.
    kv_compression : float, default=0.5
        Compression ratio for key-value pairs.
    kv_compression_sharing : str, default='key-value'
        Sharing strategy for key-value compression ('headwise', or 'key-
        value').
    """

    # Override parent defaults
    d_model: int = 128

    # Transformer-specific architecture
    n_layers: int = 4
    n_heads: int = 8
    attn_dropout: float = 0.2
    transformer_dim_feedforward: int = 256
    fprenorm: bool = False
    bias: bool = True
    use_cls: bool = False
    kv_compression: float = 0.5
    kv_compression_sharing: str = "key-value"
