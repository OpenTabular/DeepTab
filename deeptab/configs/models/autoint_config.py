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
    kv_compression : float | None, default=None
        Fraction of feature tokens kept for attention keys and values, in
        ``(0, 1]``. ``None`` attends over every feature token.
    kv_compression_sharing : str, default='key-value'
        Sharing strategy for key-value compression: ``'layerwise'``,
        ``'headwise'``, or ``'key-value'``. Used only when
        ``kv_compression`` is set.
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
    kv_compression: float | None = None
    kv_compression_sharing: str = "key-value"
