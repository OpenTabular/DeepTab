from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn

from ..base_model_config import BaseModelConfig


@dataclass
class ModernNCAConfig(BaseModelConfig):
    """Architecture-only configuration for ModernNCA models (DeepTab 2.0 API).

    Parameters
    ----------
    embedding_type : str, default='plr'
        Type of feature embedding to use (e.g., 'plr', 'ple').
    plr_lite : bool, default=True
        Whether to use the lightweight PLR embedding variant.
    n_frequencies : int, default=75
        Number of random Fourier feature frequencies.
    frequencies_init_scale : float, default=0.045
        Scale for initializing Fourier feature frequencies.
    dim : int, default=128
        Embedding dimensionality per feature.
    d_block : int, default=512
        Hidden size of each residual block.
    n_blocks : int, default=4
        Number of residual blocks.
    dropout : float, default=0.1
        Dropout rate applied inside each block.
    temperature : float, default=0.75
        Temperature scaling for NCA softmax similarity.
    sample_rate : float, default=0.5
        Fraction of training candidates used per forward pass.
    num_embeddings : dict | None, default=None
        Optional dict mapping feature indices to embedding sizes.
    head_layer_sizes : list, default=field(default_factory=list
        Sizes of the fully connected layers in the prediction head.
    head_dropout : float, default=0.5
        Dropout rate for the head layers.
    head_skip_layers : bool, default=False
        Whether to use skip connections in the head layers.
    head_activation : Callable, default=nn.SELU()
        Activation function for the head layers.
    head_use_batch_norm : bool, default=False
        Whether to use batch normalization in the head layers.
    """

    # Override parent defaults
    embedding_type: str = "plr"
    plr_lite: bool = True
    n_frequencies: int = 75
    frequencies_init_scale: float = 0.045

    # ModernNCA-specific architecture
    dim: int = 128
    d_block: int = 512
    n_blocks: int = 4
    dropout: float = 0.1
    temperature: float = 0.75
    sample_rate: float = 0.5
    num_embeddings: dict | None = None

    # Head
    head_layer_sizes: list = field(default_factory=list)
    head_dropout: float = 0.5
    head_skip_layers: bool = False
    head_activation: Callable = nn.SELU()  # noqa: RUF009
    head_use_batch_norm: bool = False
