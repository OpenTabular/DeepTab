from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn

from .base_config import BaseConfig


@dataclass
class DefaultModernNCAConfig(BaseConfig):
    """
    Default configuration for the ModernNCA model.
    """

    # Architecture Parameters
    dim: int = 128  # Hidden dimension for encoding
    d_block: int = 512  # Block size for MLP layers
    n_blocks: int = 4  # Number of MLP blocks
    dropout: float = 0.1  # Dropout rate
    temperature: float = 0.75  # Temperature scaling for distance weighting
    sample_rate: float = 0.5  # Fraction of candidate samples used
    num_embeddings: dict | None = None  # Dictionary for categorical embeddings

    # Training Parameters
    optimizer_type: str = "AdamW"  # Optimizer type
    weight_decay: float = 1e-5  # Weight decay for optimizer
    learning_rate: float = 1e-02  # Learning rate
    lr_patience: int = 10  # Patience for LR scheduler
    lr_factor: float = 0.1  # Factor for LR scheduler

    # Head Parameters
    head_layer_sizes: list = field(default_factory=list)
    head_dropout: float = 0.5
    head_skip_layers: bool = False
    head_activation: Callable = nn.SELU()  # noqa: RUF009
    head_use_batch_norm: bool = False

    # Embedding Parameters
    embedding_type: str = "plr"
    plr_lite: bool = True
    n_frequencies: int = 75
    frequencies_init_scale: float = 0.045
