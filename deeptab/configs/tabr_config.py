from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn

from .base_config import BaseConfig


@dataclass
class DefaultTabRConfig(BaseConfig):
    """Configuration class for the default TabR model with predefined hyperparameters.
    Parameters
    ----------
    """

    # Optimizer Parameters
    lr: float = 0.0003121273641315169
    weight_decay: float = 1.2260352006404615e-06
    lr_patience = 10
    lr_factor: float = 0.1  # Factor for LR scheduler

    # Architecture Parameters
    d_main: int = 256
    context_dropout: float = 0.38920071545944357
    d_multiplier: int = 2
    encoder_n_blocks: int = 0
    predictor_n_blocks: int = 1
    mixer_normalization: str = "auto"
    dropout0: float = 0.38852797479169876
    dropout1: float = 0.0
    normalization: str = "LayerNorm"
    activation: Callable = nn.ReLU()  # noqa: RUF009
    memory_efficient: bool = False
    candidate_encoding_batch_size: int = 0
    context_size: int = 96

    # Embedding Parameters
    embedding_type: str = "plr"
    plr_lite: bool = True
    n_frequencies: int = 75
    frequencies_init_scale: float = 0.045
