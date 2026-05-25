from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn

from ..base_model_config import BaseModelConfig


@dataclass
class ResNetConfig(BaseModelConfig):
    """Architecture-only configuration for ResNet models (DeepTab 2.0 API).

    Parameters
    ----------
    activation : Callable, default=nn.SELU()
        Activation function for the ResNet layers.
    layer_sizes : list, default=[256, 128, 32]
        Sizes of the layers in the ResNet.
    dropout : float, default=0.5
        Dropout rate for regularization.
    norm : bool, default=False
        Whether to use normalization in the ResNet.
    num_blocks : int, default=3
        Number of residual blocks in the ResNet.
    """

    # Override parent defaults
    activation: Callable = nn.SELU()  # noqa: RUF009

    # ResNet-specific architecture
    layer_sizes: list = field(default_factory=lambda: [256, 128, 32])
    dropout: float = 0.5
    norm: bool = False
    num_blocks: int = 3
