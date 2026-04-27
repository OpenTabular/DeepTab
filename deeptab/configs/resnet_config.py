from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn

from .base_config import BaseConfig


@dataclass
class DefaultResNetConfig(BaseConfig):
    """Configuration class for the default ResNet model with predefined hyperparameters.

    Parameters
    ----------
    layer_sizes : list, default=(256, 128, 32)
        Sizes of the layers in the ResNet.
    activation : callable, default=nn.SELU()
        Activation function for the ResNet layers.
    skip_layers : bool, default=False
        Whether to skip layers in the ResNet.
    dropout : float, default=0.5
        Dropout rate for regularization.
    norm : bool, default=False
        Whether to use normalization in the ResNet.
    use_glu : bool, default=False
        Whether to use Gated Linear Units (GLU) in the ResNet.
    skip_connections : bool, default=True
        Whether to use skip connections in the ResNet.
    num_blocks : int, default=3
        Number of residual blocks in the ResNet.
    average_embeddings : bool, default=True
        Whether to average embeddings during the forward pass.
    """

    # model params
    layer_sizes: list = field(default_factory=lambda: [256, 128, 32])
    activation: Callable = nn.SELU()  # noqa: RUF009
    skip_layers: bool = False
    dropout: float = 0.5
    norm: bool = False
    use_glu: bool = False
    skip_connections: bool = True
    num_blocks: int = 3

    # embedding params
    average_embeddings: bool = True
