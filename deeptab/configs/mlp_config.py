from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn

from .base_config import BaseConfig


@dataclass
class DefaultMLPConfig(BaseConfig):
    """Configuration class for the default Multi-Layer Perceptron (MLP) model with predefined hyperparameters.

    Parameters
    ----------
    layer_sizes : list, default=(256, 128, 32)
        Sizes of the layers in the MLP.
    activation : callable, default=nn.ReLU()
        Activation function for the MLP layers.
    skip_layers : bool, default=False
        Whether to skip layers in the MLP.
    dropout : float, default=0.2
        Dropout rate for regularization.
    use_glu : bool, default=False
        Whether to use Gated Linear Units (GLU) in the MLP.
    skip_connections : bool, default=False
        Whether to use skip connections in the MLP.
    """

    # Architecture Parameters
    layer_sizes: list = field(default_factory=lambda: [256, 128, 32])
    activation: Callable = nn.ReLU()  # noqa: RUF009
    skip_layers: bool = False
    dropout: float = 0.2
    use_glu: bool = False
    skip_connections: bool = False
