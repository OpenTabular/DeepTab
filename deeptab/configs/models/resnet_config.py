from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn

from ..core import BaseModelConfig


@dataclass
class ResNetConfig(BaseModelConfig):
    """Architecture-only configuration for ResNet models (DeepTab 2.0 API).

    Parameters
    ----------
    activation : Callable, default=nn.SELU()
        Activation function for the ResNet layers.
    layer_sizes : list, default=[256, 128, 32]
        Width schedule of the ResNet. The initial projection maps the input to
        ``layer_sizes[0]``, and residual block ``i`` maps ``layer_sizes[i]`` to
        ``layer_sizes[i + 1]``.
    dropout : float, default=0.5
        Dropout rate for regularization.
    norm : bool, default=False
        Whether to use normalization in the ResNet.
    num_blocks : int, default=3
        Number of residual blocks in the ResNet. Blocks beyond the width
        schedule keep the last width of ``layer_sizes``; with fewer blocks the
        output layer reads the width of the last block.
    """

    # Override parent defaults
    activation: Callable = nn.SELU()  # noqa: RUF009

    # ResNet-specific architecture
    layer_sizes: list = field(default_factory=lambda: [256, 128, 32])
    dropout: float = 0.5
    norm: bool = False
    num_blocks: int = 3
