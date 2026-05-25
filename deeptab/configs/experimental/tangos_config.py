from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn

from ..core import BaseModelConfig


@dataclass
class TangosConfig(BaseModelConfig):
    """Architecture-only configuration for Tangos models (DeepTab 2.0 API).

    Parameters
    ----------
    activation : Callable, default=nn.ReLU()
        Activation function for the TANGOS layers.
    layer_sizes : list, default=[256, 128, 32]
        Sizes of the layers in the TANGOS.
    skip_layers : bool, default=False
        Whether to skip layers in the TANGOS.
    dropout : float, default=0.2
        Dropout rate for regularization.
    use_glu : bool, default=False
        Whether to use Gated Linear Units (GLU) in the TANGOS.
    skip_connections : bool, default=False
        Whether to use skip connections in the TANGOS.
    lamda1 : float, default=0.5
        Weight on the task-specific orthogonality regularisation term.
    lamda2 : float, default=0.1
        Weight on the cross-task specialisation regularisation term.
    subsample : float, default=0.5
        Fraction of features subsampled for regularisation estimation.
    """

    # Override parent defaults
    activation: Callable = nn.ReLU()  # noqa: RUF009

    # Tangos-specific architecture
    layer_sizes: list = field(default_factory=lambda: [256, 128, 32])
    skip_layers: bool = False
    dropout: float = 0.2
    use_glu: bool = False
    skip_connections: bool = False
    lamda1: float = 0.5
    lamda2: float = 0.1
    subsample: float = 0.5
