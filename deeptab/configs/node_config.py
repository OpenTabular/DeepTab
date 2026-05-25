from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn

from .base_model_config import BaseModelConfig


@dataclass
class NODEConfig(BaseModelConfig):
    """Architecture-only configuration for NODE models (DeepTab 2.0 API).

    Parameters
    ----------
    num_layers : int, default=4
        Number of dense layers in the model.
    layer_dim : int, default=128
        Dimensionality of each dense layer.
    tree_dim : int, default=1
        Dimensionality of the output from each tree leaf.
    depth : int, default=6
        Depth of each decision tree in the ensemble.
    norm : str | None, default=None
        Type of normalization to use in the model.
    head_layer_sizes : list, default=field(default_factory=list
        Sizes of the layers in the model's head.
    head_dropout : float, default=0.3
        Dropout rate for the head layers.
    head_skip_layers : bool, default=False
        Whether to skip layers in the head.
    head_activation : Callable, default=nn.ReLU()
        Activation function for the head layers.
    head_use_batch_norm : bool, default=False
        Whether to use batch normalization in the head layers.
    """

    # NODE-specific architecture
    num_layers: int = 4
    layer_dim: int = 128
    tree_dim: int = 1
    depth: int = 6
    norm: str | None = None

    # Head
    head_layer_sizes: list = field(default_factory=list)
    head_dropout: float = 0.3
    head_skip_layers: bool = False
    head_activation: Callable = nn.ReLU()  # noqa: RUF009
    head_use_batch_norm: bool = False
