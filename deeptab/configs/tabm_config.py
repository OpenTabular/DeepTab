from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import torch.nn as nn

from .base_config import BaseConfig


@dataclass
class DefaultTabMConfig(BaseConfig):
    """Configuration class for the TabM model with batch ensembling and predefined hyperparameters.

    Parameters
    ----------
    layer_sizes : list, default=(512, 512, 128)
        Sizes of the layers in the model.
    activation : callable, default=nn.ReLU()
        Activation function for the model layers.
    dropout : float, default=0.3
        Dropout rate for regularization.
    norm : str, default=None
        Normalization method to be used, if any.
    use_glu : bool, default=False
        Whether to use Gated Linear Units (GLU) in the model.
    ensemble_size : int, default=32
        Number of ensemble members for batch ensembling.
    ensemble_scaling_in : bool, default=True
        Whether to use input scaling for each ensemble member.
    ensemble_scaling_out : bool, default=True
        Whether to use output scaling for each ensemble member.
    ensemble_bias : bool, default=True
        Whether to use a unique bias term for each ensemble member.
    scaling_init : {"ones", "random-signs", "normal"}, default="normal"
        Initialization method for scaling weights.
    average_ensembles : bool, default=False
        Whether to average the outputs of the ensembles.
    model_type : {"mini", "full"}, default="mini"
        Model type to use ('mini' for reduced version, 'full' for complete model).
    """

    # arch params
    layer_sizes: list = field(default_factory=lambda: [256, 256, 128])
    activation: Callable = nn.ReLU()  # noqa: RUF009
    dropout: float = 0.5
    norm: str | None = None
    use_glu: bool = False

    # Batch ensembling specific configurations
    ensemble_size: int = 32
    ensemble_scaling_in: bool = True
    ensemble_scaling_out: bool = True
    ensemble_bias: bool = True
    scaling_init: Literal["ones", "random-signs", "normal"] = "ones"
    average_ensembles: bool = False
    model_type: Literal["mini", "full"] = "mini"
    average_embeddings: bool = True
