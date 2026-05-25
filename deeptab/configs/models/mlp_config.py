from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn

from ..core import BaseModelConfig


@dataclass
class MLPConfig(BaseModelConfig):
    """Architecture-only configuration for MLP models (DeepTab 2.0 API).

    Contains only structural hyperparameters.  Training parameters (``lr``,
    ``max_epochs``, …) go in :class:`~deeptab.configs.trainer_config.TrainerConfig`
    and preprocessing parameters go in
    :class:`~deeptab.configs.preprocessing_config.PreprocessingConfig`.

    Parameters
    ----------
    layer_sizes : list, default=[256, 128, 32]
        Number of units in each hidden layer.
    activation : Callable, default=nn.ReLU()
        Activation function for the MLP layers.
    skip_layers : bool, default=False
        Whether to include skip layers.
    dropout : float, default=0.2
        Dropout rate applied after each hidden layer.
    use_glu : bool, default=False
        Whether to use Gated Linear Units instead of the plain activation.
    skip_connections : bool, default=False
        Whether to use residual/skip connections between layers.
    """

    # MLP-specific architecture parameters
    layer_sizes: list = field(default_factory=lambda: [256, 128, 32])
    dropout: float = 0.2
    use_glu: bool = False
    skip_connections: bool = False
