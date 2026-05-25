from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn

from ..arch_utils.transformer_utils import ReGLU
from .base_model_config import BaseModelConfig


@dataclass
class TromptConfig(BaseModelConfig):
    """Architecture-only configuration for Trompt models (DeepTab 2.0 API).

    Parameters
    ----------
    d_model : int, default=128
        Dimensionality of the transformer model.
    n_cycles : int, default=6
        Number of cycles in the Trompt model.
    n_cells : int, default=4
        Number of cells in each cycle.
    P : int, default=128
        Number of steps in the Trompt model.
    """

    # Trompt-specific architecture
    d_model: int = 128
    n_cycles: int = 6
    n_cells: int = 4
    P: int = 128
