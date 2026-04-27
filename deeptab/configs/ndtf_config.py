from dataclasses import dataclass

from .base_config import BaseConfig


@dataclass
class DefaultNDTFConfig(BaseConfig):
    """Configuration class for the default Neural Decision Tree Forest (NDTF) model with predefined hyperparameters.

    Parameters
    ----------
    min_depth : int, default=2
        Minimum depth of trees in the forest. Controls the simplest model structure.
    max_depth : int, default=10
        Maximum depth of trees in the forest. Controls the maximum complexity of the trees.
    temperature : float, default=0.1
        Temperature parameter for softening the node decisions during path probability calculation.
    node_sampling : float, default=0.3
        Fraction of nodes sampled for regularization penalty calculation. Reduces computation by focusing
        on a subset of nodes.
    lamda : float, default=0.3
        Regularization parameter to control the complexity of the paths, penalizing overconfident
        or imbalanced paths.
    n_ensembles : int, default=12
        Number of trees in the forest
    penalty_factor : float, default=0.01
        Factor with which the penalty is multiplied
    """

    min_depth: int = 4
    max_depth: int = 16
    temperature: float = 0.1
    node_sampling: float = 0.3
    lamda: float = 0.3
    n_ensembles: int = 12
    penalty_factor: float = 1e-08
