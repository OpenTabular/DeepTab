from collections.abc import Callable  # inherited by sphinx-autodoc-typehints
from dataclasses import dataclass

from deeptab.core.exceptions import incompatible_params_error

from ..core import BaseModelConfig


@dataclass
class NDTFConfig(BaseModelConfig):
    """Architecture-only configuration for NDTF models (DeepTab 2.0 API).

    Parameters
    ----------
    min_depth : int, default=4
        Minimum depth of trees in the forest. Controls the simplest model
        structure.
    max_depth : int, default=16
        Exclusive upper bound of the tree depths: depths are drawn from
        ``min_depth`` up to ``max_depth - 1``, so ``max_depth`` must be greater
        than ``min_depth``. Controls the maximum complexity of the trees.
    temperature : float, default=0.1
        Temperature parameter for softening the node decisions during path
        probability calculation.
    node_sampling : float, default=0.3
        Fraction of nodes sampled for regularization penalty calculation.
        Reduces computation by focusing on a subset of nodes.
    lamda : float, default=0.3
        Regularization parameter to control the complexity of the paths,
        penalizing overconfident or imbalanced paths.
    n_ensembles : int, default=12
        Number of trees in the forest
    penalty_factor : float, default=1e-08
        Factor with which the penalty is multiplied
    """

    min_depth: int = 4
    max_depth: int = 16
    temperature: float = 0.1
    node_sampling: float = 0.3
    lamda: float = 0.3
    n_ensembles: int = 12
    penalty_factor: float = 1e-8

    def __post_init__(self) -> None:  # type: ignore[override]
        super().__post_init__()
        if self.max_depth <= self.min_depth:
            raise incompatible_params_error(
                "NDTFConfig",
                f"max_depth ({self.max_depth}) must be greater than min_depth ({self.min_depth}): "
                "tree depths are drawn from min_depth up to, but excluding, max_depth.",
            )
