from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn

from ..core import BaseModelConfig


@dataclass
class TabRConfig(BaseModelConfig):
    """Architecture-only configuration for TabR models (DeepTab 2.0 API).

    Training fields (``lr``, ``weight_decay``, ``lr_factor``) are configured
    via :class:`~deeptab.configs.trainer_config.TrainerConfig`.

    Parameters
    ----------
    embedding_type : str, default='plr'
        Type of feature embedding to use (e.g., 'plr', 'ple').
    plr_lite : bool, default=True
        Whether to use the lightweight PLR embedding variant.
    n_frequencies : int, default=75
        Number of random Fourier feature frequencies.
    frequencies_init_scale : float, default=0.045
        Scale for initializing Fourier feature frequencies.
    d_main : int, default=256
        Main hidden dimensionality of the predictor network.
    context_dropout : float, default=0.38920071545944357
        Dropout applied to context (candidate) representations.
    d_multiplier : int, default=2
        Multiplier for intermediate dimensions inside the predictor.
    encoder_n_blocks : int, default=0
        Number of residual blocks in the feature encoder.
    predictor_n_blocks : int, default=1
        Number of residual blocks in the predictor network.
    mixer_normalization : str, default='auto'
        Normalization strategy for the mixer (``'auto'`` selects adaptively).
    dropout0 : float, default=0.38852797479169876
        Dropout rate on the first linear projection.
    dropout1 : float, default=0.0
        Dropout rate on the second linear projection.
    normalization : str, default='LayerNorm'
        Type of normalization layer to use.
    memory_efficient : bool, default=False
        Whether to trade compute for lower memory in candidate lookups.
    candidate_encoding_batch_size : int, default=0
        Batch size for encoding candidates (0 = full batch).
    context_size : int, default=96
        Number of nearest-neighbour candidates to retrieve per sample.
    """

    # Override embedding defaults specific to TabR
    embedding_type: str = "plr"
    plr_lite: bool = True
    n_frequencies: int = 75
    frequencies_init_scale: float = 0.045

    # Architecture
    d_main: int = 256
    context_dropout: float = 0.38920071545944357
    d_multiplier: int = 2
    encoder_n_blocks: int = 0
    predictor_n_blocks: int = 1
    mixer_normalization: str = "auto"
    dropout0: float = 0.38852797479169876
    dropout1: float = 0.0
    normalization: str = "LayerNorm"
    memory_efficient: bool = False
    candidate_encoding_batch_size: int = 0
    context_size: int = 96
