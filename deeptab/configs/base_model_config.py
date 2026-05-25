from collections.abc import Callable
from dataclasses import dataclass

import torch.nn as nn
from sklearn.base import BaseEstimator


@dataclass
class BaseModelConfig(BaseEstimator):
    """Shared architecture hyperparameters for all DeepTab models.

    This class contains only architectural / structural configuration.
    Training-related parameters (``lr``, ``weight_decay``, ``max_epochs``, …)
    belong in :class:`~deeptab.configs.trainer_config.TrainerConfig`.
    Preprocessing parameters belong in
    :class:`~deeptab.configs.preprocessing_config.PreprocessingConfig`.

    Parameters
    ----------
    use_embeddings : bool, default=False
        Whether to use embedding layers for numerical/categorical features.
    embedding_activation : Callable, default=nn.Identity()
        Activation function applied to embeddings.
    embedding_type : str, default="linear"
        Type of embedding (``"linear"``, ``"plr"``, etc.).
    embedding_bias : bool, default=False
        Whether to add a bias term to embedding layers.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalisation after the embedding layer.
    d_model : int, default=32
        Embedding / model dimensionality.
    plr_lite : bool, default=False
        Whether to use the lightweight PLR embedding variant.
    n_frequencies : int, default=48
        Number of frequency components for PLR embeddings.
    frequencies_init_scale : float, default=0.01
        Initial scale for PLR frequency components.
    embedding_projection : bool, default=True
        Whether to apply a linear projection after embeddings.
    batch_norm : bool, default=False
        Whether to use batch normalisation in the model body.
    layer_norm : bool, default=False
        Whether to use layer normalisation in the model body.
    layer_norm_eps : float, default=1e-5
        Epsilon for layer normalisation numerical stability.
    activation : Callable, default=nn.ReLU()
        Activation function used throughout the model body.
    cat_encoding : str, default="int"
        How categorical features are encoded at the model input
        (``"int"``, ``"one-hot"``, ``"linear"``).
    """

    # Embedding parameters
    use_embeddings: bool = False
    embedding_activation: Callable = nn.Identity()  # noqa: RUF009
    embedding_type: str = "linear"
    embedding_bias: bool = False
    layer_norm_after_embedding: bool = False
    d_model: int = 32
    plr_lite: bool = False
    n_frequencies: int = 48
    frequencies_init_scale: float = 0.01
    embedding_projection: bool = True

    # Architecture parameters
    batch_norm: bool = False
    layer_norm: bool = False
    layer_norm_eps: float = 1e-05
    activation: Callable = nn.ReLU()  # noqa: RUF009
    cat_encoding: str = "int"
