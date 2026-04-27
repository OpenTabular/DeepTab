import numpy as np
import torch

from ..arch_utils.get_norm_fn import get_normalization_layer
from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..arch_utils.mamba_utils.mambattn_arch import MambAttn
from ..arch_utils.mlp_utils import MLPhead
from ..configs.mambattention_config import DefaultMambAttentionConfig
from .utils.basemodel import BaseModel


class MambAttention(BaseModel):
    """A MambAttention model for tabular data, integrating feature embeddings, attention-based Mamba transformations,
    and a customizable architecture for handling categorical and numerical features.

    Parameters
    ----------
    cat_feature_info : dict
        Dictionary containing information about categorical features, including their names and dimensions.
    num_feature_info : dict
        Dictionary containing information about numerical features, including their names and dimensions.
    num_classes : int, optional
        The number of output classes or target dimensions for regression, by default 1.
    config : DefaultMambAttentionConfig, optional
        Configuration object with model hyperparameters such as dropout rates, head layer sizes, attention settings,
        and other architectural configurations, by default DefaultMambAttentionConfig().
    **kwargs : dict
        Additional keyword arguments for the BaseModel class.

    Attributes
    ----------
    pooling_method : str
        Pooling method to aggregate features after the Mamba attention layer.
    shuffle_embeddings : bool
        Flag indicating if embeddings should be shuffled, as specified in the configuration.
    mamba : MambAttn
        Mamba attention layer to process embedded features.
    norm_f : nn.Module
        Normalization layer for the processed features.
    embedding_layer : EmbeddingLayer
        Layer for embedding categorical and numerical features.
    tabular_head : MLPhead
        MLPhead layer to produce the final prediction based on the output of the Mamba attention layer.
    perm : torch.Tensor, optional
        Permutation tensor used for shuffling embeddings, if enabled.

    Methods
    -------
    forward(num_features, cat_features)
        Perform a forward pass through the model, including embedding, Mamba attention transformation, pooling,
        and prediction steps.
    """

    def __init__(
        self,
        feature_information: tuple,  # Expecting (num_feature_info, cat_feature_info, embedding_feature_info)
        num_classes=1,
        config: DefaultMambAttentionConfig = DefaultMambAttentionConfig(),  # noqa: B008
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["feature_information"])

        self.returns_ensemble = False

        try:
            self.pooling_method = self.hparams.pooling_method
        except AttributeError:
            self.pooling_method = config.pooling_method

        try:
            self.shuffle_embeddings = self.hparams.shuffle_embeddings
        except AttributeError:
            self.shuffle_embeddings = config.shuffle_embeddings

        self.mamba = MambAttn(config)
        self.norm_f = get_normalization_layer(config)

        # embedding layer
        self.embedding_layer = EmbeddingLayer(
            *feature_information,
            config=config,
        )

        try:
            head_activation = self.hparams.head_activation
        except AttributeError:
            head_activation = config.head_activation

        try:
            input_dim = self.hparams.d_model
        except AttributeError:
            input_dim = config.d_model

        self.tabular_head = MLPhead(
            input_dim=input_dim,
            config=config,
            output_dim=num_classes,
        )

        if self.shuffle_embeddings:
            self.perm = torch.randperm(self.embedding_layer.seq_len)

        # pooling
        n_inputs = np.sum([len(info) for info in feature_information])
        self.initialize_pooling_layers(config=config, n_inputs=n_inputs)

    def forward(self, *data):
        """Defines the forward pass of the model.

        Parameters
        ----------
        data : tuple
            Input tuple of tensors of num_features, cat_features, embeddings.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.embedding_layer(*data)

        if self.shuffle_embeddings:
            x = x[:, self.perm, :]

        x = self.mamba(x)

        x = self.pool_sequence(x)

        x = self.norm_f(x)  # type: ignore
        preds = self.tabular_head(x)

        return preds
