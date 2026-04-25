import numpy as np
import torch
import torch.nn as nn

from ..arch_utils.enode_utils import DenseBlock
from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..arch_utils.mlp_utils import MLPhead
from ..configs.enode_config import DefaultENODEConfig
from ..utils.get_feature_dimensions import get_feature_dimensions
from .utils.basemodel import BaseModel


class ENODE(BaseModel):
    """A Neural Oblivious Decision Ensemble (NODE) model for tabular data, integrating feature embeddings, dense blocks,
    and customizable heads for predictions.

    Parameters
    ----------
    cat_feature_info : dict
        Dictionary containing information about categorical features, including their names and dimensions.
    num_feature_info : dict
        Dictionary containing information about numerical features, including their names and dimensions.
    num_classes : int, optional
        The number of output classes or target dimensions for regression, by default 1.
    config : DefaultNODEConfig, optional
        Configuration object containing model hyperparameters such as the number of dense layers, layer dimensions,
        tree depth, embedding settings, and head layer configurations, by default DefaultNODEConfig().
    **kwargs : dict
        Additional keyword arguments for the BaseModel class.

    Attributes
    ----------
    cat_feature_info : dict
        Stores categorical feature information.
    num_feature_info : dict
        Stores numerical feature information.
    use_embeddings : bool
        Flag indicating if embeddings should be used for categorical and numerical features.
    embedding_layer : EmbeddingLayer, optional
        Embedding layer for features, used if `use_embeddings` is enabled.
    d_out : int
        The output dimension, usually set to `num_classes`.
    block : DenseBlock
        Dense block layer for feature transformations based on the NODE approach.
    tabular_head : MLPhead
        MLPhead layer to produce the final prediction based on the output of the dense block.

    Methods
    -------
    forward(num_features, cat_features)
        Perform a forward pass through the model, including embedding (if enabled), dense transformations,
        and prediction steps.
    """

    def __init__(
        self,
        feature_information: tuple,  # Expecting (num_feature_info, cat_feature_info, embedding_feature_info)
        num_classes: int = 1,
        config: DefaultENODEConfig = DefaultENODEConfig(),  # noqa: B008
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.returns_ensemble = False

        self.embedding_layer = EmbeddingLayer(
            *feature_information,
            config=config,
        )

        input_dim = np.sum([len(info) for info in feature_information])

        self.d_out = num_classes
        self.block = DenseBlock(
            input_dim=input_dim,
            num_layers=self.hparams.num_layers,
            layer_dim=self.hparams.layer_dim,
            embed_dim=self.hparams.d_model,
            depth=self.hparams.depth,
            tree_dim=self.hparams.tree_dim,
            flatten_output=True,
        )

        self.tabular_head = nn.Sequential(
            nn.Linear(self.hparams.d_model, self.hparams.d_model),
            nn.ReLU(),
            nn.Dropout(self.hparams.head_dropout),
            nn.Linear(self.hparams.d_model, num_classes),
        )

    def forward(self, *data):
        """Forward pass through the NODE model.

        Parameters
        ----------
        num_features : torch.Tensor
            Numerical features tensor of shape [batch_size, num_numerical_features].
        cat_features : torch.Tensor
            Categorical features tensor of shape [batch_size, num_categorical_features].

        Returns
        -------
        torch.Tensor
            Model output of shape [batch_size, num_classes].
        """

        x = self.embedding_layer(*data)

        x = self.block(x).squeeze(-1)
        x = x.mean(axis=1)
        x = self.tabular_head(x)
        return x
