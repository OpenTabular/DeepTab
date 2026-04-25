import numpy as np
import torch

from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..arch_utils.mlp_utils import MLPhead
from ..arch_utils.node_utils import DenseBlock
from ..configs.node_config import DefaultNODEConfig
from ..utils.get_feature_dimensions import get_feature_dimensions
from .utils.basemodel import BaseModel


class NODE(BaseModel):
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
        config: DefaultNODEConfig = DefaultNODEConfig(),  # noqa: B008
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.returns_ensemble = False

        if self.hparams.use_embeddings:
            self.embedding_layer = EmbeddingLayer(
                *feature_information,
                config=config,
            )
            input_dim = np.sum([len(info) * self.hparams.d_model for info in feature_information])

        else:
            input_dim = get_feature_dimensions(*feature_information)

        self.d_out = num_classes
        self.block = DenseBlock(
            input_dim=input_dim,
            num_layers=self.hparams.num_layers,
            layer_dim=self.hparams.layer_dim,
            depth=self.hparams.depth,
            tree_dim=self.hparams.tree_dim,
            flatten_output=True,
        )

        self.tabular_head = MLPhead(
            input_dim=self.hparams.num_layers * self.hparams.layer_dim,
            config=config,
            output_dim=num_classes,
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
        if self.hparams.use_embeddings:
            x = self.embedding_layer(*data)
            B, S, D = x.shape
            x = x.reshape(B, S * D)
        else:
            x = torch.cat([t for tensors in data for t in tensors], dim=1)

        x = self.block(x).squeeze(-1)
        x = self.tabular_head(x)
        return x
