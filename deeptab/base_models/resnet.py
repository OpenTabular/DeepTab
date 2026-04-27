import numpy as np
import torch
import torch.nn as nn

from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..arch_utils.resnet_utils import ResidualBlock
from ..configs.resnet_config import DefaultResNetConfig
from ..utils.get_feature_dimensions import get_feature_dimensions
from .utils.basemodel import BaseModel


class ResNet(BaseModel):
    """A ResNet model for tabular data, combining feature embeddings, residual blocks, and customizable architecture for
    processing categorical and numerical features.

    Parameters
    ----------
    cat_feature_info : dict
        Dictionary containing information about categorical features, including their names and dimensions.
    num_feature_info : dict
        Dictionary containing information about numerical features, including their names and dimensions.
    num_classes : int, optional
        The number of output classes or target dimensions for regression, by default 1.
    config : DefaultResNetConfig, optional
        Configuration object containing model hyperparameters such as layer sizes, number of residual blocks,
        dropout rates, activation functions, and normalization settings, by default DefaultResNetConfig().
    **kwargs : dict
        Additional keyword arguments for the BaseModel class.

    Attributes
    ----------
    layer_sizes : list of int
        List specifying the number of units in each layer of the ResNet.
    cat_feature_info : dict
        Stores categorical feature information.
    num_feature_info : dict
        Stores numerical feature information.
    activation : callable
        Activation function used in the residual blocks.
    use_embeddings : bool
        Flag indicating if embeddings should be used for categorical and numerical features.
    embedding_layer : EmbeddingLayer, optional
        Embedding layer for features, used if `use_embeddings` is enabled.
    initial_layer : nn.Linear
        Initial linear layer to project input features into the model's hidden dimension.
    blocks : nn.ModuleList
        List of residual blocks to process the hidden representations.
    output_layer : nn.Linear
        Output layer that produces the final prediction.

    Methods
    -------
    forward(num_features, cat_features)
        Perform a forward pass through the model, including embedding (if enabled), residual blocks,
        and prediction steps.
    """

    def __init__(
        self,
        feature_information: tuple,  # Expecting (num_feature_info, cat_feature_info, embedding_feature_info)
        num_classes: int = 1,
        config: DefaultResNetConfig = DefaultResNetConfig(),  # noqa: B008
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["feature_information"])

        self.returns_ensemble = False

        if self.hparams.use_embeddings:
            self.embedding_layer = EmbeddingLayer(
                *feature_information,
                config=config,
            )
            input_dim = np.sum([len(info) * self.hparams.d_model for info in feature_information])
        else:
            input_dim = get_feature_dimensions(*feature_information)

        self.initial_layer = nn.Linear(input_dim, self.hparams.layer_sizes[0])

        self.blocks = nn.ModuleList()
        for i in range(self.hparams.num_blocks):
            input_dim = self.hparams.layer_sizes[i]
            output_dim = (
                self.hparams.layer_sizes[i + 1]
                if i + 1 < len(self.hparams.layer_sizes)
                else self.hparams.layer_sizes[-1]
            )
            block = ResidualBlock(
                input_dim,
                output_dim,
                self.hparams.activation,
                self.hparams.norm,
                self.hparams.dropout,
            )
            self.blocks.append(block)

        self.output_layer = nn.Linear(self.hparams.layer_sizes[-1], num_classes)

    def forward(self, *data):
        """Forward pass of the ResNet model.

        Parameters
        ----------
        data : tuple
            Input tuple of tensors of num_features, cat_features, embeddings.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        if self.hparams.use_embeddings:
            x = self.embedding_layer(*data)
            B, S, D = x.shape
            x = x.reshape(B, S * D)
        else:
            x = torch.cat([t for tensors in data for t in tensors], dim=1)

        x = self.initial_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x
