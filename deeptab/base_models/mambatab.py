import torch
import torch.nn as nn

from ..arch_utils.layer_utils.normalization_layers import LayerNorm
from ..arch_utils.mamba_utils.mamba_arch import Mamba
from ..arch_utils.mamba_utils.mamba_original import MambaOriginal
from ..arch_utils.mlp_utils import MLPhead
from ..configs.mambatab_config import DefaultMambaTabConfig
from ..utils.get_feature_dimensions import get_feature_dimensions
from .utils.basemodel import BaseModel


class MambaTab(BaseModel):
    """A MambaTab model for tabular data processing, integrating feature embeddings,
    normalization, and a configurable architecture for flexible deployment of Mamba-based
    feature transformation layers.

    Parameters
    ----------
    cat_feature_info : dict
        Dictionary containing information about categorical features, including their names and dimensions.
    num_feature_info : dict
        Dictionary containing information about numerical features, including their names and dimensions.
    num_classes : int, optional
        The number of output classes or target dimensions for regression, by default 1.
    config : DefaultMambaTabConfig, optional
        Configuration object with model hyperparameters such as dropout rates, hidden layer sizes, Mamba version, and
        other architectural configurations, by default DefaultMambaTabConfig().
    **kwargs : dict
        Additional keyword arguments for the BaseModel class.

    Attributes
    ----------
    cat_feature_info : dict
        Stores categorical feature information.
    num_feature_info : dict
        Stores numerical feature information.
    initial_layer : nn.Linear
        Linear layer for the initial transformation of concatenated feature embeddings.
    norm_f : LayerNorm
        Layer normalization applied after the initial transformation.
    embedding_activation : callable
        Activation function applied to the embedded features.
    axis : int
        Axis used to adjust the shape of features during transformation.
    tabular_head : MLPhead
        MLPhead layer to produce the final prediction based on transformed features.
    mamba : Mamba or MambaOriginal
        Mamba-based feature transformation layer based on the version specified in config.

    Methods
    -------
    forward(num_features, cat_features)
        Perform a forward pass through the model, including feature concatenation, initial transformation,
        Mamba processing, and prediction steps.
    """

    def __init__(
        self,
        feature_information: tuple,  # Expecting (num_feature_info, cat_feature_info, embedding_feature_info)
        num_classes=1,
        config: DefaultMambaTabConfig = DefaultMambaTabConfig(),  # noqa: B008
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["feature_information"])

        input_dim = get_feature_dimensions(*feature_information)

        self.returns_ensemble = False

        self.initial_layer = nn.Linear(input_dim, config.d_model)
        self.norm_f = LayerNorm(config.d_model)

        self.embedding_activation = self.hparams.embedding_activation

        self.axis = config.axis

        self.tabular_head = MLPhead(
            input_dim=self.hparams.d_model,
            config=config,
            output_dim=num_classes,
        )

        if config.mamba_version == "mamba-torch":
            self.mamba = Mamba(config)
        else:
            self.mamba = MambaOriginal(config)

    def forward(self, *data):
        """Forward pass of the Mambatab model

        Parameters
        ----------
        data : tuple
            Input tuple of tensors of num_features, cat_features, embeddings.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = torch.cat([t for tensors in data for t in tensors], dim=1)

        x = self.initial_layer(x)
        if self.axis == 1:
            x = x.unsqueeze(1)

        else:
            x = x.unsqueeze(0)

        x = self.norm_f(x)
        x = self.embedding_activation(x)
        if self.axis == 1:
            x = x.squeeze(1)
        else:
            x = x.squeeze(0)

        preds = self.tabular_head(x)

        return preds
