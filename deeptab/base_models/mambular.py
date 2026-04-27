import numpy as np
import torch

from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..arch_utils.mamba_utils.mamba_arch import Mamba
from ..arch_utils.mamba_utils.mamba_original import MambaOriginal
from ..arch_utils.mlp_utils import MLPhead
from ..configs.mambular_config import DefaultMambularConfig
from .utils.basemodel import BaseModel


class Mambular(BaseModel):
    """A Mambular model for tabular data, integrating feature embeddings, Mamba transformations, and a configurable
    architecture for processing categorical and numerical features with pooling and normalization.

    Parameters
    ----------
    cat_feature_info : dict
        Dictionary containing information about categorical features, including their names and dimensions.
    num_feature_info : dict
        Dictionary containing information about numerical features, including their names and dimensions.
    num_classes : int, optional
        The number of output classes or target dimensions for regression, by default 1.
    config : DefaultMambularConfig, optional
        Configuration object with model hyperparameters such as dropout rates, head layer sizes, Mamba version, and
        other architectural configurations, by default DefaultMambularConfig().
    **kwargs : dict
        Additional keyword arguments for the BaseModel class.

    Attributes
    ----------
    pooling_method : str
        Pooling method to aggregate features after the Mamba layer.
    shuffle_embeddings : bool
        Flag indicating if embeddings should be shuffled, as specified in the configuration.
    embedding_layer : EmbeddingLayer
        Layer for embedding categorical and numerical features.
    mamba : Mamba or MambaOriginal
        Mamba-based transformation layer based on the version specified in config.
    norm_f : nn.Module
        Normalization layer for the processed features.
    tabular_head : MLP
        MLP layer to produce the final prediction based on the output of the Mamba layer.
    perm : torch.Tensor, optional
        Permutation tensor used for shuffling embeddings, if enabled.

    Methods
    -------
    forward(num_features, cat_features)
        Perform a forward pass through the model, including embedding, Mamba transformation, pooling,
        and prediction steps.
    """

    def __init__(
        self,
        feature_information: tuple,  # Expecting (cat_feature_info, num_feature_info, embedding_feature_info)
        num_classes=1,
        config: DefaultMambularConfig = DefaultMambularConfig(),  # noqa: B008
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["feature_information"])

        self.returns_ensemble = False

        # embedding layer
        self.embedding_layer = EmbeddingLayer(
            *feature_information,
            config=config,
        )

        if config.mamba_version == "mamba-torch":
            self.mamba = Mamba(config)
        else:
            self.mamba = MambaOriginal(config)

        self.tabular_head = MLPhead(
            input_dim=self.hparams.d_model,
            config=config,
            output_dim=num_classes,
        )

        if self.hparams.shuffle_embeddings:
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
        Tensor
            The output predictions of the model.
        """
        x = self.embedding_layer(*data)

        if self.hparams.shuffle_embeddings:
            x = x[:, self.perm, :]

        x = self.mamba(x)

        x = self.pool_sequence(x)

        preds = self.tabular_head(x)

        return preds
