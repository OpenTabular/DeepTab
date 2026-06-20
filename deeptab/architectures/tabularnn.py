from dataclasses import replace

import torch
import torch.nn as nn

from deeptab.core import BaseModel
from deeptab.nn.blocks.common import ConvRNN, EmbeddingLayer
from deeptab.nn.blocks.mlp import MLPhead
from deeptab.nn.normalization import get_normalization_layer

from ..configs.models.tabularnn_config import TabulaRNNConfig


class TabulaRNN(BaseModel):
    """Recurrent network for tabular data.

    TabulaRNN treats the embedded features of a row as a sequence and processes
    them with a convolutional RNN, combining the pooled recurrent output with a
    linear projection of the mean feature embedding before the prediction head.
    This lets the model capture interactions across the feature sequence.

    Parameters
    ----------
    feature_information : tuple
        A tuple containing feature information for numerical, categorical, and
        embedding features.
    num_classes : int, optional (default=1)
        The output dimension. ``1`` for scalar regression, the number of
        classes for classification, or the distribution parameter count for
        distributional (LSS) models.
    config : TabulaRNNConfig, optional (default=TabulaRNNConfig())
        Configuration object defining model hyperparameters.
    **kwargs : dict
        Additional arguments for the base model.

    Attributes
    ----------
    returns_ensemble : bool
        Whether the model returns an ensemble of predictions. Always ``False``.
    rnn : ConvRNN
        The convolutional recurrent block applied to the feature sequence.
    embedding_layer : EmbeddingLayer
        Embedding layer for numerical, categorical, and embedding features.
    linear : nn.Linear
        Projects the mean feature embedding into the feedforward dimension.
    norm_f : nn.Module or None
        Optional normalization layer applied before the head.
    tabular_head : MLPhead
        The final output head.
    """

    def __init__(
        self,
        feature_information: tuple,  # Expecting (num_feature_info, cat_feature_info, embedding_feature_info)
        num_classes=1,
        config: TabulaRNNConfig = TabulaRNNConfig(),  # noqa: B008
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["feature_information"])

        self.returns_ensemble = False

        self.rnn = ConvRNN(config)

        self.embedding_layer = EmbeddingLayer(
            *feature_information,
            config=config,
        )

        self.tabular_head = MLPhead(
            input_dim=self.hparams.dim_feedforward,
            config=config,
            output_dim=num_classes,
        )

        self.linear = nn.Linear(
            self.hparams.d_model,
            self.hparams.dim_feedforward,
        )

        temp_config = replace(config, d_model=config.dim_feedforward)
        self.norm_f = get_normalization_layer(temp_config)

        # pooling
        n_inputs = [len(info) for info in feature_information]
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
        # RNN forward pass
        out, _ = self.rnn(x)
        z = self.linear(torch.mean(x, dim=1))

        x = self.pool_sequence(out)
        x = x + z
        if self.norm_f is not None:
            x = self.norm_f(x)
        preds = self.tabular_head(x)

        return preds
