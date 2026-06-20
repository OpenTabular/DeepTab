import numpy as np
import torch
import torch.nn as nn

from deeptab.configs.experimental.trompt_config import TromptConfig
from deeptab.core import BaseModel
from deeptab.nn.blocks.common import EmbeddingLayer
from deeptab.nn.blocks.trompt import TromptCell, TromptDecoder
from deeptab.nn.normalization import get_normalization_layer


class Trompt(BaseModel):
    """Prompt-based network for tabular data.

    Trompt iterates over a number of cycles, each using a learned set of
    prompts to derive feature importances and a per-cycle representation
    through a :class:`TromptCell`, then decoding it with a
    :class:`TromptDecoder`. The per-cycle outputs are stacked and returned as
    an ensemble, which the training loop can average or supervise jointly.

    Parameters
    ----------
    feature_information : tuple
        A tuple containing feature information for numerical, categorical, and
        embedding features.
    num_classes : int, optional (default=1)
        The output dimension. ``1`` for scalar regression, the number of
        classes for classification, or the distribution parameter count for
        distributional (LSS) models.
    config : TromptConfig, optional (default=TromptConfig())
        Configuration object defining model hyperparameters.
    **kwargs : dict
        Additional arguments for the base model.

    Attributes
    ----------
    returns_ensemble : bool
        Whether the model returns an ensemble of predictions. Always ``True``.
    cells : nn.ModuleList
        One :class:`TromptCell` per cycle.
    decoder : TromptDecoder
        Decodes each cycle's representation into predictions.
    init_rec : nn.Parameter
        Learned initial prompt representation shared across rows.
    n_cycles : int
        Number of prompt cycles.
    """

    def __init__(
        self,
        feature_information: tuple,  # Expecting (num_feature_info, cat_feature_info, embedding_feature_info)
        num_classes=1,
        config: TromptConfig = TromptConfig(),  # noqa: B008
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["feature_information"])
        self.returns_ensemble = True

        # embedding layer
        self.cells = nn.ModuleList(TromptCell(feature_information, config) for _ in range(config.n_cycles))
        self.decoder = TromptDecoder(config.d_model, num_classes)
        self.init_rec = nn.Parameter(torch.empty(config.P, config.d_model))
        self.n_cycles = config.n_cycles

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
        O = self.init_rec.unsqueeze(0).repeat(data[0][0].shape[0], 1, 1)  # noqa: E741
        outputs = []

        for i in range(self.n_cycles):
            O = self.cells[i](*data, O=O)  # noqa: E741
            # print(O.shape)
            # print(self.tdown(O).shape)
            outputs.append(self.decoder(O))

        out = torch.stack(outputs, dim=1).squeeze(-1)
        # preds = out.mean(dim=1)
        return out
