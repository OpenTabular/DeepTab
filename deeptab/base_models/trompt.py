import numpy as np
import torch
import torch.nn as nn

from ..arch_utils.get_norm_fn import get_normalization_layer
from ..arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from ..arch_utils.trompt_utils import TromptCell, TromptDecoder
from ..configs.trompt_config import DefaultTromptConfig
from .utils.basemodel import BaseModel


class Trompt(BaseModel):
    def __init__(
        self,
        feature_information: tuple,  # Expecting (num_feature_info, cat_feature_info, embedding_feature_info)
        num_classes=1,
        config: DefaultTromptConfig = DefaultTromptConfig(),  # noqa: B008
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
