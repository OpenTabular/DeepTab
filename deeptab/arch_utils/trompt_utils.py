import numpy as np
import torch
import torch.nn as nn

from .layer_utils.embedding_layer import EmbeddingLayer
from .layer_utils.importance import ImportanceGetter


class Expander(nn.Module):  # Figure 3 part 3
    def __init__(self, P):
        super().__init__()
        self.lin = nn.Linear(1, P)
        self.relu = nn.ReLU()
        self.gn = nn.GroupNorm(2, P)

    def forward(self, x):
        res = self.relu(self.lin(x.unsqueeze(-1)))

        return x.unsqueeze(1) + self.gn(torch.permute(res, (0, 3, 1, 2)))


class TromptCell(nn.Module):
    def __init__(self, feature_information, config):
        super().__init__()
        C = np.sum([len(info) for info in feature_information])
        self.enc = EmbeddingLayer(
            *feature_information,
            config=config,
        )
        self.fe = ImportanceGetter(config.P, C, config.d_model)
        self.ex = Expander(config.P)

    def forward(self, *data, O=None):  # noqa: E741
        x_res = self.ex(self.enc(*data))

        M = self.fe(O)

        return (M.unsqueeze(-1) * x_res).sum(dim=2)


class TromptDecoder(nn.Module):
    def __init__(self, d, d_out):
        super().__init__()
        self.l1 = nn.Linear(d, 1)
        self.l2 = nn.Linear(d, d)
        self.relu = nn.ReLU()
        self.laynorm1 = nn.LayerNorm(d)
        self.lf = nn.Linear(d, d_out)

    def forward(self, x):
        pw = torch.softmax(self.l1(x).squeeze(-1), dim=-1)

        xnew = (pw.unsqueeze(-1) * x).sum(dim=-2)

        return self.lf(self.laynorm1(self.relu(self.l2(xnew))))
