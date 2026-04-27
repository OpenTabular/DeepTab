import torch
import torch.nn as nn


class ImportanceGetter(nn.Module):  # Figure 3 part 1
    def __init__(self, P, C, d):
        super().__init__()
        self.colemb = nn.Parameter(torch.empty(C, d))
        self.pemb = nn.Parameter(torch.empty(P, d))
        torch.nn.init.normal_(self.colemb, std=0.01)
        torch.nn.init.normal_(self.pemb, std=0.01)
        self.C = C
        self.P = P
        self.d = d
        self.dense = nn.Linear(2 * self.d, self.d)
        self.laynorm1 = nn.LayerNorm(self.d)
        self.laynorm2 = nn.LayerNorm(self.d)

    def forward(self, O):  # noqa: E741
        eprompt = self.pemb.unsqueeze(0).repeat(O.shape[0], 1, 1)

        dense_out = self.dense(torch.cat((self.laynorm1(eprompt), O), dim=-1))

        dense_out = dense_out + eprompt + O

        ecolumn = self.laynorm2(self.colemb.unsqueeze(0).repeat(O.shape[0], 1, 1))

        return torch.softmax(dense_out @ ecolumn.transpose(1, 2), dim=-1)
