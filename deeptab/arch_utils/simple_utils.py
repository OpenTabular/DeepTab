import torch
import torch.nn as nn


class MLP_Block(nn.Module):
    def __init__(self, d_in: int, d: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(d_in), nn.Linear(d_in, d), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d, d_in)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


import torch  # noqa: E402


def make_random_batches(train_size: int, batch_size: int, device=None):
    permutation = torch.randperm(train_size, device=device)
    batches = permutation.split(batch_size)

    assert torch.equal(torch.arange(train_size, device=device), permutation.sort().values)  # noqa: S101
    return batches
