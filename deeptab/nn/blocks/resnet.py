import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation, norm=False, dropout=0.0):
        """Residual Block used in ResNet.

        Parameters
        ----------
        input_dim : int
            Input dimension of the block.
        output_dim : int
            Output dimension of the block.
        activation : Callable
            Activation function.
        norm_layer : Callable, optional
            Normalization layer function, by default None.
        dropout : float, optional
            Dropout rate, by default 0.0.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.activation = activation
        self.norm1 = nn.LayerNorm(output_dim) if norm else None
        self.norm2 = nn.LayerNorm(output_dim) if norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x):
        z = self.linear1(x)
        out = z
        if self.norm1:
            out = self.norm1(out)
        out = self.activation(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.linear2(out)
        if self.norm2:
            out = self.norm2(out)
        out += z
        out = self.activation(out)
        return out
