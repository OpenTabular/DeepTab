import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square normalization layer.

    Attributes:
        d_model (int): The dimensionality of the input and output tensors.
        eps (float): Small value to avoid division by zero.
        weight (nn.Parameter): Learnable parameter for scaling.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


class LayerNorm(nn.Module):
    """Layer normalization layer.

    Attributes:
        d_model (int): The dimensionality of the input and output tensors.
        eps (float): Small value to avoid division by zero.
        weight (nn.Parameter): Learnable parameter for scaling.
        bias (nn.Parameter): Learnable parameter for shifting.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        output = (x - mean) / (std + self.eps)
        output = output * self.weight + self.bias
        return output


class BatchNorm(nn.Module):
    """Batch normalization layer.

    Attributes:
        d_model (int): The dimensionality of the input and output tensors.
        eps (float): Small value to avoid division by zero.
        momentum (float): The value used for the running mean and variance computation.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(d_model))
        self.register_buffer("running_var", torch.ones(d_model))
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            # Use unbiased=False for consistency with BatchNorm
            var = x.var(dim=0, unbiased=False)
            # Update running stats in-place
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)  # type: ignore[union-attr]
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)  # type: ignore[union-attr]
        else:
            mean = self.running_mean
            var = self.running_var
        output = (x - mean) / torch.sqrt(var + self.eps)  # type: ignore[operator]
        output = output * self.weight + self.bias
        return output


class InstanceNorm(nn.Module):
    """Instance normalization layer.

    Attributes:
        d_model (int): The dimensionality of the input and output tensors.
        eps (float): Small value to avoid division by zero.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True)
        output = (x - mean) / torch.sqrt(var + self.eps)
        output = output * self.weight.unsqueeze(0).unsqueeze(2) + self.bias.unsqueeze(0).unsqueeze(2)
        return output


class GroupNorm(nn.Module):
    """Group normalization layer.

    Attributes:
        num_groups (int): Number of groups to separate the channels into.
        d_model (int): The dimensionality of the input and output tensors.
        eps (float): Small value to avoid division by zero.
    """

    def __init__(self, num_groups: int, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, self.num_groups, -1)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        output = (x - mean) / torch.sqrt(var + self.eps)
        output = output.view(b, c, h, w)
        output = output * self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) + self.bias.unsqueeze(0).unsqueeze(
            2
        ).unsqueeze(3)
        return output


class LearnableLayerScaling(nn.Module):
    """Learnable Layer Scaling (LLS) normalization layer.

    Attributes:
        d_model (int): The dimensionality of the input and output tensors.
    """

    def __init__(self, d_model: int):
        """Initialize LLS normalization layer."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * self.weight.unsqueeze(0)
        return output
