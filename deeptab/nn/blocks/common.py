# ruff: noqa: E402
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class SNLinear(nn.Module):
    """Separate linear layers for each feature embedding."""

    def __init__(self, n: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(n, in_features, out_features))
        self.bias = Parameter(torch.empty(n, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_in_rsqrt = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("SNLinear requires a 3D input (batch, features, embedding).")
        if x.shape[-(self.weight.ndim - 1) :] != self.weight.shape[:-1]:
            raise ValueError("Input shape mismatch with weight dimensions.")

        x = x.transpose(0, 1) @ self.weight
        return x.transpose(0, 1) + self.bias


from torch.autograd import Function


def _make_ix_like(x, dim=0):
    """
    Creates a tensor of indices like the input tensor along the specified dimension.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor whose shape will be used to determine the shape of the output tensor.
    dim : int, optional
        Dimension along which to create the index tensor. Default is 0.

    Returns
    -------
    torch.Tensor
        A tensor containing indices along the specified dimension.
    """
    d = x.size(dim)
    rho = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    view = [1] * x.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


class SparsemaxFunction(Function):
    """
    Implements the sparsemax function, a sparse alternative to softmax.

    References
    ----------
    Martins, A. F., & Astudillo, R. F. (2016). "From Softmax to Sparsemax: A Sparse Model of
    Attention and Multi-Label Classification."
    """

    @staticmethod
    def forward(ctx, input_, dim=-1):
        """
        Forward pass of sparsemax: a normalizing, sparse transformation.

        Parameters
        ----------
        input_ : torch.Tensor
            The input tensor on which sparsemax will be applied.
        dim : int, optional
            Dimension along which to apply sparsemax. Default is -1.

        Returns
        -------
        torch.Tensor
            A tensor with the same shape as the input, with sparsemax applied.
        """
        ctx.dim = dim
        max_val, _ = input_.max(dim=dim, keepdim=True)
        input_ -= max_val  # Numerical stability trick, as with softmax.
        tau, supp_size = SparsemaxFunction._threshold_and_support(input_, dim=dim)
        output = torch.clamp(input_ - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        """
        Backward pass of sparsemax, calculating gradients.

        Parameters
        ----------
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output of sparsemax.

        Returns
        -------
        tuple
            Gradients of the loss with respect to the input of sparsemax and None for the dimension argument.
        """
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def _threshold_and_support(input_, dim=-1):
        """
        Computes the threshold and support for sparsemax.

        Parameters
        ----------
        input_ : torch.Tensor
            The input tensor on which to compute the threshold and support.
        dim : int, optional
            Dimension along which to compute the threshold and support. Default is -1.

        Returns
        -------
        tuple
            - torch.Tensor : The threshold value for sparsemax.
            - torch.Tensor : The support size tensor.
        """
        input_srt, _ = torch.sort(input_, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = _make_ix_like(input_, dim)
        support = rhos * input_srt > input_cumsum

        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input_.dtype)
        return tau, support_size


def sparsemax(tensor, dim=-1):
    return SparsemaxFunction.apply(tensor, dim)


def sparsemoid(tensor):
    return (0.5 * tensor + 0.5).clamp_(0, 1)


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


import torch.nn as nn


class BlockDiagonal(nn.Module):
    def __init__(self, in_features, out_features, num_blocks, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks

        if out_features % num_blocks != 0:
            raise ValueError("out_features must be divisible by num_blocks")

        block_out_features = out_features // num_blocks

        self.blocks = nn.ModuleList([nn.Linear(in_features, block_out_features, bias=bias) for _ in range(num_blocks)])

    def forward(self, x):
        x = [block(x) for block in self.blocks]
        x = torch.cat(x, dim=-1)
        return x


import torch.nn as nn


class LearnableFourierFeatures(nn.Module):
    def __init__(self, num_features=64, d_model=512):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(num_features, d_model))
        self.phases = nn.Parameter(torch.randn(num_features) * 2 * torch.pi)

    def forward(self, x):
        B, K, _D = x.shape
        positions = torch.arange(K, device=x.device).unsqueeze(1)
        encoding = torch.sin(positions * self.freqs.T + self.phases)
        return x + encoding.unsqueeze(0).expand(B, K, -1)


class LearnableFourierMask(nn.Module):
    def __init__(self, sequence_length, keep_ratio=0.5):
        super().__init__()
        cutoff_index = int(sequence_length * keep_ratio)
        self.mask = nn.Parameter(torch.ones(sequence_length))
        self.mask[cutoff_index:] = 0  # Start with a low-frequency cutoff

    def forward(self, x):
        freq_repr = torch.fft.fft(x, dim=1)
        masked_freq = freq_repr * self.mask.unsqueeze(1)  # Apply learnable mask
        return torch.fft.ifft(masked_freq, dim=1).real


class LearnableRandomPositionalPerturbation(nn.Module):
    def __init__(self, num_features=64, d_model=512):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(num_features))
        self.amplitude = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, K, D = x.shape
        positions = torch.arange(K, device=x.device).unsqueeze(1)
        random_features = torch.sin(positions * self.freqs.T)
        perturbation = random_features.unsqueeze(0).expand(B, K, D) * self.amplitude
        return x + perturbation


class LearnableRandomProjection(nn.Module):
    def __init__(self, d_model=512, projection_dim=64):
        super().__init__()
        self.projection_matrix = nn.Parameter(torch.randn(d_model, projection_dim))

    def forward(self, x):
        return torch.einsum("bkd,dp->bkp", x, self.projection_matrix)


class PositionalInvariance(nn.Module):
    def __init__(self, config, invariance_type, seq_len, in_channels=None):
        super().__init__()
        # Select the appropriate layer based on config.invariance_type
        if invariance_type == "lfm":  # Learnable Fourier Mask
            self.layer = LearnableFourierMask(sequence_length=seq_len, keep_ratio=getattr(config, "keep_ratio", 0.5))
        elif invariance_type == "lff":  # Learnable Fourier Features
            self.layer = LearnableFourierFeatures(num_features=seq_len, d_model=config.d_model)
        elif invariance_type == "lprp":  # Learnable Positional Random Perturbation
            self.layer = LearnableRandomPositionalPerturbation(num_features=seq_len, d_model=config.d_model)
        elif invariance_type == "lrp":  # Learnable Random Projection
            self.layer = LearnableRandomProjection(
                d_model=config.d_model,
                projection_dim=getattr(config, "projection_dim", 64),
            )

        elif invariance_type == "conv":
            self.layer = nn.Conv1d(
                in_channels=in_channels,  # type: ignore
                out_channels=in_channels,  # type: ignore
                kernel_size=config.d_conv,
                padding=config.d_conv - 1,
                bias=config.conv_bias,
                groups=in_channels,  # type: ignore
            )
        else:
            raise ValueError(f"Unknown positional invariance type: {config.invariance_type}")

    def forward(self, x):
        # Pass the input through the selected layer
        return self.layer(x)


import math

import torch.nn as nn


class Periodic(nn.Module):
    """Periodic transformation with learned frequency coefficients."""

    def __init__(self, n_features: int, k: int, sigma: float) -> None:
        super().__init__()
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, but got {sigma=}")

        self._sigma = sigma
        self.weight = Parameter(torch.empty(n_features, k))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = self._sigma * 3
        nn.init.trunc_normal_(self.weight, 0.0, self._sigma, a=-bound, b=bound)

    def forward(self, x):
        x = 2 * math.pi * self.weight * x[..., None]
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


class PeriodicEmbeddings(nn.Module):
    """Embeddings for continuous features using Periodic + Linear (+ ReLU) transformations.

    Supports PL, PLR, and PLR(lite) embedding types.

    Shape:
        - Input: (*, n_features)
        - Output: (*, n_features, d_embedding)
    """

    def __init__(
        self,
        n_features: int,
        d_embedding: int = 24,
        *,
        n_frequencies: int = 48,
        frequency_init_scale: float = 0.01,
        activation: bool = True,
        lite: bool = False,
    ):
        """
        Args:
            n_features (int): Number of features.
            d_embedding (int): Size of each feature embedding.
            n_frequencies (int): Number of frequencies per feature.
            frequency_init_scale (float): Initialization scale for frequency coefficients.
            activation (bool): If True, applies ReLU, making it PLR; otherwise, PL.
            lite (bool): If True, uses shared linear layer (PLR lite); otherwise, separate layers.
        """
        super().__init__()
        self.periodic = Periodic(n_features, n_frequencies, frequency_init_scale)

        # Choose linear transformation: shared or separate
        if lite:
            if not activation:
                raise ValueError("lite=True requires activation=True")
            self.linear = nn.Linear(2 * n_frequencies, d_embedding)
        else:
            self.linear = SNLinear(n_features, 2 * n_frequencies, d_embedding)

        self.activation = nn.ReLU() if activation else None

    def forward(self, x):
        """Forward pass."""
        x = self.periodic(x)
        x = self.linear(x)
        return self.activation(x) if self.activation else x


import torch.nn as nn
import torch.nn.functional as F


class NeuralEmbeddingTree(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        temperature=0.0,
    ):
        """Initialize the neural decision tree with a neural network at each leaf.

        Parameters:
        -----------
        input_dim: int
            The number of input features.
        depth: int
            The depth of the tree. The number of leaves will be 2^depth.
        output_dim: int
            The number of output classes (default is 1 for regression tasks).
        lamda: float
            Regularization parameter.
        """
        super().__init__()

        self.temperature = temperature
        self.output_dim = output_dim
        self.depth = int(math.log2(output_dim))

        # Initialize internal nodes with linear layers followed by hard thresholds
        self.inner_nodes = nn.Sequential(
            nn.Linear(input_dim + 1, output_dim, bias=False),
        )

    def forward(self, X):
        """Implementation of the forward pass with hard decision boundaries."""
        batch_size = X.size()[0]
        X = self._data_augment(X)

        # Get the decision boundaries for the internal nodes
        decision_boundaries = self.inner_nodes(X)

        # Apply hard thresholding to simulate binary decisions
        if self.temperature > 0.0:
            # Replace sigmoid with Gumbel-Softmax for path_prob calculation
            logits = decision_boundaries / self.temperature
            path_prob = (logits > 0).float() + logits.sigmoid() - logits.sigmoid().detach()
        else:
            path_prob = (decision_boundaries > 0).float()

        # Prepare for routing at the internal nodes
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)

        _mu = X.data.new(batch_size, 1, 1).fill_(1.0)

        # Iterate through internal nodes in each layer to compute the final path
        # probabilities and the regularization term.
        begin_idx = 0
        end_idx = 1

        for layer_idx in range(0, self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]

            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)

            _mu = _mu * _path_prob  # update path probabilities

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        mu = _mu.view(batch_size, self.output_dim)

        return mu

    def _data_augment(self, X):
        return F.pad(X, (1, 0), value=1)


import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures


class ScaledPolynomialLayer(nn.Module):
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree

        # Initialize polynomial feature generator
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        # Initialize learnable scaling parameter
        self.weights = nn.Parameter(torch.ones(self.degree))

    def forward(self, x):
        # Scale the input to the range [-1, 1]
        x_np = x.detach().cpu().numpy()
        scaler = MinMaxScaler(feature_range=(-1, 1))
        x_scaled = scaler.fit_transform(x_np) * 1e-05

        # Generate polynomial features
        poly_features = self.poly.fit_transform(x_scaled)

        # Convert polynomial features back to tensor
        poly_features = torch.tensor(poly_features, dtype=torch.float32).to(x.device)

        # Apply the learnable scaling parameter
        output = poly_features * self.weights

        output = torch.clamp(output, min=-1e5, max=1e3)

        return output


import torch.nn as nn


class PeriodicLinearEncodingLayer(nn.Module):
    def __init__(self, bins=10, learn_bins=True):
        super().__init__()
        self.bins = bins
        self.learn_bins = learn_bins

        if self.learn_bins:
            # Learnable bin boundaries
            self.bin_boundaries = nn.Parameter(torch.linspace(0, 1, self.bins + 1))
        else:
            self.bin_boundaries = torch.linspace(-1, 1, self.bins + 1)

    def forward(self, x):
        if self.learn_bins:
            # Ensure bin boundaries are sorted
            sorted_bins = torch.sort(self.bin_boundaries)[0]
        else:
            sorted_bins = self.bin_boundaries

        # Initialize z with zeros
        z = torch.zeros(x.size(0), self.bins, device=x.device)

        for t in range(1, self.bins + 1):
            b_t_1 = sorted_bins[t - 1]
            b_t = sorted_bins[t]
            mask1 = x < b_t_1
            mask2 = x >= b_t
            mask3 = (x >= b_t_1) & (x < b_t)

            z[mask1.squeeze(), t - 1] = 0
            z[mask2.squeeze(), t - 1] = 1
            z[mask3.squeeze(), t - 1] = (x[mask3] - b_t_1) / (b_t - b_t_1)

        return z


import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, num_feature_info, cat_feature_info, emb_feature_info, config):
        """Embedding layer that handles numerical and categorical embeddings.

        Parameters
        ----------
        num_feature_info : dict
            Dictionary where keys are numerical feature names and values are their respective input dimensions.
        cat_feature_info : dict
            Dictionary where keys are categorical feature names and values are the number of categories
            for each feature.
        config : Config
            Configuration object containing all required settings.
        """
        super().__init__()

        self.d_model = getattr(config, "d_model", 128)
        self.embedding_activation = getattr(config, "embedding_activation", nn.Identity())
        self.layer_norm_after_embedding = getattr(config, "layer_norm_after_embedding", False)
        self.embedding_projection = getattr(config, "embedding_projection", True)
        self.use_cls = getattr(config, "use_cls", False)
        self.cls_position = getattr(config, "cls_position", 0)
        self.embedding_dropout = (
            nn.Dropout(getattr(config, "embedding_dropout", 0.0))
            if getattr(config, "embedding_dropout", None) is not None
            else None
        )
        self.embedding_type = getattr(config, "embedding_type", "linear")
        self.embedding_bias = getattr(config, "embedding_bias", False)

        # Sequence length
        self.seq_len = len(num_feature_info) + len(cat_feature_info)

        # Initialize numerical embeddings based on embedding_type
        if self.embedding_type == "ndt":
            self.num_embeddings = nn.ModuleList(
                [
                    NeuralEmbeddingTree(feature_info["dimension"], self.d_model)
                    for feature_name, feature_info in num_feature_info.items()
                ]
            )
        elif self.embedding_type == "plr":
            self.num_embeddings = PeriodicEmbeddings(
                n_features=len(num_feature_info),
                d_embedding=self.d_model,
                n_frequencies=getattr(config, "n_frequencies", 48),
                frequency_init_scale=getattr(config, "frequency_init_scale", 0.01),
                activation=True,
                lite=getattr(config, "plr_lite", False),
            )
        elif self.embedding_type == "linear":
            self.num_embeddings = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(
                            feature_info["dimension"],
                            self.d_model,
                            bias=self.embedding_bias,
                        ),
                        self.embedding_activation,
                    )
                    for feature_name, feature_info in num_feature_info.items()
                ]
            )
        # for splines and other embeddings
        # splines followed by linear if n_knots actual knots is less than the defined knots
        else:
            raise ValueError("Invalid embedding_type. Choose from 'linear', 'ndt', or 'plr'.")

        self.cat_embeddings = nn.ModuleList(
            [
                (
                    nn.Sequential(
                        nn.Embedding(feature_info["categories"] + 1, self.d_model),
                        self.embedding_activation,
                    )
                    if feature_info["dimension"] == 1
                    else nn.Sequential(
                        nn.Linear(
                            feature_info["dimension"],
                            self.d_model,
                            bias=self.embedding_bias,
                        ),
                        self.embedding_activation,
                    )
                )
                for feature_name, feature_info in cat_feature_info.items()
            ]
        )

        if len(emb_feature_info) >= 1:
            if self.embedding_projection:
                self.emb_embeddings = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(
                                feature_info["dimension"],
                                self.d_model,
                                bias=self.embedding_bias,
                            ),
                            self.embedding_activation,
                        )
                        for feature_name, feature_info in emb_feature_info.items()
                    ]
                )

        # Class token if required
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        # Layer normalization if required
        if self.layer_norm_after_embedding:
            self.embedding_norm = nn.LayerNorm(self.d_model)

        self.feature_info = (num_feature_info, cat_feature_info, emb_feature_info)

    def forward(self, num_features, cat_features, emb_features):
        """Defines the forward pass of the model.

        Parameters
        ----------
        data: tuple of lists of tensors

        Returns
        -------
        Tensor
            The output embeddings of the model.

        Raises
        ------
        ValueError
            If no features are provided to the model.
        """
        num_embeddings, cat_embeddings, emb_embeddings = None, None, None

        # Class token initialization
        if self.use_cls:
            batch_size = (
                cat_features[0].size(0)  # type: ignore
                if cat_features != []
                else num_features[0].size(0)  # type: ignore
            )  # type: ignore
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Process categorical embeddings
        if self.cat_embeddings and cat_features is not None:
            cat_embeddings = [
                (emb(cat_features[i]) if emb(cat_features[i]).ndim == 3 else emb(cat_features[i]).unsqueeze(1))
                for i, emb in enumerate(self.cat_embeddings)
            ]

            cat_embeddings = torch.stack(cat_embeddings, dim=1)
            cat_embeddings = torch.squeeze(cat_embeddings, dim=2)
            if self.layer_norm_after_embedding:
                cat_embeddings = self.embedding_norm(cat_embeddings)

        # Process numerical embeddings based on embedding_type
        if self.embedding_type == "plr":
            # check pre-processing type compatibility with plr
            self.check_plr_embedding_compatibility(self.feature_info)
            # For PLR, pass all numerical features together
            if num_features is not None:
                num_features = torch.stack(num_features, dim=1).squeeze(
                    -1
                )  # Stack features along the feature dimension
                # Use the single PLR layer for all features
                num_embeddings = self.num_embeddings(num_features)
                if self.layer_norm_after_embedding:
                    num_embeddings = self.embedding_norm(num_embeddings)
        else:
            # For linear and ndt embeddings, handle each feature individually
            if self.num_embeddings and num_features is not None:
                num_embeddings = [emb(num_features[i]) for i, emb in enumerate(self.num_embeddings)]  # type: ignore
                num_embeddings = torch.stack(num_embeddings, dim=1)
                if self.layer_norm_after_embedding:
                    num_embeddings = self.embedding_norm(num_embeddings)

        if emb_features != []:
            if self.embedding_projection:
                emb_embeddings = [emb(emb_features[i]) for i, emb in enumerate(self.emb_embeddings)]
                emb_embeddings = torch.stack(emb_embeddings, dim=1)
            else:
                emb_embeddings = torch.stack(emb_features, dim=1)
            if self.layer_norm_after_embedding:
                emb_embeddings = self.embedding_norm(emb_embeddings)

        embeddings = [e for e in [cat_embeddings, num_embeddings, emb_embeddings] if e is not None]

        if embeddings:
            x = torch.cat(embeddings, dim=1) if len(embeddings) > 1 else embeddings[0]

        else:
            raise ValueError("No features provided to the model.")

        # Add class token if required
        if self.use_cls:
            if self.cls_position == 0:
                x = torch.cat([cls_tokens, x], dim=1)  # type: ignore
            elif self.cls_position == 1:
                x = torch.cat([x, cls_tokens], dim=1)  # type: ignore
            else:
                raise ValueError("Invalid cls_position value. It should be either 0 or 1.")

        # Apply dropout to embeddings if specified in config
        if self.embedding_dropout is not None:
            x = self.embedding_dropout(x)

        return x

    def check_plr_embedding_compatibility(self, feature_info: tuple):
        # List of incompatible preprocessing terms for PLR embedding
        incompatible_terms = ["ple", "one-hot", "polynomial", "splines", "sigmoid", "rbf"]

        # Iterate through each dictionary in the tuple (data)
        for sub_dict in feature_info:
            # Iterate through each feature in the current dictionary
            for feature, properties in sub_dict.items():
                preprocessing = properties.get("preprocessing", "")

                # Check for incompatible terms in the preprocessing string
                for term in incompatible_terms:
                    if term in preprocessing:
                        raise ValueError(f"PLR embedding type doesn't work with the '{term}' pre-processing method.\n")


class OneHotEncoding(nn.Module):
    def __init__(self, num_categories):
        super().__init__()
        self.num_categories = num_categories

    def forward(self, x):
        return torch.nn.functional.one_hot(x, num_classes=self.num_categories).float()


from collections.abc import Callable
from typing import Literal

import torch.nn as nn


class LinearBatchEnsembleLayer(nn.Module):
    """A configurable BatchEnsemble layer that supports optional input scaling, output scaling,
    and output bias terms as per the 'BatchEnsemble' paper.
    It provides initialization options for scaling terms to diversify ensemble members.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        ensemble_scaling_in: bool = True,
        ensemble_scaling_out: bool = True,
        ensemble_bias: bool = False,
        scaling_init: Literal["ones", "random-signs"] = "ones",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        # Base weight matrix W, shared across ensemble members
        self.W = nn.Parameter(torch.randn(out_features, in_features))

        # Optional scaling factors and shifts for each ensemble member
        self.r = nn.Parameter(torch.empty(ensemble_size, in_features)) if ensemble_scaling_in else None
        self.s = nn.Parameter(torch.empty(ensemble_size, out_features)) if ensemble_scaling_out else None
        self.bias = (
            nn.Parameter(torch.empty(out_features))
            if not ensemble_bias and out_features > 0
            else (nn.Parameter(torch.empty(ensemble_size, out_features)) if ensemble_bias else None)
        )

        # Initialize parameters
        self.reset_parameters(scaling_init)

    def reset_parameters(self, scaling_init: Literal["ones", "random-signs", "normal"]):
        # Initialize W using a uniform distribution
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

        # Initialize scaling factors r and s based on selected initialization
        scaling_init_fn = {
            "ones": nn.init.ones_,
            "random-signs": lambda x: torch.sign(torch.randn_like(x)),
            "normal": lambda x: nn.init.normal_(x, mean=0.0, std=1.0),
        }

        if self.r is not None:
            scaling_init_fn[scaling_init](self.r)
        if self.s is not None:
            scaling_init_fn[scaling_init](self.s)

        # Initialize bias
        if self.bias is not None:
            if self.bias.shape == (self.out_features,):
                nn.init.uniform_(self.bias, -0.1, 0.1)
            else:
                nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            # Shape: (B, n_ensembles, N)
            x = x.unsqueeze(1).expand(-1, self.ensemble_size, -1)
        elif x.size(1) != self.ensemble_size:
            raise ValueError(f"Input shape {x.shape} is invalid. Expected shape: (B, n_ensembles, N)")

        # Apply input scaling if enabled
        if self.r is not None:
            x = x * self.r

        # Linear transformation with W
        output = torch.einsum("bki,oi->bko", x, self.W)

        # Apply output scaling if enabled
        if self.s is not None:
            output = output * self.s

        # Add bias if enabled
        if self.bias is not None:
            output = output + self.bias

        return output


class RNNBatchEnsembleLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        ensemble_size: int,
        nonlinearity: Callable = torch.tanh,
        dropout: float = 0.0,
        ensemble_scaling_in: bool = True,
        ensemble_scaling_out: bool = True,
        ensemble_bias: bool = False,
        scaling_init: Literal["ones", "random-signs", "normal"] = "ones",
    ):
        """A batch ensemble RNN layer with optional bidirectionality and shared weights.

        Parameters
        ----------
        input_size : int
            The number of input features.
        hidden_size : int
            The number of features in the hidden state.
        ensemble_size : int
            The number of ensemble members.
        nonlinearity : Callable, default=torch.tanh
            Activation function to apply after each RNN step.
        dropout : float, default=0.0
            Dropout rate applied to the hidden state.
        ensemble_scaling_in : bool, default=True
            Whether to use input scaling for each ensemble member.
        ensemble_scaling_out : bool, default=True
            Whether to use output scaling for each ensemble member.
        ensemble_bias : bool, default=False
            Whether to use a unique bias term for each ensemble member.
        """
        super().__init__()
        self.input_size = input_size
        self.ensemble_size = ensemble_size
        self.nonlinearity = nonlinearity
        self.dropout_layer = nn.Dropout(dropout)
        self.bidirectional = False
        self.num_directions = 1
        self.hidden_size = hidden_size

        # Shared RNN weight matrices for all ensemble members
        self.W_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Ensemble-specific scaling factors and bias for each ensemble member
        self.r = nn.Parameter(torch.empty(ensemble_size, input_size)) if ensemble_scaling_in else None
        self.s = nn.Parameter(torch.empty(ensemble_size, hidden_size)) if ensemble_scaling_out else None
        self.bias = nn.Parameter(torch.zeros(ensemble_size, hidden_size)) if ensemble_bias else None

        # Initialize parameters
        self.reset_parameters(scaling_init)

    def reset_parameters(self, scaling_init: Literal["ones", "random-signs", "normal"]):
        # Initialize scaling factors r and s based on selected initialization
        scaling_init_fn = {
            "ones": nn.init.ones_,
            "random-signs": lambda x: torch.sign(torch.randn_like(x)),
            "normal": lambda x: nn.init.normal_(x, mean=0.0, std=1.0),
        }

        if self.r is not None:
            scaling_init_fn[scaling_init](self.r)
        if self.s is not None:
            scaling_init_fn[scaling_init](self.s)

        # Xavier initialization for W_ih and W_hh like a standard RNN
        nn.init.xavier_uniform_(self.W_ih)
        nn.init.xavier_uniform_(self.W_hh)

        # Initialize bias to zeros if applicable
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None) -> torch.Tensor:  # type: ignore
        """Forward pass for the BatchEnsembleRNNLayer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size).
        hidden : torch.Tensor, optional
            Hidden state tensor of shape (num_directions, ensemble_size, batch_size, hidden_size), by default None.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, ensemble_size, hidden_size * num_directions).
        """
        # Check input shape and expand if necessary
        if x.dim() == 3:  # Case: (B, L, D) - no ensembles
            batch_size, seq_len, _ = x.shape
            # Shape: (B, L, ensemble_size, D)
            x = x.unsqueeze(2).expand(-1, -1, self.ensemble_size, -1)
        elif x.dim() == 4 and x.size(2) == self.ensemble_size:  # Case: (B, L, ensemble_size, D)
            batch_size, seq_len, ensemble_size, _ = x.shape
            if ensemble_size != self.ensemble_size:
                raise ValueError(f"Input shape {x.shape} is invalid. Expected shape: (B, S, ensemble_size, N)")
        else:
            raise ValueError(f"Input shape {x.shape} is invalid. Expected shape: (B, L, D) or (B, L, ensemble_size, D)")

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(
                self.num_directions,
                self.ensemble_size,
                batch_size,
                self.hidden_size,
                device=x.device,
            )

        outputs = []

        for t in range(seq_len):
            hidden_next_directions = []

            for direction in range(self.num_directions):
                # Select forward or backward timestep `t`

                t_index = t if direction == 0 else seq_len - 1 - t
                x_t = x[:, t_index, :, :]

                # Apply input scaling if enabled
                if self.r is not None:
                    x_t = x_t * self.r

                # Input and hidden term calculations with shared weights
                input_term = torch.einsum("bki,hi->bkh", x_t, self.W_ih)
                # Access the hidden state for the current direction, reshape for matrix multiplication
                # Shape: (E, B, hidden_size)
                hidden_direction = hidden[direction]
                hidden_direction = hidden_direction.permute(1, 0, 2)  # Shape: (B, E, hidden_size)
                # Shape: (B, E, hidden_size)
                hidden_term = torch.einsum("bki,hi->bkh", hidden_direction, self.W_hh)
                hidden_next = input_term + hidden_term

                # Apply output scaling, bias, and non-linearity
                if self.s is not None:
                    hidden_next = hidden_next * self.s
                if self.bias is not None:
                    hidden_next = hidden_next + self.bias

                hidden_next = self.nonlinearity(hidden_next)
                hidden_next = hidden_next.permute(1, 0, 2)

                hidden_next_directions.append(hidden_next)

            # Stack `hidden_next_directions` along the first dimension to update `hidden` for all directions
            hidden = torch.stack(
                hidden_next_directions, dim=0
            )  # Shape: (num_directions, ensemble_size, batch_size, hidden_size)

            # Concatenate outputs for both directions along the last dimension if bidirectional
            output = torch.cat(
                [hn.permute(1, 0, 2) for hn in hidden_next_directions], dim=-1
            )  # Shape: (batch_size, ensemble_size, hidden_size * num_directions)
            outputs.append(output)

        # Apply dropout only to the final layer output if dropout is set
        if self.dropout_layer is not None:
            outputs[-1] = self.dropout_layer(outputs[-1])

        # Stack outputs for all timesteps
        outputs = torch.stack(
            outputs, dim=1
        )  # Shape: (batch_size, seq_len, ensemble_size, hidden_size * num_directions)

        return outputs, hidden  # type: ignore


class MultiHeadAttentionBatchEnsemble(nn.Module):
    """Multi-head attention module with batch ensembling.

    This module implements the multi-head attention mechanism with optional batch
    ensembling on selected projections. Batch ensembling allows for efficient ensembling
    by sharing weights across ensemble members while introducing diversity through scaling factors.

    Parameters
    ----------
    embed_dim : int
        The dimension of the embedding (input and output feature dimension).
    num_heads : int
        Number of attention heads.
    ensemble_size : int
        Number of ensemble members.
    scaling_init : {'ones', 'random-signs', 'normal'}, optional
        Initialization method for the scaling factors `r` and `s`. Default is 'ones'.
        - 'ones': Initialize scaling factors to ones.
        - 'random-signs': Initialize scaling factors to random signs (+1 or -1).
        - 'normal': Initialize scaling factors from a normal distribution (mean=0, std=1).
    batch_ensemble_projections : list of str, optional
        List of projections to which batch ensembling should be applied.
        Valid values are any combination of ['query', 'key', 'value', 'out_proj']. Default is ['query'].

    Attributes
    ----------
    embed_dim : int
        The dimension of the embedding.
    num_heads : int
        Number of attention heads.
    head_dim : int
        Dimension of each attention head (embed_dim // num_heads).
    ensemble_size : int
        Number of ensemble members.
    batch_ensemble_projections : list of str
        List of projections to which batch ensembling is applied.
    q_proj : nn.Linear
        Linear layer for projecting queries.
    k_proj : nn.Linear
        Linear layer for projecting keys.
    v_proj : nn.Linear
        Linear layer for projecting values.
    out_proj : nn.Linear
        Linear layer for projecting outputs.
    r : nn.ParameterDict
        Dictionary of input scaling factors for batch ensembling.
    s : nn.ParameterDict
        Dictionary of output scaling factors for batch ensembling.

    Methods
    -------
    reset_parameters(scaling_init)
        Initialize the parameters of the module.
    forward(query, key, value, mask=None)
        Perform the forward pass of the multi-head attention with batch ensembling.
    process_projection(x, linear_layer, proj_name)
        Process a projection with or without batch ensembling.
    batch_ensemble_linear(x, linear_layer, r, s)
        Apply a linear transformation with batch ensembling.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ensemble_size: int,
        scaling_init: Literal["ones", "random-signs", "normal"] = "ones",
        batch_ensemble_projections: list[str] = ["query"],
    ):
        super().__init__()
        # Ensure embedding dimension is divisible by the number of heads
        if embed_dim % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ensemble_size = ensemble_size
        self.batch_ensemble_projections = batch_ensemble_projections

        # Linear layers for projecting queries, keys, and values
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # Output linear layer
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Batch ensembling parameters
        self.r = nn.ParameterDict()
        self.s = nn.ParameterDict()
        # Initialize batch ensembling parameters for specified projections
        for proj_name in batch_ensemble_projections:
            if proj_name == "query":
                self.r["query"] = nn.Parameter(torch.Tensor(ensemble_size, embed_dim))
                self.s["query"] = nn.Parameter(torch.Tensor(ensemble_size, embed_dim))
            elif proj_name == "key":
                self.r["key"] = nn.Parameter(torch.Tensor(ensemble_size, embed_dim))
                self.s["key"] = nn.Parameter(torch.Tensor(ensemble_size, embed_dim))
            elif proj_name == "value":
                self.r["value"] = nn.Parameter(torch.Tensor(ensemble_size, embed_dim))
                self.s["value"] = nn.Parameter(torch.Tensor(ensemble_size, embed_dim))
            elif proj_name == "out_proj":
                self.r["out_proj"] = nn.Parameter(torch.Tensor(ensemble_size, embed_dim))
                self.s["out_proj"] = nn.Parameter(torch.Tensor(ensemble_size, embed_dim))
            else:
                raise ValueError(
                    f"Invalid projection name '{proj_name}'. Must be one of 'query', 'key', 'value', 'out_proj'."
                )

        # Initialize parameters
        self.reset_parameters(scaling_init)

    def reset_parameters(self, scaling_init: Literal["ones", "random-signs", "normal"]):
        """Initialize the parameters of the module.

        Parameters
        ----------
        scaling_init : {'ones', 'random-signs', 'normal'}
            Initialization method for the scaling factors `r` and `s`.
            - 'ones': Initialize scaling factors to ones.
            - 'random-signs': Initialize scaling factors to random signs (+1 or -1).
            - 'normal': Initialize scaling factors from a normal distribution (mean=0, std=1).

        Raises
        ------
        ValueError
            If an invalid `scaling_init` method is provided.
        """
        # Initialize weight matrices using Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.q_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.k_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))

        # Initialize biases uniformly
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(layer.bias, -bound, bound)

        # Initialize scaling factors r and s based on selected initialization
        scaling_init_fn = {
            "ones": nn.init.ones_,
            "random-signs": lambda x: torch.sign(torch.randn_like(x)),
            "normal": lambda x: nn.init.normal_(x, mean=0.0, std=1.0),
        }

        init_fn = scaling_init_fn.get(scaling_init)
        if init_fn is None:
            raise ValueError(f"Invalid scaling_init '{scaling_init}'. Must be one of 'ones', 'random-signs', 'normal'.")

        # Initialize r and s for specified projections
        for key in self.r.keys():
            init_fn(self.r[key])
        for key in self.s.keys():
            init_fn(self.s[key])

    def forward(self, query, key, value, mask=None):
        """Perform the forward pass of the multi-head attention with batch ensembling.

        Parameters
        ----------
        query : torch.Tensor
            The query tensor of shape (N, S, E, D), where:
                - N: Batch size
                - S: Sequence length
                - E: Ensemble size
                - D: Embedding dimension
        key : torch.Tensor
            The key tensor of shape (N, S, E, D).
        value : torch.Tensor
            The value tensor of shape (N, S, E, D).
        mask : torch.Tensor, optional
            An optional mask tensor that is broadcastable to shape (N, 1, 1, 1, S).
            Positions with zero in the mask will be masked out.

        Returns
        -------
        torch.Tensor
            The output tensor of shape (N, S, E, D).

        Raises
        ------
        AssertionError
            If the ensemble size `E` does not match `self.ensemble_size`.
        """

        N, S, E, _ = query.size()
        if E != self.ensemble_size:
            raise ValueError("Ensemble size mismatch.")

        # Process projections with or without batch ensembling
        Q = self.process_projection(query, self.q_proj, "query")  # Shape: (N, S, E, D)
        K = self.process_projection(key, self.k_proj, "key")  # Shape: (N, S, E, D)
        V = self.process_projection(value, self.v_proj, "value")  # Shape: (N, S, E, D)

        # Reshape for multi-head attention
        Q = Q.view(N, S, E, self.num_heads, self.head_dim).permute(0, 2, 3, 1, 4)  # (N, E, num_heads, S, head_dim)
        K = K.view(N, S, E, self.num_heads, self.head_dim).permute(0, 2, 3, 1, 4)
        V = V.view(N, S, E, self.num_heads, self.head_dim).permute(0, 2, 3, 1, 4)

        # Compute scaled dot-product attention
        # (N, E, num_heads, S, S)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # Expand mask to match attn_scores shape
            mask = mask.unsqueeze(1).unsqueeze(1)  # (N, 1, 1, 1, S)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # (N, E, num_heads, S, S)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention weights to values
        # (N, E, num_heads, S, head_dim)
        context = torch.matmul(attn_weights, V)

        # Reshape and permute back to (N, S, E, D)
        context = context.permute(0, 3, 1, 2, 4).contiguous().view(N, S, E, self.embed_dim)  # (N, S, E, D)

        # Apply output projection
        output = self.process_projection(context, self.out_proj, "out_proj")  # (N, S, E, D)

        return output

    def process_projection(self, x, linear_layer, proj_name):
        """Process a projection (query, key, value, or output) with or without batch ensembling.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (N, S, E, D_in), where:
                - N: Batch size
                - S: Sequence length
                - E: Ensemble size
                - D_in: Input feature dimension
        linear_layer : torch.nn.Linear
            The linear layer to apply.
        proj_name : str
            The name of the projection ('q_proj', 'k_proj', 'v_proj', or 'out_proj').

        Returns
        -------
        torch.Tensor
            The output tensor of shape (N, S, E, D_out).
        """
        if proj_name in self.batch_ensemble_projections:
            # Apply batch ensemble linear layer
            r = self.r[proj_name]
            s = self.s[proj_name]
            return self.batch_ensemble_linear(x, linear_layer, r, s)
        else:
            # Process normally without batch ensembling
            N, S, E, D_in = x.size()
            x = x.view(N * E, S, D_in)  # Combine batch and ensemble dimensions
            y = linear_layer(x)  # Apply linear layer
            D_out = y.size(-1)
            y = y.view(N, E, S, D_out).permute(0, 2, 1, 3)  # (N, S, E, D_out)
            return y

    def batch_ensemble_linear(self, x, linear_layer, r, s):
        """Apply a linear transformation with batch ensembling.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (N, S, E, D_in), where:
                - N: Batch size
                - S: Sequence length
                - E: Ensemble size
                - D_in: Input feature dimension
        linear_layer : torch.nn.Linear
            The linear layer with weight matrix `W` of shape (D_out, D_in).
        r : torch.Tensor
            The input scaling factors of shape (E, D_in).
        s : torch.Tensor
            The output scaling factors of shape (E, D_out).

        Returns
        -------
        torch.Tensor
            The output tensor of shape (N, S, E, D_out).
        """
        W = linear_layer.weight  # Shape: (D_out, D_in)
        b = linear_layer.bias  # Shape: (D_out)

        N, S, E, D_in = x.shape
        D_out = W.shape[0]

        # Multiply input by r
        x_r = x * r.view(1, 1, E, D_in)  # (N, S, E, D_in)

        # Reshape x_r to (N*S*E, D_in)
        x_r = x_r.view(-1, D_in)  # (N*S*E, D_in)

        # Compute x_r @ W^T + b
        y = F.linear(x_r, W, b)  # (N*S*E, D_out)

        # Reshape y back to (N, S, E, D_out)
        y = y.view(N, S, E, D_out)  # (N, S, E, D_out)

        # Multiply by s
        y = y * s.view(1, 1, E, D_out)  # (N, S, E, D_out)

        return y


import torch
import torch.nn as nn


class mLSTMblock(nn.Module):
    """MLSTM block with convolutions, gated mechanisms, and projection layers.

    Parameters
    ----------
    x_example : torch.Tensor
        Example input tensor for defining input dimensions.
    factor : float
        Factor to scale hidden size relative to input size.
    depth : int
        Depth of block diagonal layers.
    dropout : float, optional
        Dropout probability (default is 0.2).
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bidirectional=None,
        batch_first=None,
        nonlinearity=F.silu,
        dropout=0.2,
        bias=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = nonlinearity

        self.ln = nn.LayerNorm(self.input_size)

        self.left = nn.Linear(self.input_size, self.hidden_size)
        self.right = nn.Linear(self.input_size, self.hidden_size)

        self.conv = nn.Conv1d(
            in_channels=self.hidden_size,  # Hidden size for subsequent layers
            out_channels=self.hidden_size,  # Output channels
            kernel_size=3,
            padding="same",  # Padding to maintain sequence length
            bias=True,
            groups=self.hidden_size,
        )
        self.drop = nn.Dropout(dropout + 0.1)

        self.lskip = nn.Linear(self.hidden_size, self.hidden_size)

        self.wq = BlockDiagonal(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            num_blocks=num_layers,
            bias=bias,
        )
        self.wk = BlockDiagonal(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            num_blocks=num_layers,
            bias=bias,
        )
        self.wv = BlockDiagonal(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            num_blocks=num_layers,
            bias=bias,
        )
        self.dropq = nn.Dropout(dropout / 2)
        self.dropk = nn.Dropout(dropout / 2)
        self.dropv = nn.Dropout(dropout / 2)

        self.i_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.f_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_gate = nn.Linear(self.hidden_size, self.hidden_size)

        self.ln_c = nn.LayerNorm(self.hidden_size)
        self.ln_n = nn.LayerNorm(self.hidden_size)

        self.lnf = nn.LayerNorm(self.hidden_size)
        self.lno = nn.LayerNorm(self.hidden_size)
        self.lni = nn.LayerNorm(self.hidden_size)

        self.GN = nn.LayerNorm(self.hidden_size)
        self.ln_out = nn.LayerNorm(self.hidden_size)

        self.drop2 = nn.Dropout(dropout)

        self.proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.ln_proj = nn.LayerNorm(self.hidden_size)

        # Remove fixed-size initializations for dynamic state initialization
        self.ct_1 = None
        self.nt_1 = None

    def init_states(self, batch_size, seq_length, device):
        """Initialize the state tensors with the correct batch and sequence dimensions.

        Parameters
        ----------
        batch_size : int
            The batch size.
        seq_length : int
            The sequence length.
        device : torch.device
            The device to place the tensors on.
        """
        self.ct_1 = torch.zeros(batch_size, seq_length, self.hidden_size, device=device)
        self.nt_1 = torch.zeros(batch_size, seq_length, self.hidden_size, device=device)

    def forward(self, x):
        """Forward pass through mLSTM block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, sequence_length, input_size).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, sequence_length, input_size).
        """
        if x.ndim != 3:
            raise ValueError("Input tensor must have 3 dimensions (batch, sequence_length, input_size)")
        B, N, _ = x.shape
        device = x.device

        # Initialize states dynamically based on input shape
        if self.ct_1 is None or self.ct_1.shape[0] != B or self.ct_1.shape[1] != N:
            self.init_states(B, N, device)

        x = self.ln(x)  # layer norm on x

        left = self.left(x)  # part left
        # part right with just swish (silu) function
        right = self.activation(self.right(x))

        left_left = left.transpose(1, 2)
        left_left = self.activation(self.drop(self.conv(left_left).transpose(1, 2)))
        l_skip = self.lskip(left_left)

        # start mLSTM
        q = self.dropq(self.wq(left_left))
        k = self.dropk(self.wk(left_left))
        v = self.dropv(self.wv(left))

        i = torch.exp(self.lni(self.i_gate(left_left)))
        f = torch.exp(self.lnf(self.f_gate(left_left)))
        o = torch.sigmoid(self.lno(self.o_gate(left_left)))

        ct_1 = self.ct_1

        ct = f * ct_1 + i * v * k  # type: ignore[operator]
        ct = torch.mean(self.ln_c(ct), [0, 1], keepdim=True)
        self.ct_1 = ct.detach()

        nt_1 = self.nt_1
        nt = f * nt_1 + i * k  # type: ignore[operator]
        nt = torch.mean(self.ln_n(nt), [0, 1], keepdim=True)
        self.nt_1 = nt.detach()

        ht = o * ((ct * q) / torch.max(nt * q))
        # end mLSTM
        ht = ht

        left = self.drop2(self.GN(ht + l_skip))

        out = self.ln_out(left * right)
        out = self.ln_proj(self.proj(out))

        return out, None


class sLSTMblock(nn.Module):
    """SLSTM block with convolutions, gated mechanisms, and projection layers.

    Parameters
    ----------
    input_size : int
        Size of the input features.
    hidden_size : int
        Size of the hidden state.
    num_layers : int
        Depth of block diagonal layers.
    dropout : float, optional
        Dropout probability (default is 0.2).
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bidirectional=None,
        batch_first=None,
        nonlinearity=F.silu,
        dropout=0.2,
        bias=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = nonlinearity

        self.drop = nn.Dropout(dropout)

        self.i_gate = BlockDiagonal(
            in_features=self.input_size,
            out_features=self.input_size,
            num_blocks=num_layers,
            bias=bias,
        )
        self.f_gate = BlockDiagonal(
            in_features=self.input_size,
            out_features=self.input_size,
            num_blocks=num_layers,
            bias=bias,
        )
        self.o_gate = BlockDiagonal(
            in_features=self.input_size,
            out_features=self.input_size,
            num_blocks=num_layers,
            bias=bias,
        )
        self.z_gate = BlockDiagonal(
            in_features=self.input_size,
            out_features=self.input_size,
            num_blocks=num_layers,
            bias=bias,
        )

        self.ri_gate = BlockDiagonal(self.input_size, self.input_size, num_layers, bias=False)
        self.rf_gate = BlockDiagonal(self.input_size, self.input_size, num_layers, bias=False)
        self.ro_gate = BlockDiagonal(self.input_size, self.input_size, num_layers, bias=False)
        self.rz_gate = BlockDiagonal(self.input_size, self.input_size, num_layers, bias=False)

        self.ln_i = nn.LayerNorm(self.input_size)
        self.ln_f = nn.LayerNorm(self.input_size)
        self.ln_o = nn.LayerNorm(self.input_size)
        self.ln_z = nn.LayerNorm(self.input_size)

        self.GN = nn.LayerNorm(self.input_size)
        self.ln_c = nn.LayerNorm(self.input_size)
        self.ln_n = nn.LayerNorm(self.input_size)
        self.ln_h = nn.LayerNorm(self.input_size)

        self.left_linear = nn.Linear(self.input_size, int(self.input_size * (4 / 3)))
        self.right_linear = nn.Linear(self.input_size, int(self.input_size * (4 / 3)))

        self.ln_out = nn.LayerNorm(int(self.input_size * (4 / 3)))

        self.proj = nn.Linear(int(self.input_size * (4 / 3)), self.hidden_size)

        # Remove initial fixed-size states
        self.ct_1 = None
        self.nt_1 = None
        self.ht_1 = None
        self.mt_1 = None

    def init_states(self, batch_size, seq_length, device):
        """Initialize the state tensors with the correct batch and sequence dimensions.

        Parameters
        ----------
        batch_size : int
            The batch size.
        seq_length : int
            The sequence length.
        device : torch.device
            The device to place the tensors on.
        """
        self.nt_1 = torch.zeros(batch_size, seq_length, self.input_size, device=device)
        self.ct_1 = torch.zeros(batch_size, seq_length, self.input_size, device=device)
        self.ht_1 = torch.zeros(batch_size, seq_length, self.input_size, device=device)
        self.mt_1 = torch.zeros(batch_size, seq_length, self.input_size, device=device)

    def forward(self, x):
        """Forward pass through sLSTM block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, sequence_length, input_size).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, sequence_length, input_size).
        """
        B, N, _ = x.shape
        device = x.device

        # Initialize states dynamically based on input shape
        if self.ct_1 is None or self.nt_1 is None or self.nt_1.shape[0] != B or self.nt_1.shape[1] != N:
            self.init_states(B, N, device)

        x = self.activation(x)

        # Start sLSTM operations
        ht_1 = self.ht_1

        i = torch.exp(self.ln_i(self.i_gate(x) + self.ri_gate(ht_1)))
        f = torch.exp(self.ln_f(self.f_gate(x) + self.rf_gate(ht_1)))

        # Use expand_as to match the shapes of f and i for element-wise operations
        m = torch.max(
            torch.log(f) + self.mt_1.expand_as(f),  # type: ignore
            torch.log(i),  # type: ignore
        )
        i = torch.exp(torch.log(i) - m)
        f = torch.exp(torch.log(f) + self.mt_1.expand_as(f) - m)  # type: ignore
        self.mt_1 = m.detach()

        o = torch.sigmoid(self.ln_o(self.o_gate(x) + self.ro_gate(ht_1)))
        z = torch.tanh(self.ln_z(self.z_gate(x) + self.rz_gate(ht_1)))

        ct_1 = self.ct_1
        ct = f * ct_1 + i * z  # type: ignore[operator]
        ct = torch.mean(self.ln_c(ct), [0, 1], keepdim=True)
        self.ct_1 = ct.detach()

        nt_1 = self.nt_1
        nt = f * nt_1 + i  # type: ignore[operator]
        nt = torch.mean(self.ln_n(nt), [0, 1], keepdim=True)
        self.nt_1 = nt.detach()

        ht = o * (ct / nt)
        ht = torch.mean(self.ln_h(ht), [0, 1], keepdim=True)
        self.ht_1 = ht.detach()

        slstm_out = self.GN(ht)

        left = self.left_linear(slstm_out)
        right = F.gelu(self.right_linear(slstm_out))

        out = self.ln_out(left * right)
        out = self.proj(out)
        return out, None


import torch
import torch.nn as nn


class ConvRNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Configuration parameters with defaults where needed
        # 'RNN', 'LSTM', or 'GRU'
        self.model_type = getattr(config, "model_type", "RNN")
        self.input_size = getattr(config, "d_model", 128)
        self.hidden_size = getattr(config, "dim_feedforward", 128)
        self.num_layers = getattr(config, "n_layers", 4)
        self.rnn_dropout = getattr(config, "rnn_dropout", 0.0)
        self.bias = getattr(config, "bias", True)
        self.conv_bias = getattr(config, "conv_bias", True)
        self.rnn_activation = getattr(config, "rnn_activation", "relu")
        self.d_conv = getattr(config, "d_conv", 4)
        self.residuals = getattr(config, "residuals", False)
        self.dilation = getattr(config, "dilation", 1)

        # Choose RNN layer based on model_type
        rnn_layer = {
            "RNN": nn.RNN,
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
            "mLSTM": mLSTMblock,
            "sLSTM": sLSTMblock,
        }[self.model_type]

        # Convolutional layers
        self.convs = nn.ModuleList()
        self.layernorms_conv = nn.ModuleList()  # LayerNorms for Conv layers

        if self.residuals:
            self.residual_matrix = nn.ParameterList(
                [nn.Parameter(torch.randn(self.hidden_size, self.hidden_size)) for _ in range(self.num_layers)]
            )

        # First Conv1d layer uses input_size
        self.convs.append(
            nn.Conv1d(
                in_channels=self.input_size,
                out_channels=self.input_size,
                kernel_size=self.d_conv,
                padding=self.d_conv - 1,
                bias=self.conv_bias,
                groups=self.input_size,
                dilation=self.dilation,
            )
        )
        self.layernorms_conv.append(nn.LayerNorm(self.input_size))

        # Subsequent Conv1d layers use hidden_size as input
        for i in range(self.num_layers - 1):
            self.convs.append(
                nn.Conv1d(
                    in_channels=self.hidden_size,
                    out_channels=self.hidden_size,
                    kernel_size=self.d_conv,
                    padding=self.d_conv - 1,
                    bias=self.conv_bias,
                    groups=self.hidden_size,
                    dilation=self.dilation,
                )
            )
            self.layernorms_conv.append(nn.LayerNorm(self.hidden_size))

        # Initialize the RNN layers
        self.rnns = nn.ModuleList()
        self.layernorms_rnn = nn.ModuleList()  # LayerNorms for RNN layers

        for i in range(self.num_layers):
            rnn_args = {
                "input_size": self.input_size if i == 0 else self.hidden_size,
                "hidden_size": self.hidden_size,
                "num_layers": 1,
                "batch_first": True,
                "dropout": self.rnn_dropout if i < self.num_layers - 1 else 0,
                "bias": self.bias,
            }
            if self.model_type == "RNN":
                rnn_args["nonlinearity"] = self.rnn_activation
            self.rnns.append(rnn_layer(**rnn_args))
            self.layernorms_rnn.append(nn.LayerNorm(self.hidden_size))

    def forward(self, x):
        """Forward pass through Conv-RNN layers.

        Parameters
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length, input_size).

        Returns
        --------
        output : torch.Tensor
            Output tensor after passing through Conv-RNN layers.
        """
        _, L, _ = x.shape
        if self.residuals:
            residual = x

        # Loop through the RNN layers and apply 1D convolution before each
        for i in range(self.num_layers):
            # Transpose to (batch_size, input_size, seq_length) for Conv1d

            x = self.layernorms_conv[i](x)
            x = x.transpose(1, 2)

            # Apply the 1D convolution
            x = self.convs[i](x)[:, :, :L]

            # Transpose back to (batch_size, seq_length, input_size)
            x = x.transpose(1, 2)

            # Pass through the RNN layer
            x, _ = self.rnns[i](x)

            # Residual connection with learnable matrix
            if self.residuals:
                if i < self.num_layers and i > 0:
                    residual_proj = torch.matmul(residual, self.residual_matrix[i])  # type: ignore
                    x = x + residual_proj

                # Update residual for next layer
                residual = x

        return x, _


class EnsembleConvRNN(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.input_size = getattr(config, "d_model", 128)
        self.hidden_size = getattr(config, "dim_feedforward", 128)
        self.ensemble_size = getattr(config, "ensemble_size", 16)
        self.num_layers = getattr(config, "n_layers", 4)
        self.rnn_dropout = getattr(config, "rnn_dropout", 0.5)
        self.bias = getattr(config, "bias", True)
        self.conv_bias = getattr(config, "conv_bias", True)
        self.rnn_activation = getattr(config, "rnn_activation", torch.tanh)
        self.d_conv = getattr(config, "d_conv", 4)
        self.residuals = getattr(config, "residuals", False)
        self.ensemble_scaling_in = getattr(config, "ensemble_scaling_in", True)
        self.ensemble_scaling_out = getattr(config, "ensemble_scaling_out", True)
        self.ensemble_bias = getattr(config, "ensemble_bias", False)
        self.scaling_init = getattr(config, "scaling_init", "ones")
        self.model_type = getattr(config, "model_type", "full")

        # Convolutional layers
        self.convs = nn.ModuleList()
        self.layernorms_conv = nn.ModuleList()  # LayerNorms for Conv layers

        if self.residuals:
            self.residual_matrix = nn.ParameterList(
                [nn.Parameter(torch.randn(self.hidden_size, self.hidden_size)) for _ in range(self.num_layers)]
            )

        # First Conv1d layer uses input_size
        self.conv = nn.Conv1d(
            in_channels=self.input_size,
            out_channels=self.input_size,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            bias=self.conv_bias,
            groups=self.input_size,
        )

        self.layernorms_conv = nn.LayerNorm(self.input_size)

        # Initialize the RNN layers
        self.rnns = nn.ModuleList()
        self.layernorms_rnn = nn.ModuleList()  # LayerNorms for RNN layers

        self.rnns.append(
            RNNBatchEnsembleLayer(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                ensemble_size=self.ensemble_size,
                ensemble_scaling_in=self.ensemble_scaling_in,
                ensemble_scaling_out=self.ensemble_scaling_out,
                ensemble_bias=self.ensemble_bias,
                dropout=self.rnn_dropout,
                nonlinearity=self.rnn_activation,
                scaling_init="normal",
            )
        )

        for i in range(1, self.num_layers):
            if self.model_type == "mini":
                rnn = RNNBatchEnsembleLayer(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size,
                    ensemble_size=self.ensemble_size,
                    ensemble_scaling_in=False,
                    ensemble_scaling_out=False,
                    ensemble_bias=self.ensemble_bias,
                    dropout=self.rnn_dropout if i < self.num_layers - 1 else 0,
                    nonlinearity=self.rnn_activation,
                    scaling_init=self.scaling_init,  # type: ignore
                )
            else:
                rnn = RNNBatchEnsembleLayer(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size,
                    ensemble_size=self.ensemble_size,
                    ensemble_scaling_in=self.ensemble_scaling_in,
                    ensemble_scaling_out=self.ensemble_scaling_out,
                    ensemble_bias=self.ensemble_bias,
                    dropout=self.rnn_dropout if i < self.num_layers - 1 else 0,
                    nonlinearity=self.rnn_activation,
                    scaling_init=self.scaling_init,  # type: ignore
                )

            self.rnns.append(rnn)

    def forward(self, x):
        """Forward pass through Conv-RNN layers.

        Parameters
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length, input_size).

        Returns
        --------
        output : torch.Tensor
            Output tensor after passing through Conv-RNN layers.
        """
        _, L, _ = x.shape
        if self.residuals:
            residual = x

        x = self.layernorms_conv(x)
        x = x.transpose(1, 2)

        # Apply the 1D convolution
        x = self.conv(x)[:, :, :L]

        # Transpose back to (batch_size, seq_length, input_size)
        x = x.transpose(1, 2)

        # Loop through the RNN layers and apply 1D convolution before each
        for i, layer in enumerate(self.rnns):
            # Transpose to (batch_size, input_size, seq_length) for Conv1d

            # Pass through the RNN layer
            x, _ = layer(x)

            # Residual connection with learnable matrix
            if self.residuals:
                if i < self.num_layers and i > 0:
                    residual_proj = torch.matmul(residual, self.residual_matrix[i])  # type: ignore
                    x = x + residual_proj

                # Update residual for next layer
                residual = x

        return x, _
