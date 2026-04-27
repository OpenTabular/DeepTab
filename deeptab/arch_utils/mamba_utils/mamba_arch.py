import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..get_norm_fn import get_normalization_layer
from ..layer_utils.normalization_layers import LayerNorm, LearnableLayerScaling, RMSNorm

# Heavily inspired and mostly taken from https://github.com/alxndrTL/mamba.py


class Mamba(nn.Module):
    """Mamba model composed of multiple MambaBlocks.

    Attributes:
        config (MambaConfig): Configuration object for the Mamba model.
        layers (nn.ModuleList): List of MambaBlocks constituting the model.
    """

    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                ResidualBlock(
                    d_model=getattr(config, "d_model", 128),
                    expand_factor=getattr(config, "expand_factor", 4),
                    bias=getattr(config, "bias", True),
                    d_conv=getattr(config, "d_conv", 4),
                    conv_bias=getattr(config, "conv_bias", False),
                    dropout=getattr(config, "dropout", 0.0),
                    dt_rank=getattr(config, "dt_rank", "auto"),
                    d_state=getattr(config, "d_state", 256),
                    dt_scale=getattr(config, "dt_scale", 1.0),
                    dt_init=getattr(config, "dt_init", "random"),
                    dt_max=getattr(config, "dt_max", 0.1),
                    dt_min=getattr(config, "dt_min", 1e-04),
                    dt_init_floor=getattr(config, "dt_init_floor", 1e-04),
                    norm=get_normalization_layer(config),  # type: ignore
                    activation=getattr(config, "activation", nn.SiLU()),
                    bidirectional=getattr(config, "bidirectional", False),
                    use_learnable_interaction=getattr(config, "use_learnable_interaction", False),
                    layer_norm_eps=getattr(config, "layer_norm_eps", 1e-5),
                    AD_weight_decay=getattr(config, "AD_weight_decay", True),
                    BC_layer_norm=getattr(config, "BC_layer_norm", False),
                    use_pscan=getattr(config, "use_pscan", False),
                    dilation=getattr(config, "dilation", 1),
                )
                for _ in range(getattr(config, "n_layers", 6))
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class ResidualBlock(nn.Module):
    """Residual block composed of a MambaBlock and a normalization layer.

    Parameters
    ----------
    d_model : int, optional
        Dimension of the model input, by default 32.
    expand_factor : int, optional
        Expansion factor for the model, by default 2.
    bias : bool, optional
        Whether to use bias in the MambaBlock, by default False.
    d_conv : int, optional
        Dimension of the convolution layer in the MambaBlock, by default 16.
    conv_bias : bool, optional
        Whether to use bias in the convolution layer, by default True.
    dropout : float, optional
        Dropout rate for the layers, by default 0.01.
    dt_rank : Union[str, int], optional
        Rank for dynamic time components, 'auto' or an integer, by default 'auto'.
    d_state : int, optional
        Dimension of the state vector, by default 32.
    dt_scale : float, optional
        Scale factor for dynamic time components, by default 1.0.
    dt_init : str, optional
        Initialization strategy for dynamic time components, by default 'random'.
    dt_max : float, optional
        Maximum value for dynamic time components, by default 0.1.
    dt_min : float, optional
        Minimum value for dynamic time components, by default 1e-03.
    dt_init_floor : float, optional
        Floor value for initialization of dynamic time components, by default 1e-04.
    norm : callable, optional
        Normalization layer, by default RMSNorm.
    activation : callable, optional
        Activation function used in the MambaBlock, by default `F.silu`.
    bidirectional : bool, optional
        Whether the block is bidirectional, by default False.
    use_learnable_interaction : bool, optional
        Whether to use learnable interactions, by default False.
    layer_norm_eps : float, optional
        Epsilon for layer normalization, by default 1e-05.
    AD_weight_decay : bool, optional
        Whether to apply weight decay in adaptive dynamics, by default False.
    BC_layer_norm : bool, optional
        Whether to use layer normalization for batch compatibility, by default False.
    use_pscan : bool, optional
        Whether to use PSCAN, by default False.

    Attributes
    ----------
    layers : MambaBlock
        The main MambaBlock layers for processing input.
    norm : callable
        Normalization layer applied before the MambaBlock.

    Methods
    -------
    forward(x)
        Performs a forward pass through the block and returns the output.

    Raises
    ------
    ValueError
        If the provided normalization layer is not valid.
    """

    def __init__(
        self,
        d_model=32,
        expand_factor=2,
        bias=False,
        d_conv=16,
        conv_bias=True,
        dropout=0.01,
        dt_rank="auto",
        d_state=32,
        dt_scale=1.0,
        dt_init="random",
        dt_max=0.1,
        dt_min=1e-03,
        dt_init_floor=1e-04,
        norm=RMSNorm,
        activation=F.silu,
        bidirectional=False,
        use_learnable_interaction=False,
        layer_norm_eps=1e-05,
        AD_weight_decay=False,
        BC_layer_norm=False,
        use_pscan=False,
        dilation=1,
    ):
        super().__init__()

        VALID_NORMALIZATION_LAYERS = {
            "RMSNorm": RMSNorm,
            "LayerNorm": LayerNorm,
            "LearnableLayerScaling": LearnableLayerScaling,
        }

        # Check if the provided normalization layer is valid
        if isinstance(norm, type) and norm.__name__ not in VALID_NORMALIZATION_LAYERS:
            raise ValueError(
                f"Invalid normalization layer: {norm.__name__}. "
                f"Valid options are: {', '.join(VALID_NORMALIZATION_LAYERS.keys())}"
            )
        elif isinstance(norm, str) and norm not in VALID_NORMALIZATION_LAYERS:
            raise ValueError(
                f"Invalid normalization layer: {norm}. "
                f"Valid options are: {', '.join(VALID_NORMALIZATION_LAYERS.keys())}"
            )

        if dt_rank == "auto":
            dt_rank = math.ceil(d_model / 16)

        self.layers = MambaBlock(
            d_model=d_model,
            expand_factor=expand_factor,
            bias=bias,
            d_conv=d_conv,
            conv_bias=conv_bias,
            dropout=dropout,
            dt_rank=dt_rank,  # type: ignore
            d_state=d_state,
            dt_scale=dt_scale,
            dt_init=dt_init,
            dt_max=dt_max,
            dt_min=dt_min,
            dt_init_floor=dt_init_floor,
            activation=activation,
            bidirectional=bidirectional,
            use_learnable_interaction=use_learnable_interaction,
            layer_norm_eps=layer_norm_eps,
            AD_weight_decay=AD_weight_decay,
            BC_layer_norm=BC_layer_norm,
            use_pscan=use_pscan,
            dilation=dilation,
        )
        self.norm = norm

    def forward(self, x):
        """Forward pass through the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the block.

        Returns
        -------
        torch.Tensor
            Output tensor after applying the residual connection and MambaBlock.
        """
        output = self.layers(self.norm(x)) + x
        return output


class MambaBlock(nn.Module):
    """MambaBlock module containing the main computational components for processing input.

    Parameters
    ----------
    d_model : int, optional
        Dimension of the model input, by default 32.
    expand_factor : int, optional
        Factor by which the input is expanded in the block, by default 2.
    bias : bool, optional
        Whether to use bias in the linear projections, by default False.
    d_conv : int, optional
        Dimension of the convolution layer, by default 16.
    conv_bias : bool, optional
        Whether to use bias in the convolution layer, by default True.
    dropout : float, optional
        Dropout rate applied to the layers, by default 0.01.
    dt_rank : Union[str, int], optional
        Rank for dynamic time components, either 'auto' or an integer, by default 'auto'.
    d_state : int, optional
        Dimensionality of the state vector, by default 32.
    dt_scale : float, optional
        Scale factor applied to the dynamic time component, by default 1.0.
    dt_init : str, optional
        Initialization strategy for the dynamic time component, by default 'random'.
    dt_max : float, optional
        Maximum value for dynamic time component initialization, by default 0.1.
    dt_min : float, optional
        Minimum value for dynamic time component initialization, by default 1e-03.
    dt_init_floor : float, optional
        Floor value for dynamic time component initialization, by default 1e-04.
    activation : callable, optional
        Activation function applied in the block, by default `F.silu`.
    bidirectional : bool, optional
        Whether the block is bidirectional, by default False.
    use_learnable_interaction : bool, optional
        Whether to use learnable feature interaction, by default False.
    layer_norm_eps : float, optional
        Epsilon for layer normalization, by default 1e-05.
    AD_weight_decay : bool, optional
        Whether to apply weight decay in adaptive dynamics, by default False.
    BC_layer_norm : bool, optional
        Whether to use layer normalization for batch compatibility, by default False.
    use_pscan : bool, optional
        Whether to use the PSCAN mechanism, by default False.

    Attributes
    ----------
    in_proj : nn.Linear
        Linear projection applied to the input tensor.
    conv1d : nn.Conv1d
        1D convolutional layer for processing input.
    x_proj : nn.Linear
        Linear projection applied to input-dependent tensors.
    dt_proj : nn.Linear
        Linear projection for the dynamical time component.
    A_log : nn.Parameter
        Logarithmically stored tensor A for internal dynamics.
    D : nn.Parameter
        Tensor for the D component of the model's dynamics.
    out_proj : nn.Linear
        Linear projection applied to the output.
    learnable_interaction : LearnableFeatureInteraction
        Layer for learnable feature interactions, if `use_learnable_interaction` is True.

    Methods
    -------
    forward(x)
        Performs a forward pass through the MambaBlock.
    """

    def __init__(
        self,
        d_model=32,
        expand_factor=2,
        bias=False,
        d_conv=16,
        conv_bias=True,
        dropout=0.01,
        dt_rank="auto",
        d_state=32,
        dt_scale=1.0,
        dt_init="random",
        dt_max=0.1,
        dt_min=1e-03,
        dt_init_floor=1e-04,
        activation=F.silu,
        bidirectional=False,
        use_learnable_interaction=False,
        layer_norm_eps=1e-05,
        AD_weight_decay=False,
        BC_layer_norm=False,
        use_pscan=False,
        dilation=1,
    ):
        super().__init__()

        self.use_pscan = use_pscan

        if self.use_pscan:
            try:
                from mambapy.pscan import pscan  # type: ignore

                self.pscan = pscan  # Store the imported pscan function
            except ImportError:
                self.pscan = None  # Set to None if pscan is not available
                print("The 'mambapy' package is not installed. Please install it by running:\npip install mambapy")
        else:
            self.pscan = None

        self.d_inner = d_model * expand_factor
        self.bidirectional = bidirectional
        self.use_learnable_interaction = use_learnable_interaction

        self.in_proj_fwd = nn.Linear(d_model, 2 * self.d_inner, bias=bias)
        if self.bidirectional:
            self.in_proj_bwd = nn.Linear(d_model, 2 * self.d_inner, bias=bias)

        self.conv1d_fwd = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=conv_bias,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        if self.bidirectional:
            self.conv1d_bwd = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                bias=conv_bias,
                groups=self.d_inner,
                padding=d_conv - 1,
                dilation=dilation,
            )

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        if self.use_learnable_interaction:
            self.learnable_interaction = LearnableFeatureInteraction(self.d_inner)

        self.x_proj_fwd = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)  # type: ignore
        if self.bidirectional:
            self.x_proj_bwd = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)  # type: ignore

        self.dt_proj_fwd = nn.Linear(dt_rank, self.d_inner, bias=True)  # type: ignore
        if self.bidirectional:
            self.dt_proj_bwd = nn.Linear(dt_rank, self.d_inner, bias=True)  # type: ignore

        dt_init_std = dt_rank**-0.5 * dt_scale  # type: ignore
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj_fwd.weight, dt_init_std)
            if self.bidirectional:
                nn.init.constant_(self.dt_proj_bwd.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj_fwd.weight, -dt_init_std, dt_init_std)
            if self.bidirectional:
                nn.init.uniform_(self.dt_proj_bwd.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt_fwd = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(
            min=dt_init_floor
        )
        inv_dt_fwd = dt_fwd + torch.log(-torch.expm1(-dt_fwd))
        with torch.no_grad():
            self.dt_proj_fwd.bias.copy_(inv_dt_fwd)

        if self.bidirectional:
            dt_bwd = torch.exp(
                torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt_bwd = dt_bwd + torch.log(-torch.expm1(-dt_bwd))
            with torch.no_grad():
                self.dt_proj_bwd.bias.copy_(inv_dt_bwd)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log_fwd = nn.Parameter(torch.log(A))
        self.D_fwd = nn.Parameter(torch.ones(self.d_inner))

        if self.bidirectional:
            self.A_log_bwd = nn.Parameter(torch.log(A))
            self.D_bwd = nn.Parameter(torch.ones(self.d_inner))

        if not AD_weight_decay:
            self.A_log_fwd._no_weight_decay = True  # type: ignore
            self.D_fwd._no_weight_decay = True  # type: ignore

        if self.bidirectional:
            if not AD_weight_decay:
                self.A_log_bwd._no_weight_decay = True  # type: ignore
                self.D_bwd._no_weight_decay = True  # type: ignore

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dt_rank = dt_rank
        self.d_state = d_state

        if BC_layer_norm:
            self.dt_layernorm = RMSNorm(self.dt_rank, eps=layer_norm_eps)  # type: ignore
            self.B_layernorm = RMSNorm(self.d_state, eps=layer_norm_eps)
            self.C_layernorm = RMSNorm(self.d_state, eps=layer_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

    def forward(self, x):
        _, L, _ = x.shape

        xz_fwd = self.in_proj_fwd(x)
        x_fwd, z_fwd = xz_fwd.chunk(2, dim=-1)

        x_fwd = x_fwd.transpose(1, 2)
        x_fwd = self.conv1d_fwd(x_fwd)[:, :, :L]
        x_fwd = x_fwd.transpose(1, 2)

        if self.bidirectional:
            xz_bwd = self.in_proj_bwd(x)
            x_bwd, z_bwd = xz_bwd.chunk(2, dim=-1)

            x_bwd = x_bwd.transpose(1, 2)
            x_bwd = self.conv1d_bwd(x_bwd)[:, :, :L]
            x_bwd = x_bwd.transpose(1, 2)

        if self.use_learnable_interaction:
            x_fwd = self.learnable_interaction(x_fwd)
            if self.bidirectional:
                x_bwd = self.learnable_interaction(x_bwd)  # type: ignore

        x_fwd = self.activation(x_fwd)
        x_fwd = self.dropout(x_fwd)
        y_fwd = self.ssm(x_fwd, forward=True)

        if self.bidirectional:
            x_bwd = self.activation(x_bwd)  # type: ignore
            x_bwd = self.dropout(x_bwd)
            y_bwd = self.ssm(torch.flip(x_bwd, [1]), forward=False)
            y = y_fwd + torch.flip(y_bwd, [1])
            y = y / 2
        else:
            y = y_fwd

        z_fwd = self.activation(z_fwd)
        z_fwd = self.dropout(z_fwd)

        output = y * z_fwd
        output = self.out_proj(output)

        return output

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def ssm(self, x, forward=True):
        if forward:
            A = -torch.exp(self.A_log_fwd.float())
            D = self.D_fwd.float()
            deltaBC = self.x_proj_fwd(x)
            delta, B, C = torch.split(
                deltaBC,
                [self.dt_rank, self.d_state, self.d_state],  # type: ignore
                dim=-1,
            )
            delta, B, C = self._apply_layernorms(delta, B, C)
            delta = F.softplus(self.dt_proj_fwd(delta))
        else:
            A = -torch.exp(self.A_log_bwd.float())
            D = self.D_bwd.float()
            deltaBC = self.x_proj_bwd(x)
            delta, B, C = torch.split(
                deltaBC,
                [self.dt_rank, self.d_state, self.d_state],  # type: ignore
                dim=-1,
            )
            delta, B, C = self._apply_layernorms(delta, B, C)
            delta = F.softplus(self.dt_proj_bwd(delta))

        y = self.selective_scan_seq(x, delta, A, B, C, D)
        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)

        BX = deltaB * (x.unsqueeze(-1))

        if self.use_pscan:
            hs = self.pscan(deltaA, BX)  # type: ignore
        else:
            h = torch.zeros(x.size(0), self.d_inner, self.d_state, device=deltaA.device)
            hs = []

            for t in range(0, L):
                h = deltaA[:, t] * h + BX[:, t]
                hs.append(h)

            hs = torch.stack(hs, dim=1)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)

        y = y + D * x

        return y


class LearnableFeatureInteraction(nn.Module):
    def __init__(self, n_vars):
        super().__init__()
        self.interaction_weights = nn.Parameter(torch.Tensor(n_vars, n_vars))
        nn.init.xavier_uniform_(self.interaction_weights)

    def forward(self, x):
        batch_size, n_vars, d_model = x.size()
        interactions = torch.matmul(x, self.interaction_weights)
        return interactions.view(batch_size, n_vars, d_model)
