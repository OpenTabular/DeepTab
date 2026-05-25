# ruff: noqa: E402
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from deeptab.nn.blocks.common import LayerNorm, LearnableLayerScaling, RMSNorm
from deeptab.nn.normalization import get_normalization_layer

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
            x_bwd, _ = xz_bwd.chunk(2, dim=-1)

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


# black: noqa

import torch.nn as nn

from deeptab.nn.blocks.common import (
    BatchNorm,
    GroupNorm,
    InstanceNorm,
    RMSNorm,
)
from deeptab.nn.initialization import _init_weights


class OriginalResidualBlock(nn.Module):
    """Residual block composed of a MambaBlock and a normalization layer.

    Attributes:
        layers (MambaBlock): MambaBlock layers.
        norm (RMSNorm): Normalization layer.
    """

    MambaBlock = None  # Declare MambaBlock at the class level

    def __init__(
        self,
        d_model=32,
        expand_factor=2,
        bias=False,
        d_conv=16,
        conv_bias=True,
        d_state=32,
        dt_max=0.1,
        dt_min=1e-03,
        dt_init_floor=1e-04,
        norm=RMSNorm,
        layer_idx=0,
        mamba_version="mamba1",
    ):
        super().__init__()

        # Lazy import for Mamba and only import if it's None
        if OriginalResidualBlock.MambaBlock is None:
            self._lazy_import_mamba(mamba_version)

        VALID_NORMALIZATION_LAYERS = {
            "RMSNorm": RMSNorm,
            "LayerNorm": LayerNorm,
            "LearnableLayerScaling": LearnableLayerScaling,
            "BatchNorm": BatchNorm,
            "InstanceNorm": InstanceNorm,
            "GroupNorm": GroupNorm,
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

        # Use the imported MambaBlock to create layers
        self.layers = OriginalResidualBlock.MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand_factor,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init_floor=dt_init_floor,
            conv_bias=conv_bias,
            bias=bias,
            layer_idx=layer_idx,
        )  # type: ignore
        self.norm = norm

    def _lazy_import_mamba(self, mamba_version):
        """Lazily import Mamba or Mamba2 based on the provided version and alias it."""
        if OriginalResidualBlock.MambaBlock is None:
            try:
                if mamba_version == "mamba1":
                    from mamba_ssm import Mamba as MambaBlock  # type: ignore

                    OriginalResidualBlock.MambaBlock = MambaBlock
                    print("Successfully imported Mamba (version 1)")
                elif mamba_version == "mamba2":
                    from mamba_ssm import Mamba2 as MambaBlock  # type: ignore

                    OriginalResidualBlock.MambaBlock = MambaBlock
                    print("Successfully imported Mamba2")
                else:
                    raise ValueError(f"Invalid mamba_version: {mamba_version}. Choose 'mamba1' or 'mamba2'.")
            except ImportError:
                raise ImportError(
                    f"Failed to import {mamba_version}. Please ensure the correct version is installed."
                ) from None

    def forward(self, x):
        output = self.layers(self.norm(x)) + x
        return output


class MambaOriginal(nn.Module):
    def __init__(self, config):
        super().__init__()

        VALID_NORMALIZATION_LAYERS = {
            "RMSNorm": RMSNorm,
            "LayerNorm": LayerNorm,
            "LearnableLayerScaling": LearnableLayerScaling,
            "BatchNorm": BatchNorm,
            "InstanceNorm": InstanceNorm,
            "GroupNorm": GroupNorm,
        }

        # Get normalization layer from config
        norm = config.norm
        self.bidirectional = config.bidirectional
        if isinstance(norm, str) and norm in VALID_NORMALIZATION_LAYERS:
            self.norm_f = VALID_NORMALIZATION_LAYERS[norm](config.d_model, eps=config.layer_norm_eps)
        else:
            raise ValueError(
                f"Invalid normalization layer: {norm}. "
                f"Valid options are: {', '.join(VALID_NORMALIZATION_LAYERS.keys())}"
            )

        # Initialize Mamba layers based on the configuration

        self.fwd_layers = nn.ModuleList(
            [
                OriginalResidualBlock(
                    mamba_version=getattr(config, "mamba_version", "mamba2"),
                    d_model=getattr(config, "d_model", 128),
                    d_state=getattr(config, "d_state", 256),
                    d_conv=getattr(config, "d_conv", 4),
                    norm=get_normalization_layer(config),  # type: ignore
                    expand_factor=getattr(config, "expand_factor", 2),
                    dt_min=getattr(config, "dt_min", 1e-04),
                    dt_max=getattr(config, "dt_max", 0.1),
                    dt_init_floor=getattr(config, "dt_init_floor", 1e-04),
                    conv_bias=getattr(config, "conv_bias", False),
                    bias=getattr(config, "bias", True),
                    layer_idx=i,
                )
                for i in range(getattr(config, "n_layers", 6))
            ]
        )

        if self.bidirectional:
            self.bckwd_layers = nn.ModuleList(
                [
                    OriginalResidualBlock(
                        mamba_version=config.mamba_version,
                        d_model=config.d_model,
                        d_state=config.d_state,
                        d_conv=config.d_conv,
                        norm=get_normalization_layer(config),  # type: ignore
                        expand_factor=config.expand_factor,
                        dt_min=config.dt_min,
                        dt_max=config.dt_max,
                        dt_init_floor=config.dt_init_floor,
                        conv_bias=config.conv_bias,
                        bias=config.bias,
                        layer_idx=i + config.n_layers,
                    )
                    for i in range(config.n_layers)
                ]
            )

        # Apply weight initialization
        self.apply(
            lambda m: _init_weights(
                m,
                n_layer=config.n_layers,
                n_residuals_per_layer=1 if config.d_state == 0 else 2,
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)  # type: ignore[arg-type]
        }

    def forward(self, x):
        if self.bidirectional:
            # Reverse input and pass through backward layers
            x_reversed = torch.flip(x, [1])
        # Forward pass through forward layers
        for layer in self.fwd_layers:
            # Update x in-place as each forward layer processes it
            x = layer(x)

        if self.bidirectional:
            for layer in self.bckwd_layers:
                x_reversed = layer(x_reversed)  # type: ignore

            # Reverse the output of the backward pass to original order
            x_reversed = torch.flip(x_reversed, [1])  # type: ignore

            # Combine forward and backward outputs by averaging
            return (x + x_reversed) / 2

        # Return forward output only if not bidirectional
        return x


import torch.nn as nn


class MambAttn(nn.Module):
    """Mamba model composed of alternating MambaBlocks and Attention layers.

    Attributes:
        config (MambaConfig): Configuration object for the Mamba model.
        layers (nn.ModuleList): List of alternating ResidualBlock (Mamba layers) and
        attention layers constituting the model.
    """

    def __init__(
        self,
        config,
    ):
        super().__init__()

        # Define Mamba and Attention layers alternation
        self.layers = nn.ModuleList()

        total_blocks = config.n_layers + config.n_attention_layers  # Total blocks to be created
        attention_count = 0

        for i in range(total_blocks):
            # Insert attention layer after N Mamba layers
            if (i + 1) % (config.n_mamba_per_attention + 1) == 0:
                self.layers.append(
                    nn.MultiheadAttention(
                        embed_dim=config.d_model,
                        num_heads=config.n_heads,
                        dropout=config.attn_dropout,
                    )
                )
                attention_count += 1
            else:
                self.layers.append(
                    ResidualBlock(
                        d_model=config.d_model,
                        expand_factor=config.expand_factor,
                        bias=config.bias,
                        d_conv=config.d_conv,
                        conv_bias=config.conv_bias,
                        dropout=config.dropout,
                        dt_rank=config.dt_rank,
                        d_state=config.d_state,
                        dt_scale=config.dt_scale,
                        dt_init=config.dt_init,
                        dt_max=config.dt_max,
                        dt_min=config.dt_min,
                        dt_init_floor=config.dt_init_floor,
                        norm=get_normalization_layer(config),  # type: ignore
                        activation=config.activation,
                        bidirectional=config.bidirectional,
                        use_learnable_interaction=config.use_learnable_interaction,
                        layer_norm_eps=config.layer_norm_eps,
                        AD_weight_decay=config.AD_weight_decay,
                        BC_layer_norm=config.BC_layer_norm,
                        use_pscan=config.use_pscan,
                    )
                )

        # Check the type of the last layer and append the desired one if necessary
        if config.last_layer == "attn":
            if not isinstance(self.layers[-1], nn.MultiheadAttention):
                self.layers.append(
                    nn.MultiheadAttention(
                        embed_dim=config.d_model,
                        num_heads=config.n_heads,
                        dropout=config.dropout,
                    )
                )
        else:
            if not isinstance(self.layers[-1], ResidualBlock):
                self.layers.append(
                    ResidualBlock(
                        d_model=config.d_model,
                        expand_factor=config.expand_factor,
                        bias=config.bias,
                        d_conv=config.d_conv,
                        conv_bias=config.conv_bias,
                        dropout=config.dropout,
                        dt_rank=config.dt_rank,
                        d_state=config.d_state,
                        dt_scale=config.dt_scale,
                        dt_init=config.dt_init,
                        dt_max=config.dt_max,
                        dt_min=config.dt_min,
                        dt_init_floor=config.dt_init_floor,
                        norm=get_normalization_layer(config),  # type: ignore
                        activation=config.activation,
                        bidirectional=config.bidirectional,
                        use_learnable_interaction=config.use_learnable_interaction,
                        layer_norm_eps=config.layer_norm_eps,
                        AD_weight_decay=config.AD_weight_decay,
                        BC_layer_norm=config.BC_layer_norm,
                        use_pscan=config.use_pscan,
                    )
                )

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.MultiheadAttention):
                # If it's an attention layer, handle input shape (seq_len, batch, embed_dim)
                # Switch to (seq_len, batch, embed_dim) for attention
                x = x.transpose(0, 1)
                x, _ = layer(x, x, x)
                # Switch back to (batch, seq_len, embed_dim)
                x = x.transpose(0, 1)
            else:
                # Otherwise, pass through Mamba block
                x = layer(x)

        return x
