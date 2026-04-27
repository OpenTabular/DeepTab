# black: noqa

import torch
import torch.nn as nn

from ..get_norm_fn import get_normalization_layer
from ..layer_utils.normalization_layers import (
    BatchNorm,
    GroupNorm,
    InstanceNorm,
    LayerNorm,
    LearnableLayerScaling,
    RMSNorm,
)
from .init_weights import _init_weights


class ResidualBlock(nn.Module):
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
        if ResidualBlock.MambaBlock is None:
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
        self.layers = ResidualBlock.MambaBlock(
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
        if ResidualBlock.MambaBlock is None:
            try:
                if mamba_version == "mamba1":
                    from mamba_ssm import Mamba as MambaBlock  # type: ignore

                    ResidualBlock.MambaBlock = MambaBlock
                    print("Successfully imported Mamba (version 1)")
                elif mamba_version == "mamba2":
                    from mamba_ssm import Mamba2 as MambaBlock  # type: ignore

                    ResidualBlock.MambaBlock = MambaBlock
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
                ResidualBlock(
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
                    ResidualBlock(
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
