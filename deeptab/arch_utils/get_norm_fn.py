from .layer_utils.normalization_layers import (
    BatchNorm,
    GroupNorm,
    InstanceNorm,
    LayerNorm,
    LearnableLayerScaling,
    RMSNorm,
)


def get_normalization_layer(config):
    """Function to return the appropriate normalization layer based on the configuration.

    Parameters:
    -----------
    config : BaseModelConfig
        Configuration object containing the parameters for the model including normalization.

    Returns:
    --------
    nn.Module:
        The normalization layer as per the config.

    Raises:
    -------
    ValueError:
        If an unsupported normalization layer is specified in the config.
    """

    norm_layer = getattr(config, "norm", None)
    d_model = getattr(config, "d_model", 128)
    layer_norm_eps = getattr(config, "layer_norm_eps", 1e-05)

    if norm_layer == "RMSNorm":
        return RMSNorm(d_model, eps=layer_norm_eps)
    elif norm_layer == "LayerNorm":
        return LayerNorm(d_model, eps=layer_norm_eps)
    elif norm_layer == "BatchNorm":
        return BatchNorm(d_model, eps=layer_norm_eps)
    elif norm_layer == "InstanceNorm":
        return InstanceNorm(d_model, eps=layer_norm_eps)
    elif norm_layer == "GroupNorm":
        return GroupNorm(1, d_model, eps=layer_norm_eps)
    elif norm_layer == "LearnableLayerScaling":
        return LearnableLayerScaling(d_model)
    elif norm_layer is None:
        return None
    else:
        raise ValueError(f"Unsupported normalization layer: {norm_layer}")
