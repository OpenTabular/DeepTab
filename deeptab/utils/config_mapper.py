import torch.nn as nn
from skopt.space import Categorical, Integer, Real

from ..arch_utils.transformer_utils import ReGLU


def round_to_nearest_16(x):
    """Rounds the value to the nearest multiple of 16."""
    return int(round(x / 16) * 16)


def get_search_space(
    config,
    fixed_params={
        "pooling_method": "avg",
        "head_skip_layers": False,
        "head_layer_size_length": 0,
        "cat_encoding": "int",
        "head_skip_layer": False,
        "use_cls": False,
    },
    custom_search_space=None,
):
    """Given a model configuration, return the hyperparameter search space based on the config attributes.

    Parameters
    ----------
    config : dataclass
        The configuration object for the model.
    fixed_params : dict, optional
        Dictionary of fixed parameters and their values. Defaults to
        {"pooling_method": "avg", "head_skip_layers": False, "head_layer_size_length": 0}.
    custom_search_space : dict, optional
        Dictionary defining custom search spaces for parameters.
        Overrides the default `search_space_mapping` for the specified parameters.

    Returns
    -------
    param_names : list
        A list of parameter names to be optimized.
    param_space : list
        A list of hyperparameter ranges for Bayesian optimization.
    """

    # Handle the custom search space
    if custom_search_space is None:
        custom_search_space = {}

    # Base search space mapping
    search_space_mapping = {
        # Learning rate-related parameters
        "lr": Real(1e-6, 1e-2, prior="log-uniform"),
        "lr_patience": Integer(5, 20),
        "lr_factor": Real(0.1, 0.5),
        # Model architecture parameters
        "n_layers": Integer(1, 8),
        "d_model": Categorical([32, 64, 128, 256, 512, 1024]),
        "dropout": Real(0.0, 0.5),
        "expand_factor": Integer(1, 4),
        "d_state": Categorical([32, 64, 128, 256]),
        "ff_dropout": Real(0.0, 0.5),
        "rnn_dropout": Real(0.0, 0.5),
        "attn_dropout": Real(0.0, 0.5),
        "n_heads": Categorical([2, 4, 8]),
        "transformer_dim_feedforward": Integer(16, 512),
        # Convolution-related parameters
        "conv_bias": Categorical([True, False]),
        # Normalization and regularization
        "norm": Categorical(["LayerNorm", "RMSNorm"]),
        "weight_decay": Real(1e-8, 1e-2, prior="log-uniform"),
        "layer_norm_eps": Real(1e-7, 1e-4),
        "head_dropout": Real(0.0, 0.5),
        "bias": Categorical([True, False]),
        "norm_first": Categorical([True, False]),
        # Pooling, activation, and head layer settings
        "pooling_method": Categorical(["avg", "max", "cls", "sum"]),
        "activation": Categorical(["ReLU", "SELU", "Identity", "Tanh", "LeakyReLU", "SiLU"]),
        "embedding_activation": Categorical(["ReLU", "SELU", "Identity", "Tanh", "LeakyReLU"]),
        "rnn_activation": Categorical(["relu", "tanh"]),
        "transformer_activation": Categorical(["ReLU", "SELU", "Identity", "Tanh", "LeakyReLU", "ReGLU"]),
        "head_skip_layers": Categorical([True, False]),
        "head_use_batch_norm": Categorical([True, False]),
        # Sequence-related settings
        "bidirectional": Categorical([True, False]),
        "use_learnable_interaction": Categorical([True, False]),
        "use_cls": Categorical([True, False]),
        # Feature encoding
        "cat_encoding": Categorical(["int", "one-hot"]),
    }

    # Apply custom search space overrides
    search_space_mapping.update(custom_search_space)

    param_names = []
    param_space = []

    # Iterate through config fields
    for field in config.__dataclass_fields__:
        if field in fixed_params:
            # Fix the parameter value directly in the config
            setattr(config, field, fixed_params[field])
            continue  # Skip optimization for this parameter

        if field in search_space_mapping:
            # Add to search space if not fixed
            param_names.append(field)
            param_space.append(search_space_mapping[field])

    # Handle dynamic head_layer_sizes based on head_layer_size_length
    if "head_layer_sizes" in config.__dataclass_fields__:
        head_layer_size_length = fixed_params.get("head_layer_size_length", 0)

        # If no layers are desired, set head_layer_sizes to []
        if head_layer_size_length == 0:
            config.head_layer_sizes = []
        else:
            # Optimize the number of head layers
            max_head_layers = 5
            param_names.append("head_layer_size_length")
            param_space.append(Integer(1, max_head_layers))

            # Optimize individual layer sizes
            layer_size_min, layer_size_max = 16, 512
            for i in range(max_head_layers):
                layer_key = f"head_layer_size_{i + 1}"
                param_names.append(layer_key)
                param_space.append(Integer(layer_size_min, layer_size_max))

    return param_names, param_space


activation_mapper = {
    "ReLU": nn.ReLU(),
    "Tanh": nn.Tanh(),
    "SiLU": nn.SiLU(),
    "LeakyReLU": nn.LeakyReLU(),
    "Identity": nn.Identity(),
    "Linear": nn.Identity(),
    "SELU": nn.SELU(),
    "ReGLU": ReGLU(),
}
