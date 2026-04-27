from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn

from .base_config import BaseConfig


@dataclass
class DefaultTabulaRNNConfig(BaseConfig):
    """Configuration class for the TabulaRNN model with predefined hyperparameters.

    Parameters
    ----------
    model_type : str, default="RNN"
        Type of model, one of "RNN", "LSTM", "GRU", "mLSTM", "sLSTM".
    n_layers : int, default=4
        Number of layers in the RNN.
    rnn_dropout : float, default=0.2
        Dropout rate for the RNN layers.
    d_model : int, default=128
        Dimensionality of embeddings or model representations.
    norm : str, default="RMSNorm"
        Normalization method to be used.
    activation : callable, default=nn.SELU()
        Activation function for the RNN layers.
    residuals : bool, default=False
        Whether to include residual connections in the RNN.
    head_layer_sizes : list, default=()
        Sizes of the layers in the head of the model.
    head_dropout : float, default=0.5
        Dropout rate for the head layers.
    head_skip_layers : bool, default=False
        Whether to skip layers in the head.
    head_activation : callable, default=nn.SELU()
        Activation function for the head layers.
    head_use_batch_norm : bool, default=False
        Whether to use batch normalization in the head layers.
    pooling_method : str, default="avg"
        Pooling method to be used ('avg', 'cls', etc.).
    norm_first : bool, default=False
        Whether to apply normalization before other operations in each block.
    layer_norm_eps : float, default=1e-05
        Epsilon value for layer normalization.
    bias : bool, default=True
        Whether to use bias in the linear layers.
    rnn_activation : str, default="relu"
        Activation function for the RNN layers.
    dim_feedforward : int, default=256
        Size of the feedforward network.
    d_conv : int, default=4
        Size of the convolutional layer for embedding features.
    dilation : int, default=1
        Dilation factor for the convolution.
    conv_bias : bool, default=True
        Whether to use bias in the convolutional layers.
    """

    # Architecture params
    model_type: str = "RNN"
    d_model: int = 128
    n_layers: int = 4
    rnn_dropout: float = 0.2
    norm: str = "RMSNorm"
    activation: Callable = nn.SELU()  # noqa: RUF009
    residuals: bool = False

    # Head params
    head_layer_sizes: list = field(default_factory=list)
    head_dropout: float = 0.5
    head_skip_layers: bool = False
    head_activation: Callable = nn.SELU()  # noqa: RUF009
    head_use_batch_norm: bool = False

    # Pooling and normalization
    pooling_method: str = "avg"
    norm_first: bool = False
    layer_norm_eps: float = 1e-05

    # Additional params
    bias: bool = True
    rnn_activation: str = "relu"
    dim_feedforward: int = 256
    d_conv: int = 4
    dilation: int = 1
    conv_bias: bool = True
