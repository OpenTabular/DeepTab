from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn

from .base_model_config import BaseModelConfig


@dataclass
class TabulaRNNConfig(BaseModelConfig):
    """Architecture-only configuration for TabulaRNN models (DeepTab 2.0 API).

    Parameters
    ----------
    d_model : int, default=128
        Dimensionality of embeddings or model representations.
    activation : Callable, default=nn.SELU()
        Activation function for the RNN layers.
    model_type : str, default='RNN'
        Type of model, one of "RNN", "LSTM", "GRU", "mLSTM", "sLSTM".
    n_layers : int, default=4
        Number of layers in the RNN.
    rnn_dropout : float, default=0.2
        Dropout rate for the RNN layers.
    norm : str, default='RMSNorm'
        Normalization method to be used.
    residuals : bool, default=False
        Whether to include residual connections in the RNN.
    norm_first : bool, default=False
        Whether to apply normalization before other operations in each block.
    bias : bool, default=True
        Whether to use bias in the linear layers.
    rnn_activation : str, default='relu'
        Activation function for the RNN layers.
    dim_feedforward : int, default=256
        Size of the feedforward network.
    d_conv : int, default=4
        Size of the convolutional layer for embedding features.
    dilation : int, default=1
        Dilation factor for the convolution.
    conv_bias : bool, default=True
        Whether to use bias in the convolutional layers.
    head_layer_sizes : list, default=field(default_factory=list
        Sizes of the layers in the head of the model.
    head_dropout : float, default=0.5
        Dropout rate for the head layers.
    head_skip_layers : bool, default=False
        Whether to skip layers in the head.
    head_activation : Callable, default=nn.SELU()
        Activation function for the head layers.
    head_use_batch_norm : bool, default=False
        Whether to use batch normalization in the head layers.
    pooling_method : str, default='avg'
        Pooling method to be used ('avg', 'cls', etc.).
    """

    # Override parent defaults
    d_model: int = 128
    activation: Callable = nn.SELU()  # noqa: RUF009

    # RNN-specific architecture
    model_type: str = "RNN"
    n_layers: int = 4
    rnn_dropout: float = 0.2
    norm: str = "RMSNorm"
    residuals: bool = False
    norm_first: bool = False
    bias: bool = True
    rnn_activation: str = "relu"
    dim_feedforward: int = 256
    d_conv: int = 4
    dilation: int = 1
    conv_bias: bool = True

    # Head
    head_layer_sizes: list = field(default_factory=list)
    head_dropout: float = 0.5
    head_skip_layers: bool = False
    head_activation: Callable = nn.SELU()  # noqa: RUF009
    head_use_batch_norm: bool = False

    # Pooling
    pooling_method: str = "avg"
