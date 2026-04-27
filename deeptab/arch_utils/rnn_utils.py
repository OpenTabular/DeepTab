import torch
import torch.nn as nn

from .layer_utils.batch_ensemble_layer import RNNBatchEnsembleLayer
from .lstm_utils import mLSTMblock, sLSTMblock


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
