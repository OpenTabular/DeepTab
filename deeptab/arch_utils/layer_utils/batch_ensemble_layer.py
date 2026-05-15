import math
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


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
