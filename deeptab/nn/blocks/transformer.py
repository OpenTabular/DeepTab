# ruff: noqa: E402
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from deeptab.nn.blocks.common import LinearBatchEnsembleLayer, MultiHeadAttentionBatchEnsemble


def reglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


class ReGLU(nn.Module):
    def forward(self, x):
        return reglu(x)


class GLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x.size(-1) % 2 != 0:
            raise ValueError("Input dimension must be even")
        split_dim = x.size(-1) // 2
        return x[..., :split_dim] * torch.sigmoid(x[..., split_dim:])


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, config):
        super().__init__(
            d_model=getattr(config, "d_model", 128),
            nhead=getattr(config, "n_heads", 8),
            dim_feedforward=getattr(config, "transformer_dim_feedforward", 2048),
            dropout=getattr(config, "attn_dropout", 0.1),
            activation=getattr(config, "transformer_activation", F.relu),
            layer_norm_eps=getattr(config, "layer_norm_eps", 1e-5),
            norm_first=getattr(config, "norm_first", False),
        )
        self.bias = getattr(config, "bias", True)
        self.custom_activation = getattr(config, "transformer_activation", F.relu)

        # Additional setup based on the activation function
        if self.custom_activation in [ReGLU, GLU] or isinstance(self.custom_activation, ReGLU | GLU):
            self.linear1 = nn.Linear(
                self.linear1.in_features,
                self.linear1.out_features * 2,
                bias=self.bias,
            )
            self.linear2 = nn.Linear(
                self.linear2.in_features,
                self.linear2.out_features,
                bias=self.bias,
            )

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Use the provided activation function
        if self.custom_activation in [ReGLU, GLU] or isinstance(self.custom_activation, ReGLU | GLU):
            src2 = self.linear2(self.custom_activation(self.linear1(src)))
        else:
            src2 = self.linear2(self.custom_activation(self.linear1(src)))

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class BatchEnsembleTransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer with Batch Ensembling.

    This class implements a single layer of the Transformer encoder with batch ensembling applied to the
    multi-head attention and feedforward network as desired.

    Parameters
    ----------
    embed_dim : int
        The dimension of the embedding.
    num_heads : int
        Number of attention heads.
    ensemble_size : int
        Number of ensemble members.
    dim_feedforward : int, optional
        Dimension of the feedforward network model. Default is 2048.
    dropout : float, optional
        Dropout value. Default is 0.1.
    activation : {'relu', 'gelu'}, optional
        Activation function of the intermediate layer. Default is 'relu'.
    scaling_init : {'ones', 'random-signs', 'normal'}, optional
        Initialization method for the scaling factors in batch ensembling. Default is 'ones'.
    batch_ensemble_projections : list of str, optional
        List of projections to which batch ensembling should be applied in the attention layer.
        Default is ['query'].
    batch_ensemble_ffn : bool, optional
        Whether to apply batch ensembling to the feedforward network. Default is False.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ensemble_size: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu"] = "relu",
        scaling_init: Literal["ones", "random-signs", "normal"] = "ones",
        batch_ensemble_projections: list[str] = ["query"],
        batch_ensemble_ffn: bool = False,
        ensemble_bias=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ensemble_size = ensemble_size
        self.dim_feedforward = dim_feedforward
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.batch_ensemble_ffn = batch_ensemble_ffn

        # Multi-head attention with batch ensembling
        self.self_attn = MultiHeadAttentionBatchEnsemble(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ensemble_size=ensemble_size,
            scaling_init=scaling_init,
            batch_ensemble_projections=batch_ensemble_projections,
        )

        # Feedforward network
        if batch_ensemble_ffn:
            # Apply batch ensembling to the feedforward network
            self.linear1 = LinearBatchEnsembleLayer(
                embed_dim,
                dim_feedforward,
                ensemble_size,
                scaling_init=scaling_init,  # type: ignore
                ensemble_bias=ensemble_bias,
            )
            self.linear2 = LinearBatchEnsembleLayer(
                dim_feedforward,
                embed_dim,
                ensemble_size,
                scaling_init=scaling_init,  # type: ignore
                ensemble_bias=ensemble_bias,
            )
        else:
            # Standard feedforward network
            self.linear1 = nn.Linear(embed_dim, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation function
        if activation == "relu":
            self.activation_fn = F.relu
        elif activation == "gelu":
            self.activation_fn = F.gelu
        else:
            raise ValueError(f"Invalid activation '{activation}'. Choose from 'relu' or 'gelu'.")

    def forward(self, src, src_mask: torch.Tensor = None):  # type: ignore
        """Pass the input through the encoder layer.

        Parameters
        ----------
        src : torch.Tensor
            The input tensor of shape (N, S, E, D), where:
                - N: Batch size
                - S: Sequence length
                - E: Ensemble size
                - D: Embedding dimension
        src_mask : torch.Tensor, optional
            The source mask tensor.

        Returns
        -------
        torch.Tensor
            The output tensor of shape (N, S, E, D).
        """
        # Self-attention
        src2 = self.self_attn(src, src, src, mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward network
        if self.batch_ensemble_ffn:
            src2 = self.linear2(self.dropout(self.activation_fn(self.linear1(src))))
        else:
            N, S, E, D = src.shape
            src_reshaped = src.view(N * E * S, D)
            src2 = self.linear1(src_reshaped)
            src2 = self.activation_fn(src2)
            src2 = self.dropout(src2)
            src2 = self.linear2(src2)
            src2 = src2.view(N, S, E, D)

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class BatchEnsembleTransformerEncoder(nn.Module):
    """Transformer Encoder with Batch Ensembling.

    This class implements the Transformer encoder consisting of multiple encoder layers with batch ensembling.

    Parameters
    ----------
    num_layers : int
        Number of encoder layers to stack.
    embed_dim : int
        The dimension of the embedding.
    num_heads : int
        Number of attention heads.
    ensemble_size : int
        Number of ensemble members.
    dim_feedforward : int, optional
        Dimension of the feedforward network model. Default is 2048.
    dropout : float, optional
        Dropout value. Default is 0.1.
    activation : {'relu', 'gelu'}, optional
        Activation function of the intermediate layer. Default is 'relu'.
    scaling_init : {'ones', 'random-signs', 'normal'}, optional
        Initialization method for the scaling factors in batch ensembling. Default is 'ones'.
    batch_ensemble_projections : list of str, optional
        List of projections to which batch ensembling should be applied in the attention layer.
        Default is ['query'].
    batch_ensemble_ffn : bool, optional
        Whether to apply batch ensembling to the feedforward network. Default is False.
    norm : nn.Module, optional
        Optional layer normalization module.
    """

    def __init__(
        self,
        config,
    ):
        super().__init__()
        d_model = getattr(config, "d_model", 128)
        nhead = getattr(config, "n_heads", 8)
        dim_feedforward = getattr(config, "transformer_dim_feedforward", 256)
        dropout = getattr(config, "attn_dropout", 0.5)
        activation = getattr(config, "transformer_activation", F.relu)
        num_layers = getattr(config, "n_layers", 4)
        ff_dropout = getattr(config, "ff_dropout", 0.5)
        ensemble_projections = getattr(config, "batch_ensemble_projections", ["query"])
        scaling_init = getattr(config, "scaling_init", "ones")
        batch_ensemble_ffn = getattr(config, "batch_ensemble_ffn", False)
        ensemble_bias = getattr(config, "ensemble_bias", False)
        model_type = getattr(config, "model_type", "full")
        scaling_init = getattr(config, "scaling_init", "ones")

        self.ensemble_size = getattr(config, "ensemble_size", 32)

        self.layers = nn.ModuleList()

        self.layers.append(
            BatchEnsembleTransformerEncoderLayer(
                embed_dim=d_model,
                num_heads=nhead,
                ensemble_size=self.ensemble_size,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,  # type: ignore
                batch_ensemble_projections=ensemble_projections,
                batch_ensemble_ffn=batch_ensemble_ffn,
                scaling_init="normal",
                ensemble_bias=ensemble_bias,
            )
        )

        for i in range(1, num_layers):
            if model_type == "mini":
                self.layers.append(
                    BatchEnsembleTransformerEncoderLayer(
                        embed_dim=d_model,
                        num_heads=nhead,
                        ensemble_size=self.ensemble_size,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        activation=activation,  # type: ignore
                        scaling_init=scaling_init,  # type: ignore
                        batch_ensemble_projections=[],
                        batch_ensemble_ffn=False,
                        ensemble_bias=ensemble_bias,
                    )
                )

            else:
                self.layers.append(
                    BatchEnsembleTransformerEncoderLayer(
                        embed_dim=d_model,
                        num_heads=nhead,
                        ensemble_size=self.ensemble_size,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        activation=activation,  # type: ignore
                        batch_ensemble_projections=ensemble_projections,
                        batch_ensemble_ffn=batch_ensemble_ffn,
                        ensemble_bias=ensemble_bias,
                    )
                )

        self.ensemble_projections = ensemble_projections

    def forward(self, x, mask: torch.Tensor = None):  # type: ignore
        """Pass the input through the encoder layers in turn.

        Parameters
        ----------
        src : torch.Tensor
            The input tensor of shape (N, S, E, D).
        mask : torch.Tensor, optional
            The source mask tensor.

        Returns
        -------
        torch.Tensor
            The output tensor of shape (N, S, E, D).
        """
        if x.dim() == 3:  # Case: (B, L, D) - no ensembles
            # Shape: (B, L, ensemble_size, D)
            x = x.unsqueeze(2).expand(-1, -1, self.ensemble_size, -1)
        elif x.dim() == 4 and x.size(2) == self.ensemble_size:  # Case: (B, L, ensemble_size, D)
            _, _, ensemble_size, _ = x.shape
            if ensemble_size != self.ensemble_size:
                raise ValueError(f"Input shape {x.shape} is invalid. Expected shape: (B, S, ensemble_size, N)")
        else:
            raise ValueError(f"Input shape {x.shape} is invalid. Expected shape: (B, L, D) or (B, L, ensemble_size, D)")
        output = x

        for layer in self.layers:
            output = layer(output, src_mask=mask)

        return output


class RowColTransformer(nn.Module):
    def __init__(self, n_features, config):
        """RowColTransformer initialized with a configuration object.

        Args:
        - config: A configuration object containing all hyperparameters.
          Expected attributes:
            - d_model: Embedding dimension.
            - n_features: Number of features.
            - n_layers: Number of transformer layers.
            - n_heads: Number of attention heads.
            - dim_head: Dimension per head.
            - attn_dropout: Dropout rate for attention layers.
            - ff_dropout: Dropout rate for feedforward layers.
            - style: Transformer style ('col' or 'colrow').
        """

        super().__init__()
        d_model = getattr(config, "d_model", 128)
        n_layers = getattr(config, "n_layers", 6)
        n_heads = getattr(config, "n_heads", 8)
        attn_dropout = getattr(config, "attn_dropout", 0.1)
        ff_dropout = getattr(config, "ff_dropout", 0.1)
        activation = getattr(config, "activation", nn.GELU())

        self.layers = nn.ModuleList([])

        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.LayerNorm(d_model),
                            nn.MultiheadAttention(
                                embed_dim=d_model,
                                num_heads=n_heads,
                                dropout=attn_dropout,
                                batch_first=True,
                            ),
                            nn.Dropout(ff_dropout),
                        ),
                        nn.Sequential(
                            nn.LayerNorm(d_model),
                            nn.Sequential(
                                nn.Linear(d_model, d_model * 4),
                                activation,
                                nn.Dropout(ff_dropout),
                                nn.Linear(d_model * 4, d_model),
                            ),
                        ),
                        nn.Sequential(
                            nn.LayerNorm(d_model * n_features),
                            nn.MultiheadAttention(
                                embed_dim=d_model * n_features,
                                num_heads=n_heads,
                                dropout=attn_dropout,
                                batch_first=True,
                            ),
                            nn.Dropout(ff_dropout),
                        ),
                        nn.Sequential(
                            nn.LayerNorm(d_model * n_features),
                            nn.Sequential(
                                nn.Linear(d_model * n_features, d_model * n_features * 4),
                                activation,
                                nn.Dropout(ff_dropout),
                                nn.Linear(d_model * n_features * 4, d_model * n_features),
                            ),
                        ),
                    ]
                )
            )

    def forward(self, x):
        """
        Args:
            x: Input embeddings of shape (N, J, D),
               where N = batch size, J = number of features, D = embedding dimension.
        """
        _, n, _ = x.shape

        for attn1, ff1, attn2, ff2 in self.layers:  # type: ignore
            # Column-wise attention
            x = attn1[1](x, x, x)[0] + x  # Multihead attention with residual
            x = ff1(x) + x  # Feedforward with residual

            # Row-wise attention
            x = rearrange(x, "b n d -> 1 b (n d)")
            x = attn2[1](x, x, x)[0] + x  # Multihead attention with residual
            x = ff2(x) + x  # Feedforward with residual
            x = rearrange(x, "1 b (n d) -> b n d", n=n)

        return x


import numpy as np
import torch
import torch.nn as nn


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


def FeedForward(dim, mult=4, dropout=0.0):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim),
    )


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        dim = np.int64(dim / 2)

    def forward(self, x):
        h = self.heads
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in (q, k, v))  # type: ignore
        q = q * self.scale

        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)

        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)

        out = torch.einsum("b h i j, b h j d -> b h i d", dropped_attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out)

        return out, attn


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=attn_dropout,
                        ),
                        FeedForward(dim, dropout=ff_dropout),
                    ]
                )
            )

    def forward(self, x, return_attn=False):
        post_softmax_attns = []

        for attn, ff in self.layers:  # type: ignore
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = attn_out + x
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)


import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, j, dim, method="linear"):
        super().__init__()
        self.j = j
        self.dim = dim
        self.method = method

        if self.method == "linear":
            # Use nn.Linear approach
            self.layer = nn.Linear(dim, j * dim)
        elif self.method == "embedding":
            # Use nn.Embedding approach
            self.layer = nn.Embedding(dim, j * dim)
        elif self.method == "conv1d":
            # Use nn.Conv1d approach
            self.layer = nn.Conv1d(in_channels=dim, out_channels=j * dim, kernel_size=1)
        else:
            raise ValueError(f"Unsupported method '{method}' for reshaping.")

    def forward(self, x):
        batch_size = x.shape[0]

        if self.method == "linear" or self.method == "embedding":
            x_reshaped = self.layer(x)  # shape: (batch_size, j * dim)
            x_reshaped = x_reshaped.view(batch_size, self.j, self.dim)  # shape: (batch_size, j, dim)
        elif self.method == "conv1d":
            # For Conv1d, add dummy dimension and reshape
            x = x.unsqueeze(-1)  # Add dummy dimension for convolution
            x_reshaped = self.layer(x)  # shape: (batch_size, j * dim, 1)
            x_reshaped = x_reshaped.squeeze(-1)  # Remove dummy dimension
            x_reshaped = x_reshaped.view(batch_size, self.j, self.dim)  # shape: (batch_size, j, dim)

        return x_reshaped  # type: ignore


class AttentionNetBlock(nn.Module):
    def __init__(
        self,
        channels,
        in_channels,
        d_model,
        n_heads,
        n_layers,
        dim_feedforward,
        transformer_activation,
        output_dim,
        attn_dropout,
        layer_norm_eps,
        norm_first,
        bias,
        activation,
        embedding_activation,
        norm_f,
        method,
    ):
        super().__init__()

        self.reshape = Reshape(channels, in_channels, method)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=dim_feedforward,
            dropout=attn_dropout,
            activation=transformer_activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            bias=bias,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=norm_f,
        )

        self.linear = nn.Linear(d_model, output_dim)
        self.activation = activation
        self.embedding_activation = embedding_activation

    def forward(self, x):
        z = self.reshape(x)
        x = self.embedding_activation(z)
        x = self.encoder(x)
        x = z + x
        x = torch.sum(x, dim=1)
        x = self.linear(x)
        x = self.activation(x)
        return x


import torch
import torch.nn as nn

try:
    from rotary_embedding_torch import RotaryEmbedding  # type: ignore[import-untyped]
except ImportError:
    RotaryEmbedding = None  # type: ignore[assignment, misc]


class RotaryEmbeddingLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rotary_embedding = RotaryEmbedding(dim=dim)

    def forward(self, q, k):
        q = self.rotary_embedding.rotate_queries_or_keys(q)
        k = self.rotary_embedding.rotate_queries_or_keys(k)
        return q, k


class RotaryTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.SELU(),  # noqa: B008
        layer_norm_eps=1e-5,
        norm_first=False,
        bias=True,
        batch_first=False,
        **kwargs,
    ):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            batch_first=batch_first,
            bias=bias,
            **kwargs,
        )
        self.rotary_embedding = RotaryEmbeddingLayer(dim=d_model // nhead)
        self.nhead = nhead
        self.d_model = d_model

    def _sa_block(self, x, attn_mask, key_padding_mask):  # type: ignore
        # Multi-head attention with rotary embedding
        device = x.device
        _batch_size, _seq_length, d_model = x.size()
        head_dim = d_model // self.nhead
        qkv = nn.Linear(d_model, d_model * 3, bias=False).to(device)(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.nhead) for t in (q, k, v))

        # Apply rotary embeddings to queries and keys
        q, k = self.rotary_embedding(q, k)

        q = q * (head_dim**-0.5)
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        if attn_mask is not None:
            sim = sim.masked_fill(attn_mask == 0, float("-inf"))
        attn = sim.softmax(dim=-1)
        if self.training:
            attn = self.dropout(attn)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return nn.Linear(d_model, d_model, bias=False).to(device)(out)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Pre-norm if required
        device = src.device
        if self.norm_first:
            src = self.norm1(src)
            src2 = self._sa_block(src, src_mask, src_key_padding_mask).to(device)
            src = src + self.dropout1(src2)
            src = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
        else:
            src2 = self._sa_block(self.norm1(src), src_mask, src_key_padding_mask).to(device)
            src = src + self.dropout1(src2)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
            src = src + self.dropout2(src2)

        return src


class RotaryTransformerEncoder(nn.TransformerEncoder):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
    ):
        super().__init__(
            encoder_layer,
            num_layers,
            norm=norm,
        )

    def forward(self, src, mask=None, src_key_padding_mask=None):  # type: ignore
        return super().forward(src, mask, src_key_padding_mask)
        return super().forward(src, mask, src_key_padding_mask)
