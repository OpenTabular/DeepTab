# ruff: noqa

import torch
import torch.nn as nn
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding  # type: ignore[import-untyped]


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
        activation=nn.SELU(),
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
        batch_size, seq_length, d_model = x.size()
        head_dim = d_model // self.nhead
        qkv = nn.Linear(d_model, d_model * 3, bias=False).to(device)(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.nhead), (q, k, v))

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
