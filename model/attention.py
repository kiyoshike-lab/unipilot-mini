import math
import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention implemented directly in PyTorch."""

    def __init__(self, embedding_dim: int, n_heads: int, context_length: int, dropout: float, bias: bool = True):
        super().__init__()
        if embedding_dim % n_heads:
            raise ValueError("embedding_dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        self.projection = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, context_length, context_length), persistent=False)

    def forward(self, x: torch.Tensor, past_key_value=None, use_cache: bool = False):
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        past_length = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_length = past_k.size(2)
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        # Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        total_length = past_length + length
        mask = self.causal_mask[:, :, past_length:total_length, :total_length]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = self.attention_dropout(torch.softmax(scores, dim=-1))
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch, length, channels)
        output = self.output_dropout(self.projection(attended))
        return output, ((k, v) if use_cache else None)
