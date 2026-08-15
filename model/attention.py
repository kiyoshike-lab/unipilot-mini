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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:, :, :length, :length], float("-inf"))
        weights = self.attention_dropout(torch.softmax(scores, dim=-1))
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.output_dropout(self.projection(attended))
