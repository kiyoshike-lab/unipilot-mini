from torch import nn
from .attention import CausalSelfAttention


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, ffn_dim: int, dropout: float, bias: bool):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim, bias=bias),
            nn.GELU(),
            nn.Linear(ffn_dim, embedding_dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, ffn_dim: int, context_length: int, dropout: float, bias: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = CausalSelfAttention(embedding_dim, n_heads, context_length, dropout, bias)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, ffn_dim, dropout, bias)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))
