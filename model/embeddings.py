import torch
from torch import nn


class TokenPositionEmbedding(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embedding_dim)
        self.position = nn.Embedding(context_length, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, length = token_ids.shape
        positions = torch.arange(length, device=token_ids.device)
        return self.dropout(self.token(token_ids) + self.position(positions)[None, :, :])
