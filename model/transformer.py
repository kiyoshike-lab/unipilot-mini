from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .embeddings import TokenPositionEmbedding
from .layers import TransformerBlock


class UniPilotTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.embeddings = TokenPositionEmbedding(config.vocab_size, config.context_length, config.embedding_dim, config.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embedding_dim, config.n_heads, config.ffn_dim, config.context_length, config.dropout, config.bias)
            for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.output = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.output.weight = self.embeddings.token.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None):
        if token_ids.size(1) > self.config.context_length:
            raise ValueError(f"sequence length exceeds context length {self.config.context_length}")
        hidden = self.embeddings(token_ids)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.output(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
