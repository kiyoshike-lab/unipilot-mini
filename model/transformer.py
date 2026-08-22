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

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None,
                past_key_values=None, use_cache: bool = False):
        past_length = 0 if past_key_values is None else past_key_values[0][0].size(2)
        if past_length + token_ids.size(1) > self.config.context_length:
            raise ValueError(f"sequence length exceeds context length {self.config.context_length}")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values must contain one entry per Transformer block")
        hidden = self.embeddings(token_ids, position_offset=past_length)
        presents = []
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[index]
            hidden, present = block(hidden, past, use_cache)
            if use_cache:
                presents.append(present)
        logits = self.output(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        if use_cache:
            return logits, loss, tuple(presents)
        return logits, loss

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
