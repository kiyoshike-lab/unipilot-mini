from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ReferenceConfigV18:
    model_name: str
    vocab_size: int
    context_length: int
    embedding_dim: int
    n_layers: int
    n_heads: int
    ffn_dim: int
    dropout: float = 0.1
    bias: bool = True
    norm_epsilon: float = 1e-5
    residual_projection_init_scale: float = 1.0
    weight_tying: bool = True

    def validate(self) -> None:
        if self.embedding_dim % self.n_heads:
            raise ValueError("embedding_dim must divide evenly across attention heads")
        if min(self.vocab_size, self.context_length, self.n_layers) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.residual_projection_init_scale <= 0:
            raise ValueError("residual projection initialization scale must be positive")

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.n_heads

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class ReferenceEmbeddingV18(nn.Module):
    def __init__(self, config: ReferenceConfigV18):
        super().__init__()
        self.token = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position = nn.Embedding(config.context_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        length = token_ids.size(1)
        positions = torch.arange(length, device=token_ids.device)
        return self.dropout(
            self.token(token_ids) + self.position(positions)[None, :, :]
        )


class ReferenceBlockV18(nn.Module):
    def __init__(self, config: ReferenceConfigV18):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embedding_dim, eps=config.norm_epsilon)
        self.attention = nn.MultiheadAttention(
            config.embedding_dim,
            config.n_heads,
            dropout=config.dropout,
            bias=config.bias,
            batch_first=True,
        )
        self.attention_output_dropout = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(config.embedding_dim, eps=config.norm_epsilon)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embedding_dim, config.ffn_dim, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.embedding_dim, bias=config.bias),
            nn.Dropout(config.dropout),
        )
        mask = torch.triu(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        length = hidden.size(1)
        normalized = self.norm1(hidden)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=self.causal_mask[:length, :length],
            need_weights=False,
        )
        hidden = hidden + self.attention_output_dropout(attended)
        hidden = hidden + self.feed_forward(self.norm2(hidden))
        return hidden


class ReferenceTransformerV18(nn.Module):
    """Independent diagnostic decoder built on torch.nn.MultiheadAttention."""

    def __init__(self, config: ReferenceConfigV18):
        super().__init__()
        config.validate()
        self.config = config
        self.embeddings = ReferenceEmbeddingV18(config)
        self.blocks = nn.ModuleList([
            ReferenceBlockV18(config) for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.embedding_dim, eps=config.norm_epsilon)
        self.output = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        if config.weight_tying:
            self.output.weight = self.embeddings.token.weight
        self.apply(self._initialize_module)
        self._initialize_mha_inputs()
        self._initialize_residual_outputs()

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _initialize_mha_inputs(self) -> None:
        for block in self.blocks:
            nn.init.normal_(block.attention.in_proj_weight, mean=0.0, std=0.02)
            if block.attention.in_proj_bias is not None:
                nn.init.zeros_(block.attention.in_proj_bias)

    def _initialize_residual_outputs(self) -> None:
        residual_std = 0.02 * self.config.residual_projection_init_scale
        for block in self.blocks:
            nn.init.normal_(block.attention.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.feed_forward[2].weight, mean=0.0, std=residual_std)

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if token_ids.size(1) > self.config.context_length:
            raise ValueError("sequence length exceeds reference model context")
        hidden = self.embeddings(token_ids)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.output(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def architecture_manifest(self) -> dict:
        return {
            "implementation": "independent torch.nn.MultiheadAttention reference",
            "attention": "torch.nn.MultiheadAttention(batch_first=True)",
            "causal_mask": "upper triangular boolean attn_mask",
            "norm_flow": "Pre-LayerNorm",
            "final_norm": "PRESENT",
            "position": "learned absolute",
            "activation": "GELU",
            "weight_tying": self.config.weight_tying,
            "base_initialization_std": 0.02,
            "residual_output_projection_std": (
                0.02 * self.config.residual_projection_init_scale
            ),
            "qkv_and_mlp_input_std": 0.02,
            "custom_attention_code_shared": False,
            "custom_residual_block_code_shared": False,
        }
