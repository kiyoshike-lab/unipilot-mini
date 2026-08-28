from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from model.attention import CausalSelfAttention


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        rms = values.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(rms.to(values.dtype) + self.eps)
        return normalized * self.weight


def make_norm(kind: str, dimension: int, epsilon: float) -> nn.Module:
    if kind == "layernorm":
        return nn.LayerNorm(dimension, eps=epsilon)
    if kind == "rmsnorm":
        return RMSNorm(dimension, eps=epsilon)
    raise ValueError(f"unsupported norm: {kind}")


class DiagnosticEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        dimension: int,
        dropout: float,
        scale_token_embedding: bool,
    ):
        super().__init__()
        self.token = nn.Embedding(vocab_size, dimension)
        self.position = nn.Embedding(context_length, dimension)
        self.dropout = nn.Dropout(dropout)
        self.token_scale = math.sqrt(dimension) if scale_token_embedding else 1.0

    def forward(self, token_ids: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        _, length = token_ids.shape
        positions = torch.arange(
            position_offset, position_offset + length, device=token_ids.device
        )
        hidden = self.token(token_ids) * self.token_scale
        hidden = hidden + self.position(positions)[None, :, :]
        return self.dropout(hidden)


class DiagnosticFeedForward(nn.Module):
    def __init__(
        self, dimension: int, ffn_dimension: int, dropout: float, bias: bool,
        activation: str,
    ):
        super().__init__()
        if activation == "gelu":
            nonlinearity: nn.Module = nn.GELU()
        elif activation == "silu":
            nonlinearity = nn.SiLU()
        else:
            raise ValueError(f"unsupported activation: {activation}")
        self.network = nn.Sequential(
            nn.Linear(dimension, ffn_dimension, bias=bias),
            nonlinearity,
            nn.Linear(ffn_dimension, dimension, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.network(hidden)


class DiagnosticBlock(nn.Module):
    def __init__(self, config: "DiagnosticConfig"):
        super().__init__()
        self.norm1 = make_norm(config.norm, config.embedding_dim, config.norm_epsilon)
        self.attention = CausalSelfAttention(
            config.embedding_dim,
            config.n_heads,
            config.context_length,
            config.dropout,
            config.bias,
        )
        self.norm2 = make_norm(config.norm, config.embedding_dim, config.norm_epsilon)
        self.feed_forward = DiagnosticFeedForward(
            config.embedding_dim,
            config.ffn_dim,
            config.dropout,
            config.bias,
            config.activation,
        )

    def forward(self, hidden, past_key_value=None, use_cache: bool = False):
        attended, present = self.attention(
            self.norm1(hidden), past_key_value, use_cache
        )
        hidden = hidden + attended
        hidden = hidden + self.feed_forward(self.norm2(hidden))
        return hidden, present


@dataclass
class DiagnosticConfig:
    model_name: str
    vocab_size: int
    context_length: int
    embedding_dim: int
    n_layers: int
    n_heads: int
    ffn_dim: int
    dropout: float = 0.1
    bias: bool = True
    norm: str = "layernorm"
    norm_epsilon: float = 1e-5
    activation: str = "gelu"
    scale_token_embedding: bool = False
    weight_tying: bool = True

    def validate(self) -> None:
        if self.embedding_dim % self.n_heads:
            raise ValueError("embedding_dim must divide evenly across attention heads")
        if self.context_length <= 0 or self.vocab_size <= 0 or self.n_layers <= 0:
            raise ValueError("model dimensions must be positive")
        if self.norm not in {"layernorm", "rmsnorm"}:
            raise ValueError("norm must be layernorm or rmsnorm")
        if self.activation not in {"gelu", "silu"}:
            raise ValueError("activation must be gelu or silu")

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.n_heads

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class DiagnosticTransformer(nn.Module):
    """Isolated Phase 26 model; production and Foundation v1.3 code stay unchanged."""

    def __init__(self, config: DiagnosticConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.embeddings = DiagnosticEmbedding(
            config.vocab_size,
            config.context_length,
            config.embedding_dim,
            config.dropout,
            config.scale_token_embedding,
        )
        self.blocks = nn.ModuleList([
            DiagnosticBlock(config) for _ in range(config.n_layers)
        ])
        self.final_norm = make_norm(config.norm, config.embedding_dim, config.norm_epsilon)
        self.output = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        if config.weight_tying:
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

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None,
        past_key_values=None, use_cache: bool = False,
    ):
        past_length = 0 if past_key_values is None else past_key_values[0][0].size(2)
        if past_length + token_ids.size(1) > self.config.context_length:
            raise ValueError("sequence length exceeds diagnostic model context")
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
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        if use_cache:
            return logits, loss, tuple(presents)
        return logits, loss

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_breakdown(self) -> dict:
        groups = {
            "token_embedding": sum(p.numel() for p in self.embeddings.token.parameters()),
            "position_embedding": sum(p.numel() for p in self.embeddings.position.parameters()),
            "attention": sum(
                p.numel() for block in self.blocks for p in block.attention.parameters()
            ),
            "normalization": sum(
                p.numel()
                for block in self.blocks
                for module in (block.norm1, block.norm2)
                for p in module.parameters()
            ) + sum(p.numel() for p in self.final_norm.parameters()),
            "feed_forward": sum(
                p.numel() for block in self.blocks for p in block.feed_forward.parameters()
            ),
        }
        if not self.config.weight_tying:
            groups["lm_head_untied"] = sum(p.numel() for p in self.output.parameters())
        groups["unique_total"] = self.parameter_count()
        return groups
