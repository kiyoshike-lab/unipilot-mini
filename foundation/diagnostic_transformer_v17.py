from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from foundation.diagnostic_transformer_v15 import make_norm
from model.attention import CausalSelfAttention


class DiagnosticEmbeddingV17(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        dimension: int,
        dropout: float,
        token_embedding_scale: float,
        position_embedding_scale: float,
    ):
        super().__init__()
        self.token = nn.Embedding(vocab_size, dimension)
        self.position = nn.Embedding(context_length, dimension)
        self.dropout = nn.Dropout(dropout)
        self.token_scale = float(token_embedding_scale)
        self.position_scale = float(position_embedding_scale)

    def forward(self, token_ids: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        _, length = token_ids.shape
        positions = torch.arange(
            position_offset, position_offset + length, device=token_ids.device
        )
        token = self.token(token_ids) * self.token_scale
        position = self.position(positions)[None, :, :] * self.position_scale
        return self.dropout(token + position)


class DiagnosticFeedForwardV17(nn.Module):
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


class DiagnosticBlockV17(nn.Module):
    def __init__(self, config: "DiagnosticConfigV17"):
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
        self.feed_forward = DiagnosticFeedForwardV17(
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
class DiagnosticConfigV17:
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
    token_embedding_scale: float = 1.0
    position_embedding_scale: float = 1.0
    residual_projection_init_scale: float = 1.0
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
        if self.token_embedding_scale <= 0 or self.position_embedding_scale <= 0:
            raise ValueError("embedding scales must be positive")
        if self.residual_projection_init_scale <= 0:
            raise ValueError("residual projection initialization scale must be positive")

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.n_heads

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class DiagnosticTransformerV17(nn.Module):
    """Phase 28 isolation model; production and formal Foundation remain unchanged."""

    def __init__(self, config: DiagnosticConfigV17):
        super().__init__()
        config.validate()
        self.config = config
        self.embeddings = DiagnosticEmbeddingV17(
            config.vocab_size,
            config.context_length,
            config.embedding_dim,
            config.dropout,
            config.token_embedding_scale,
            config.position_embedding_scale,
        )
        self.blocks = nn.ModuleList([
            DiagnosticBlockV17(config) for _ in range(config.n_layers)
        ])
        self.final_norm = make_norm(config.norm, config.embedding_dim, config.norm_epsilon)
        self.output = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        if config.weight_tying:
            self.output.weight = self.embeddings.token.weight
        self.apply(self._init_weights)
        self._scale_residual_projection_initialization()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projection_initialization(self) -> None:
        scale = self.config.residual_projection_init_scale
        if scale == 1.0:
            return
        residual_std = 0.02 * scale
        for block in self.blocks:
            nn.init.normal_(block.attention.projection.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.feed_forward.network[2].weight, mean=0.0, std=residual_std)

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

    def initialization_manifest(self) -> dict:
        return {
            "base_std": 0.02,
            "token_embedding_std": 0.02,
            "position_embedding_std": 0.02,
            "effective_token_embedding_std": 0.02 * self.config.token_embedding_scale,
            "effective_position_embedding_std": 0.02 * self.config.position_embedding_scale,
            "attention_qkv_std": 0.02,
            "attention_output_projection_std": (
                0.02 * self.config.residual_projection_init_scale
            ),
            "mlp_input_projection_std": 0.02,
            "mlp_output_projection_std": (
                0.02 * self.config.residual_projection_init_scale
            ),
            "depth_scaled_formula": (
                "base_std / sqrt(2 * n_layers)"
                if math.isclose(
                    self.config.residual_projection_init_scale,
                    1 / math.sqrt(2 * self.config.n_layers),
                )
                else "not applied"
            ),
        }
