from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class ModelConfig:
    model_name: str = "UniPilot Mini v0.1"
    vocab_size: int = 512
    context_length: int = 256
    embedding_dim: int = 384
    n_layers: int = 10
    n_heads: int = 6
    ffn_dim: int = 1536
    dropout: float = 0.1
    bias: bool = True

    def validate(self) -> None:
        if self.embedding_dim % self.n_heads:
            raise ValueError("embedding_dim must be divisible by n_heads")
        if min(self.vocab_size, self.context_length, self.n_layers, self.n_heads) <= 0:
            raise ValueError("model dimensions must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        config = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        config.validate()
        return config
