from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset):
    """Memory-mapped contiguous next-token blocks for Foundation pretraining."""

    def __init__(self, path: str | Path, context_length: int):
        self.path = Path(path)
        self.context_length = context_length
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.blocks = max(0, (len(self.tokens) - 1) // context_length)
        if self.blocks == 0:
            raise ValueError(f"packed dataset is too short: {self.path}")

    def __len__(self) -> int:
        return self.blocks

    def __getitem__(self, index: int):
        start = index * self.context_length
        values = np.asarray(
            self.tokens[start:start + self.context_length + 1], dtype=np.int64
        ).copy()
        return torch.from_numpy(values[:-1]), torch.from_numpy(values[1:])
