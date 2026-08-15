from __future__ import annotations

import json
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset


CHAT_TEMPLATE = "<BOS><USER>\n{user}\n<ASSISTANT>\n{assistant}<EOS>"


def load_documents(path: str | Path) -> list[str]:
    source = Path(path)
    files = sorted(source.rglob("*")) if source.is_dir() else [source]
    documents: list[str] = []
    for file in files:
        if file.suffix == ".jsonl":
            for line in file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if "user" in row and "assistant" in row:
                    documents.append(CHAT_TEMPLATE.format(user=row["user"], assistant=row["assistant"]))
                elif "text" in row:
                    documents.append(f"<BOS>{row['text']}<EOS>")
        elif file.suffix == ".txt":
            documents.extend(f"<BOS>{line}<EOS>" for line in file.read_text(encoding="utf-8").splitlines() if line.strip())
    if not documents:
        raise ValueError(f"no .jsonl or .txt documents found at {source}")
    return documents


class LanguageModelDataset(Dataset):
    def __init__(self, documents: list[str], tokenizer, context_length: int, stride: int | None = None):
        self.context_length = context_length
        stride = stride or context_length
        self.samples: list[list[int]] = []
        for document in documents:
            ids = tokenizer.encode(document)
            for start in range(0, max(1, len(ids) - 1), stride):
                chunk = ids[start:start + context_length + 1]
                if len(chunk) >= 2:
                    self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ids = self.samples[index]
        inputs = ids[:-1]
        targets = ids[1:]
        padding = self.context_length - len(inputs)
        inputs = inputs + [0] * padding
        targets = targets + [-100] * padding
        return torch.tensor(inputs), torch.tensor(targets)


def split_documents(documents: list[str], validation_ratio: float = 0.1, seed: int = 42):
    documents = list(documents)
    random.Random(seed).shuffle(documents)
    boundary = max(1, int(len(documents) * (1 - validation_ratio)))
    return documents[:boundary], documents[boundary:]


class V02LanguageModelDataset(Dataset):
    """Pre-split v0.2 records with optional assistant-response-only loss masking."""

    def __init__(self, path: str | Path, tokenizer, context_length: int, assistant_only: bool = True, kinds: set[str] | None = None, max_records: int = 0):
        self.context_length = context_length
        self.samples: list[tuple[list[int], list[int], str]] = []
        self.kind_counts: dict[str, int] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            kind = row.get("kind", "general")
            if kinds and kind not in kinds:
                continue
            if kind == "dialogue":
                prefix = f"<BOS><USER>\n{row['user']}\n<ASSISTANT>\n"
                prefix_ids = tokenizer.encode(prefix)
                ids = prefix_ids + tokenizer.encode(row["assistant"]) + [tokenizer.eos_id]
                first_learned_target = len(prefix_ids)
            else:
                ids = tokenizer.encode(row["text"], add_bos=True, add_eos=True)
                first_learned_target = 1
            for start in range(0, max(1, len(ids) - 1), context_length):
                chunk = ids[start:start + context_length + 1]
                if len(chunk) < 2:
                    continue
                inputs, targets = chunk[:-1], chunk[1:]
                if assistant_only and kind == "dialogue":
                    for local_index in range(len(targets)):
                        target_absolute_index = start + local_index + 1
                        if target_absolute_index < first_learned_target:
                            targets[local_index] = -100
                if all(target == -100 for target in targets):
                    continue
                padding = context_length - len(inputs)
                self.samples.append((inputs + [tokenizer.pad_id] * padding, targets + [-100] * padding, kind))
                self.kind_counts[kind] = self.kind_counts.get(kind, 0) + 1
            if max_records and sum(self.kind_counts.values()) >= max_records:
                break

    def __len__(self): return len(self.samples)

    def __getitem__(self, index):
        inputs, targets, kind = self.samples[index]
        return torch.tensor(inputs), torch.tensor(targets), kind
