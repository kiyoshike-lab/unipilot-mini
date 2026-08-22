from __future__ import annotations

import json
from pathlib import Path
import torch
from torch.utils.data import Dataset


SYSTEM_TEXT = "あなたは大学生活を支援する完全ローカルのUniPilot Miniです。情報がない場合は推測せず、確認方法を案内します。"


class CurriculumDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, context_length: int, assistant_only: bool, max_records: int = 0):
        self.context_length = context_length; self.pad_id = tokenizer.pad_id; self.samples = []
        accepted = 0
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line); kind = row["kind"]
            if kind == "conversation":
                # v0.4 also has a structured metadata field named ``context``.
                # Only v0.7 retrieval-grounding rows use prompt-ready text here.
                context = row.get("context") if isinstance(row.get("context"), str) else None
                context_block = f"<CONTEXT>\n{context}\n" if context else ""
                prefix = f"<BOS><SYSTEM>\n{SYSTEM_TEXT}\n{context_block}<USER>\n{row['user']}\n<ASSISTANT>\n"
                prefix_ids = tokenizer.encode(prefix)
                ids = prefix_ids + tokenizer.encode(row["assistant"]) + [tokenizer.eos_id]
                first_assistant_token = len(prefix_ids)
            else:
                ids = tokenizer.encode(row["text"], add_bos=True, add_eos=True)
                first_assistant_token = 1
            for start in range(0, max(1, len(ids) - 1), context_length):
                chunk = ids[start:start + context_length + 1]
                if len(chunk) < 2: continue
                inputs, targets = chunk[:-1], chunk[1:]
                if assistant_only and kind == "conversation":
                    for local in range(len(targets)):
                        if start + local + 1 < first_assistant_token: targets[local] = -100
                if all(target == -100 for target in targets): continue
                padding = context_length - len(inputs)
                self.samples.append((inputs + [self.pad_id] * padding, targets + [-100] * padding, kind, row["id"]))
            accepted += 1
            if max_records and accepted >= max_records: break

    def __len__(self): return len(self.samples)

    def __getitem__(self, index):
        inputs, targets, kind, row_id = self.samples[index]
        return torch.tensor(inputs), torch.tensor(targets), kind, row_id


def dynamic_collate(batch):
    inputs = torch.stack([item[0] for item in batch]); targets = torch.stack([item[1] for item in batch])
    pad_id = 0; lengths = (inputs != pad_id).sum(dim=1); length = max(2, int(lengths.max()))
    return inputs[:, :length], targets[:, :length], [item[2] for item in batch], [item[3] for item in batch]
