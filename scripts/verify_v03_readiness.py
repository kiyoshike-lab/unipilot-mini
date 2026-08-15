from __future__ import annotations

import json
from pathlib import Path
import torch

from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from scripts.check_dataset_v03 import check
from tokenizer.tokenizer import BPETokenizer
from training.dataset_v03 import CurriculumDataset


def main():
    settings = json.loads(Path("configs/unipilot-v03.json").read_text(encoding="utf-8"))
    tokenizer = BPETokenizer.load(settings["tokenizer"]); assert tokenizer.vocab_size == 512
    config = ModelConfig(**settings["model"]); model = UniPilotTransformer(config); assert model.parameter_count() == 19_814_784
    dataset_report = check(); assert dataset_report["invalid_samples"] == 0 and dataset_report["conversation_eos_valid_rate"] == 1.0
    conversation = CurriculumDataset("data/v03/stage_c/train.jsonl", tokenizer, config.context_length, True, max_records=1)
    inputs, targets, _, row_id = conversation[0]
    learned = targets != -100
    assert (~learned).any() and learned.any(), "assistant mask must contain ignored and learned targets"
    learned_ids = []
    for _, chunk_targets, _, chunk_row_id in conversation:
        if chunk_row_id == row_id: learned_ids.extend(chunk_targets[chunk_targets != -100].tolist())
    assert tokenizer.eos_id in learned_ids, "EOS must be a learned assistant target across conversation chunks"
    checkpoint = Path("checkpoints/unipilot-v02-step-1000/checkpoint-step-1000.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    old_config = ModelConfig(**payload["config"])
    compatible = old_config.to_dict() | {"model_name": config.model_name} == config.to_dict()
    model.load_state_dict(payload["model_state"])
    assert compatible and payload.get("optimizer_state") and payload.get("scheduler_state")
    print(json.dumps({"dataset_validation": "pass", "mask_validation": "pass", "eos_validation": "pass",
                      "v02_checkpoint_compatible": True, "scratch_supported": True, "resume_v02_supported": True,
                      "parameters": model.parameter_count(), "vocab_size": tokenizer.vocab_size}, indent=2))


if __name__ == "__main__": main()
