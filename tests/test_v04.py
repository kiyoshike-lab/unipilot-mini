import json
from pathlib import Path

import torch

from evaluation.metrics_v04 import broken_generation_metrics, ngram_repetition
from inference.generate import generate_text
from scripts.prepare_dataset_v04 import OPENINGS
from tokenizer.tokenizer import BPETokenizer
from training.dataset_v03 import CurriculumDataset
from training.train_v04 import eos_weighted_loss


def rows():
    return [json.loads(line) for path in Path("data/v04/stage_c").glob("*.jsonl") for line in path.read_text(encoding="utf-8").splitlines()]


def test_clean_stage_c_validation():
    data = rows(); assert len(data) == 8000
    assert len({row["user"] + row["assistant"] for row in data}) == len(data)
    assert all(row["dataset_version"] == "unipilot-clean-conversation-v04" for row in data)


def test_eos_coverage_and_ending_quality():
    data = rows(); assert all(row["eos_required"] and row["eos_ending_valid"] for row in data)
    assert all(row["assistant"].endswith(("。", "！", "？")) for row in data)


def test_v04_masking_and_eos_target():
    tokenizer = BPETokenizer.load("tokenizer/vocab-v02-512.json")
    dataset = CurriculumDataset("data/v04/stage_c/test.jsonl", tokenizer, 256, True, max_records=1)
    _, targets, _, _ = dataset[0]; learned = targets[targets != -100]
    assert len(learned) and tokenizer.eos_id in learned.tolist() and (targets == -100).any()


def test_eos_weighted_loss_changes_eos_contribution():
    logits = torch.zeros(1, 2, 4); targets = torch.tensor([[1, 3]])
    logits[0, 1, 3] = -4
    assert eos_weighted_loss(logits, targets, 3, 2.0) > eos_weighted_loss(logits, targets, 3, 1.0)


def test_generation_stops_immediately_on_eos():
    tokenizer = BPETokenizer()
    class EosModel(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(1)); self.config = type("Config", (), {"context_length": 32})()
        def forward(self, inputs):
            logits = torch.full((1, inputs.size(1), tokenizer.vocab_size), -10.0)
            logits[:, -1, tokenizer.eos_id] = 10.0
            return logits, None
    _, metrics = generate_text(EosModel(), tokenizer, "test", max_new_tokens=10, temperature=0)
    assert metrics["eos_reached"] and metrics["tokens"] == 1


def test_too_short_ngram_and_broken_metrics():
    assert ngram_repetition("試験試験試験", 2) > 0
    assert broken_generation_metrics("正常な日本語です。")["invalid_sequence_rate"] == 0
    assert broken_generation_metrics("�") ["broken_byte_rate"] > 0


def test_category_contamination_and_opening_distribution():
    quality = json.loads(Path("evaluation/dataset-quality-v04.json").read_text(encoding="utf-8"))
    assert quality["category_contamination"] == 0 and quality["max_opening_ratio"] < .10
    assert quality["exact_duplicates"] == 0 and quality["broken_samples"] == 0


def test_human_evaluation_persistence_schema():
    items = json.loads(Path("evaluation/human-eval-v03.json").read_text(encoding="utf-8"))
    assert len(items) == 50 and all("score" in item and "notes" in item for item in items)


def test_v04_experiment_manifest_schema():
    required = {"experiment_id", "base_checkpoint", "dataset_version", "eos_weight", "step", "seed", "generation", "git_commit"}
    source = Path("training/train_v04.py").read_text(encoding="utf-8")
    assert all(key in source for key in required)
