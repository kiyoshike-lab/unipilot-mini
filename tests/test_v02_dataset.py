import json
from pathlib import Path

from scripts.check_dataset import inspect
from scripts.generate_dataset_v02 import fixed_prompts, split_for_family
from tokenizer.tokenizer import BPETokenizer
from training.dataset import V02LanguageModelDataset


def test_template_family_split_policy():
    assert [split_for_family(value) for value in (0, 17, 18, 19)] == ["train", "train", "validation", "test"]


def test_fixed_evaluation_prompts_are_complete():
    prompts = fixed_prompts()
    assert len(prompts) == 300 and len({item["id"] for item in prompts}) == 300
    assert len({item["category"] for item in prompts}) == 10


def test_dataset_quality_detection(tmp_path):
    rows = [{"id": "1", "kind": "dialogue", "category": "exams", "template_family": "a", "split": "train",
             "user": "試験について相談したいです。", "assistant": "まず試験範囲を確認して、今日の復習内容を決めましょう。"}]
    path = tmp_path / "sample.jsonl"; path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")
    result = inspect([path]); assert result["invalid_samples"] == 0 and result["duplicates"] == 0


def test_assistant_only_loss_mask(tmp_path):
    row = {"kind": "dialogue", "user": "質問です", "assistant": "回答です"}
    path = tmp_path / "dialogue.jsonl"; path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    tokenizer = BPETokenizer(); dataset = V02LanguageModelDataset(path, tokenizer, 64, assistant_only=True)
    _, targets, _ = dataset[0]
    assert (targets == -100).any() and (targets != -100).any()
