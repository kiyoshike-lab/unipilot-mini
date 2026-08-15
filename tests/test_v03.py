import json
from pathlib import Path
import torch

from evaluation.metrics_v03 import broken_text_metrics, infer_category, semantic_score
from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from scripts.prepare_dataset_v03 import fixed_prompts
from tokenizer.tokenizer import BPETokenizer
from training.dataset_v03 import CurriculumDataset
from training.optimizer import create_optimizer
from training.train_v03 import save_v03_checkpoint, stage_for_step


def test_v03_assistant_mask_and_eos(tmp_path):
    row = {"id": "x", "kind": "conversation", "user": "試験は明日です", "assistant": "試験範囲を確認します。"}
    path = tmp_path / "data.jsonl"; path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    tokenizer = BPETokenizer(); dataset = CurriculumDataset(path, tokenizer, 128, True)
    _, targets, _, _ = dataset[0]; learned = targets != -100
    assert (~learned).any() and learned.any()
    assert tokenizer.eos_id in targets[learned].tolist()


def test_v03_fixed_prompt_semantic_schema():
    prompts = fixed_prompts(); assert len(prompts) == 300
    assert all(item["intent"] and item["expected_keywords"] and "forbidden_keywords" in item for item in prompts)


def test_semantic_keyword_and_category_metrics():
    item = {"category": "exam", "expected_keywords": ["試験", "範囲", "復習"], "forbidden_keywords": ["履修登録"]}
    result = semantic_score("試験範囲を確認して復習しましょう。", item)
    assert result["expected_keyword_rate"] == 1 and result["category_correct"] and result["relevance_score"] >= 50
    assert infer_category("課題の締切を確認します")[0] == "assignment"
    unrelated = semantic_score("これは自然な日本語の文章ですが、内容は別です。", item)
    assert unrelated["meaningful_response"] is False


def test_broken_text_metric():
    assert broken_text_metrics("試験範囲を確認します。") ["broken"] is False
    assert broken_text_metrics("�\x01") ["broken"] is True


def test_human_evaluation_schema():
    rows = json.loads(Path("evaluation/human-eval-v03.json").read_text(encoding="utf-8"))
    assert len(rows) == 50 and all(row["score"] is None and "score_guide" in row for row in rows)


def test_curriculum_stage_transition():
    stages = [{"name": "A", "start_step": 0, "end_step": 10}, {"name": "B", "start_step": 10, "end_step": 20}]
    assert stage_for_step(stages, 9)["name"] == "A" and stage_for_step(stages, 10)["name"] == "B"


def test_training_manifest_and_resume_state(tmp_path):
    config = ModelConfig(vocab_size=263, context_length=8, embedding_dim=16, n_layers=1, n_heads=4, ffn_dim=32)
    model = UniPilotTransformer(config); optimizer = create_optimizer(model, 1e-3)
    settings = {"dataset_version": "unipilot-dataset-v03", "experiment_id": "test", "seed": 1, "weight_decay": .1,
                "generation": {}, "stages": []}
    stage = {"name": "C", "learning_rate": 1e-3, "warmup_steps": 1}
    path = tmp_path / "checkpoint.pt"; save_v03_checkpoint(path, model, optimizer, 12, 2.0, settings, stage, "scratch-v03", {})
    payload = torch.load(path, weights_only=False)
    assert payload["step"] == 12 and payload["v03_manifest"]["dataset"] == "unipilot-dataset-v03"
