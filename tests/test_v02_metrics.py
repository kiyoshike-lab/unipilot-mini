from evaluation.compare_checkpoints import comparison
from evaluation.evaluate_v02 import balanced_prompts
from evaluation.metrics_v02 import japanese_character_ratio, repetition_rate


def test_repetition_metric_detects_loop():
    assert repetition_rate("大学大学大学大学大学") > repetition_rate("まず範囲を確認します")


def test_japanese_character_ratio():
    assert japanese_character_ratio("今日は大学です。") == 1.0
    assert japanese_character_ratio("abc大学") < 1.0


def test_checkpoint_comparison_markdown():
    result = {"model": "test", "step": 1, "validation_loss": 2.0, "perplexity": 7.39,
              "metrics": {"repetition_rate": 0.1, "japanese_character_ratio": 0.9, "response_not_empty": 1.0, "keyword_relevance": 0.5},
              "generations": []}
    text = comparison([result]); assert "| test | 1 |" in text


def test_balanced_prompt_selection():
    prompts = [{"category": category, "id": f"{category}-{index}"} for category in ["a", "b"] for index in range(3)]
    selected = balanced_prompts(prompts, 4)
    assert [item["category"] for item in selected].count("a") == 2
