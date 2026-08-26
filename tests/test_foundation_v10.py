from __future__ import annotations

import json
from pathlib import Path

from foundation.base_tokenizer import FoundationTokenizer
from foundation.packed_dataset import PackedTokenDataset
from scripts.extract_foundation_v10_dump import clean_wikitext


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_base_tokenizer_roundtrips_and_has_no_old_campus_phrase_tokens():
    tokenizer = FoundationTokenizer.load(ROOT / "tokenizer/foundation-v10-base-4096.json")
    texts = [
        "自然な日本語と基礎推論を学習する。",
        "WikipediaとWikibooksの本文だけでTokenizerを作る。",
        "改行も保持する。\n次の文章です。",
    ]
    assert tokenizer.vocab_size == 4096
    assert all(tokenizer.decode(tokenizer.encode(text)) == text for text in texts)
    pieces = [tokenizer.decode([index]) for index in range(tokenizer.vocab_size)]
    assert not any("入学年度の学生便覧と成績表" in piece for piece in pieces)
    assert not any("大学生活の相談です" in piece for piece in pieces)


def test_dump_cleaner_removes_markup_and_reference_sections():
    text, metrics = clean_wikitext(
        "{{Infobox|x=y}}\n'''自然言語'''は[[言語学|言語]]の対象である。<ref>出典</ref>\n"
        "== 脚注 ==\n* 外部リンク\n"
    )
    assert "自然言語" in text and "言語の対象" in text
    assert "Infobox" not in text and "出典" not in text and "外部リンク" not in text
    assert metrics["modified"] is True


def test_corpus_is_large_split_and_contamination_safe():
    audit = load("evaluation/foundation-v10-data-audit.json")
    packed = load("data/foundation_v10/packed/vocab-4096/manifest.json")
    assert audit["unique_documents"] == packed["total_documents"] == 11_499
    assert packed["total_tokens"] >= 30_000_000
    assert set(packed["splits"]) == {"train", "validation", "test"}
    assert all(packed["splits"][split]["tokens"] > 0 for split in packed["splits"])
    assert audit["excluded"].get("semantic_holdout_overlap", 0) == 0
    assert audit["holdout_audit"]["maximum_segment_question_similarity"] < .78
    assert audit["excluded"]["semantic_duplicate"] > 0


def test_packed_dataset_reads_contiguous_next_token_blocks():
    manifest = load("data/foundation_v10/packed/vocab-4096/manifest.json")
    dataset = PackedTokenDataset(ROOT / manifest["splits"]["validation"]["path"], 512)
    inputs, targets = dataset[0]
    assert inputs.shape == targets.shape == (512,)
    assert inputs[1:].tolist() == targets[:-1].tolist()


def test_tokenizer_comparison_uses_base_train_and_selects_4096():
    report = load("evaluation/foundation-v10-tokenizer-benchmark.json")
    assert report["tokenizers_trained_from_scratch_on_base_train_only"] is True
    assert report["comparison_is_heldout_validation_and_test"] is True
    assert report["selected_vocab"] == 4096
    assert report["compression_gain_4096_over_2048"] >= .10
    assert all(row["exact_roundtrip_rate"] == 1 for row in report["results"])


def test_model_matrix_selects_20m_by_efficiency():
    report = load("evaluation/foundation-v10-model-comparison-50.json")
    assert [row["architecture"] for row in report["results"]] == ["20m", "30m", "46m"]
    assert all(row["healthy_loss_curve"] for row in report["results"])
    assert report["selected_architecture"] == "20m"


def test_selected_learning_curve_is_monotonic_but_base_gate_fails():
    values = []
    for step in (50, 100, 250, 500):
        manifest = load(
            f"checkpoints/foundation-v10-sanity/20m/checkpoint-step-{step}.manifest.json"
        )
        if not values:
            values.append(manifest["initial_validation_loss"])
        values.append(manifest["training_metrics"]["validation_loss"])
    assert all(left > right for left, right in zip(values, values[1:]))
    result = load("evaluation/foundation-v10-base-100-step-500.json")
    assert result["questions"] == 100 and result["campus_questions"] == 0
    assert result["base_gate"] == "FAIL"
    assert result["metrics"]["natural_japanese_rate"] < .95


def test_curriculum_stays_separated_and_only_stage_a_runs():
    config = load("configs/unipilot-foundation-v10.json")
    assert [row["stage"] for row in config["curriculum"]] == list("ABCDEF")
    assert [row["enabled_for_sanity"] for row in config["curriculum"]] == [
        True, False, False, False, False, False,
    ]
    summary = load("evaluation/foundation-v10-summary.json")
    assert summary["decisions"]["start_campus_or_instruction_stage"] is False
    assert summary["decisions"]["start_reward_or_dpo"] is False
    assert summary["baseline"]["protected_changes"] == []
