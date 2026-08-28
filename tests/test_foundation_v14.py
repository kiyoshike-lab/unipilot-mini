from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


from evaluation.investigate_foundation_v14 import (
    evaluator_audit,
    frequency_baselines,
    language_proxy,
)
from training.investigate_foundation_v14 import (
    learning_rate,
    macro_batch,
    macro_permutation,
)


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_v14_config_is_scratch_equal_token_and_protected():
    config = load("configs/unipilot-foundation-v14.json")
    assert config["initialization"] == "scratch-common-initialization-no-resume"
    assert config["core_token_budget"] == 128_000
    assert config["macro_block_tokens"] == 512
    assert {row["context_length"] for row in config["experiments"]} == {128, 256, 512}
    for experiment in config["experiments"]:
        assert experiment["context_length"] * experiment["micro_batch"] == 512
    assert not any(config[key] for key in (
        "foundation_500_continuation_enabled",
        "foundation_1000_continuation_enabled",
        "standard_46m_enabled",
        "corpus_expansion_enabled",
        "campus_pretraining_enabled",
        "instruction_tuning_enabled",
        "conversation_training_enabled",
        "dpo_enabled",
        "production_enabled",
    ))


def test_macro_batches_use_the_exact_same_target_tokens_across_contexts(tmp_path):
    values = np.arange(1025, dtype=np.uint16)
    path = tmp_path / "tokens.bin"
    values.tofile(path)
    tokens = np.memmap(path, dtype=np.uint16, mode="r")
    _, targets512 = macro_batch(tokens, [1], 512)
    _, targets256 = macro_batch(tokens, [1], 256)
    _, targets128 = macro_batch(tokens, [1], 128)
    assert torch.equal(targets512.flatten(), targets256.flatten())
    assert torch.equal(targets512.flatten(), targets128.flatten())
    assert targets512.numel() == targets256.numel() == targets128.numel() == 512


def test_seeded_data_shuffle_and_schedules_are_deterministic():
    settings = load("configs/unipilot-foundation-v14.json")
    first = macro_permutation(100, settings["seed"])
    second = macro_permutation(100, settings["seed"])
    assert torch.equal(first, second)
    assert not torch.equal(first, torch.arange(100))
    short_final = learning_rate(settings, "short_cosine_250", 249)
    constant_final = learning_rate(settings, "constant_after_warmup20", 249)
    long_final = learning_rate(settings, "long_cosine_1000", 249)
    assert short_final < long_final < constant_final
    assert constant_final == settings["learning_rate"]
    assert learning_rate(settings, "warmup50_constant", 49) == settings["learning_rate"]


def test_generation_evaluator_audit_rejects_old_false_positives():
    old = load("evaluation/foundation-v13-generation.json")
    audit = evaluator_audit(old)
    assert audit["status"] == "ISSUE FOUND"
    assert audit["old_automatic_positive_count"] == 6
    assert all(
        not row["strict_proxy"]["natural_japanese_proxy"]
        and not row["strict_proxy"]["semantic_coherence_proxy"]
        for row in audit["old_automatic_positives"]
    )
    proxy = language_proxy("ははのにが、の。が。\n\nマに、。")
    assert not proxy["natural_japanese_proxy"]
    assert not proxy["semantic_coherence_proxy"]


def test_unigram_and_bigram_baselines_are_exact_on_toy_sequence(tmp_path):
    train_path = tmp_path / "train.bin"
    validation_path = tmp_path / "validation.bin"
    np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint16).tofile(train_path)
    np.asarray([0, 1, 0, 1, 0], dtype=np.uint16).tofile(validation_path)
    train = np.memmap(train_path, dtype=np.uint16, mode="r")
    validation = np.memmap(validation_path, dtype=np.uint16, mode="r")
    result = frequency_baselines(train, validation, vocab=2, probe_tokens=4, alpha=0.1)
    assert result["bigram"]["loss"] < result["unigram"]["loss"]
    assert result["bigram"]["top_1_accuracy"] == 1.0
    assert result["bigram"]["observed_validation_pair_rate"] == 1.0


def test_v14_core_experiments_finished_at_exactly_128k_with_valid_checkpoints():
    config = load("configs/unipilot-foundation-v14.json")
    output = ROOT / "checkpoints/foundation-v14-investigation"
    for experiment in config["experiments"]:
        result = load(f"checkpoints/foundation-v14-investigation/{experiment['name']}.json")
        checkpoint = ROOT / result["checkpoint"]
        assert result["status"] == "COMPLETED"
        assert result["scratch_start"] is True
        assert result["token_budget"] == 128_000
        assert result["updates"] == 250
        assert result["history"][-1]["tokens_processed"] == 128_000
        assert not result["nan_or_inf"] and not result["diverged"]
        assert checkpoint.parent == output
        assert checkpoint.exists()
        assert checkpoint.stat().st_size == result["checkpoint_bytes"]
        assert sha256(checkpoint) == result["checkpoint_sha256"]


def test_v14_batch_pilots_use_same_token_budget_and_expected_updates():
    for multiplier, updates in ((1, 64), (2, 32), (4, 16)):
        result = load(
            f"checkpoints/foundation-v14-investigation/effective_batch_{multiplier}x.json"
        )
        assert result["status"] == "COMPLETED"
        assert result["token_budget"] == 32_768
        assert result["updates"] == updates
        assert result["macro_blocks_seen"] == 64
        assert result["shared_macro_permutation"] is True


def test_v14_summary_has_all_diagnostics_and_protection():
    summary = load("evaluation/foundation-v14-language-investigation.json")
    assert summary["generation_evaluator"]["status"] == "ISSUE FOUND"
    assert [row["step"] for row in summary["v13_diagnostics"]] == [0, 50, 100, 150, 200, 250]
    assert len(summary["experiment_comparison"]) == 5
    assert len(summary["effective_batch_comparison"]) == 3
    assert summary["baseline_lms"]["token_matched_128k"]["method"]["train_tokens"] == 128_000
    assert summary["baseline_lms"]["full_corpus_33m"]["method"]["train_tokens"] == 33_402_759
    assert summary["data_audit"]["randomized"] is True
    assert summary["data_audit"]["initial_128k"]["tokens"] == 128_000
    assert summary["gate"]["status"] in {
        "CONTINUE_PRETRAINING", "CURRICULUM_CHANGE",
        "ARCHITECTURE_INVESTIGATE", "DATA_INVESTIGATE",
    }
    assert summary["protected"]["final_blind_content_opened"] is False
    assert summary["protected"]["final_blind_sha256"] == (
        "fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b"
    )
    assert not any(
        value for key, value in summary["protected"].items()
        if key.endswith("_executed") or key.endswith("_changed")
    )
