from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from foundation.base_tokenizer import FoundationTokenizer
from foundation.packed_dataset import PackedTokenDataset
from training.foundation_v12_diagnostics import causal_mask_audit, loss_audit


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_target_shift_is_exactly_one_token_for_ten_blocks():
    manifest = load("data/foundation_v11/packed/vocab-4096/manifest.json")
    path = ROOT / manifest["splits"]["train"]["path"]
    dataset = PackedTokenDataset(path, 512)
    raw = np.memmap(path, dtype=np.uint16, mode="r")
    for block in range(10):
        inputs, targets = dataset[block]
        start = block * 512
        assert torch.equal(inputs, torch.from_numpy(
            np.asarray(raw[start:start + 512], dtype=np.int64).copy()
        ))
        assert torch.equal(targets, torch.from_numpy(
            np.asarray(raw[start + 1:start + 513], dtype=np.int64).copy()
        ))
        assert torch.equal(inputs[1:], targets[:-1])


def test_future_tokens_do_not_change_past_logits_and_cache_matches():
    audit = causal_mask_audit()
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())


def test_cross_entropy_pad_eos_and_vocabulary_alignment():
    audit = loss_audit()
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())


def test_foundation_tokenizers_still_roundtrip_exactly():
    samples = [
        "今日は晴れです。",
        "大学では、根拠を確認して文章を書く。",
        "ASCII 123 / 日本語 / 改行\n二行目です。",
    ]
    for vocab in (2048, 4096):
        tokenizer = FoundationTokenizer.load(ROOT / f"tokenizer/foundation-v11-base-{vocab}.json")
        assert all(tokenizer.decode(tokenizer.encode(text)) == text for text in samples)


def test_phase23_report_only_allows_lr_after_training_core_pass():
    path = ROOT / "evaluation/foundation-v12-training-dynamics.json"
    if not path.exists():
        return
    report = load("evaluation/foundation-v12-training-dynamics.json")
    if report["gates"]["training_core"] != "PASS":
        assert report["lr_sweep"]["status"] == "NOT_RUN"
        assert report["tokenizer_training_comparison"]["status"] == "NOT_RUN"
    assert report["protected"]["final_blind_content_opened"] is False
    assert report["decisions"]["full_clean_250_executed"] is False
    assert report["decisions"]["full_clean_500_executed"] is False


def test_tiny_overfit_and_eos_learning_gates_pass():
    report = load("evaluation/foundation-v12-training-dynamics.json")
    tiny = report["tiny_overfit"]["results"]["10_documents"]
    assert tiny["metrics"]["final_train_loss"] < 1.0
    assert tiny["memorization"]["mean_token_exact_rate"] >= .75
    assert tiny["memorization"]["eos_rate"] >= .8
    assert report["eos_sanity"]["status"] == "PASS"
    assert report["eos_sanity"]["eos_top1_rate"] >= .9


def test_lr_sweep_and_vocab_comparison_are_fixed_and_complete():
    report = load("evaluation/foundation-v12-training-dynamics.json")
    assert report["gates"]["training_core"] == "PASS"
    assert {row["learning_rate"] for row in report["lr_sweep"]["results"]} == {
        3e-5, 1e-4, 3e-4, 6e-4
    }
    assert all(not row["metrics"]["diverged"] for row in report["lr_sweep"]["results"])
    comparison = report["tokenizer_training_comparison"]
    assert {row["metrics"]["vocab"] for row in comparison["results"]} == {2048, 4096}
    assert comparison["parameter_gap_percent"] < .02


def test_gradient_weight_checkpoint_and_protection_gates_pass():
    report = load("evaluation/foundation-v12-training-dynamics.json")
    assert report["gradient_health"]["status"] == "PASS"
    assert report["weight_update"]["status"] == "PASS"
    assert report["verification"]["resume"] == "PASS"
    assert report["verification"]["checkpoint_integrity"] == "PASS"
    assert report["protected"]["final_blind_sha256"] == report["protected"][
        "final_blind_expected_sha256"
    ]
