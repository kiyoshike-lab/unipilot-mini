import json

import numpy as np
import torch

from evaluation.analyze_foundation_v33_context_gate import context_gate
from evaluation.evaluate_foundation_v33_context_gate import (
    CONTEXT_LENGTHS,
    CONTEXT_TARGETS,
    context_positions,
)
from foundation.base_tokenizer import FoundationTokenizer
from training import run_foundation_v33_context_gate as training


def _row(seed: int, full: float, delta: float = 0.0) -> dict:
    measured = full + delta
    context = {
        str(length): {
            "loss": measured
            + {512: 0.0, 256: .01, 128: .02, 64: .03, 32: .06, 16: .15, 8: .30, 2: .80, 1: 1.00}[length]
        }
        for length in CONTEXT_LENGTHS
    }
    context["512"]["per_target_losses"] = [measured] * CONTEXT_TARGETS
    context["full_context_advantage_vs_1"] = 1.0
    return {
        "seed": seed,
        "context": context,
        "sanity": {"pass": True},
        "checkpoint_unchanged": True,
    }


def test_context_positions_are_deterministic_regular_validation_targets():
    tokenizer = FoundationTokenizer.load(training.ROOT / "tokenizer/foundation-v11-base-4096.json")
    validation = np.memmap(
        training.ROOT / "data/foundation_v11/packed/vocab-4096/validation.bin",
        dtype=np.uint16,
        mode="r",
    )
    first = context_positions(validation, tokenizer)
    second = context_positions(validation, tokenizer)
    assert len(first) == CONTEXT_TARGETS
    assert np.array_equal(first, second)
    assert not np.isin(validation[first], list(tokenizer.special_to_id.values())).any()


def test_small_context_delta_passes_pre_registered_threshold():
    baseline = [_row(seed, 5.0) for seed in training.SEEDS]
    candidate = [_row(seed, 5.0, .02) for seed in training.SEEDS]
    result = context_gate(baseline, candidate)
    assert result["pass"]
    assert not result["context_regression"]


def test_material_reproduced_context_delta_fails():
    baseline = [_row(seed, 5.0) for seed in training.SEEDS]
    candidate = [_row(seed, 5.0, .08) for seed in training.SEEDS]
    result = context_gate(baseline, candidate)
    assert result["context_regression"]
    assert not result["pass"]


def test_single_seed_anomaly_is_not_called_multiseed_regression_but_stops_gate():
    baseline = [_row(seed, 5.0) for seed in training.SEEDS]
    candidate = [_row(42, 5.0, .20), _row(123, 5.0), _row(2026, 5.0)]
    result = context_gate(baseline, candidate)
    assert not result["context_regression"]
    assert result["single_seed_anomaly"]
    assert not result["pass"]


def test_gate_checkpoints_never_promote_phase43_pilot():
    for seed in training.SEEDS:
        assert training.source_checkpoint(1, seed) == training.formal_checkpoint(seed)
        assert "phase43" not in str(training.source_checkpoint(1, seed))
        assert training.source_checkpoint(2, seed) == training.gate_checkpoint(1, seed)


def test_checkpoint_resume_state_guard():
    payload = {
        "seed": 42,
        "optimizer_state": {"state": {}},
        "scheduler_state": {"global_step": 30500},
        "random_state": {
            "python": 1,
            "numpy": 2,
            "torch_cpu": 3,
            "torch_cuda": 4,
        },
        "permutation": torch.tensor([1]),
        "tokens_processed": 15_616_000,
        "precision_mode": "fp32",
        "phase": 44,
        "eos_loss_weight": 1.5,
        "repetition_auxiliary": False,
        "phase43_experimental_promoted": False,
    }
    assert training.verify_payload(
        payload, seed=42, update=30500, tokens=15_616_000
    )["pass"]


def test_phase44_final_classification_matches_recorded_evidence():
    summary = json.loads(
        (training.ROOT / "evaluation/foundation-v33-context-gate-summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["gate1_decision"] == "CONTINUE_PLUS_256K"
    assert summary["gate2"]["decision"] == "STOP_LM_GATE"
    assert summary["overall_512k"]["context"]["pass"]
    assert summary["overall_512k"]["lm"]["pass"]
    assert summary["overall_512k"]["eos"]["pass"]
    assert not summary["context_regression"]
    assert summary["final_gate"] == "CONTINUE_SHORT_GPU_GATES"
    assert not summary["continue_20m_permission"]
    assert not summary["foundation_base_complete"]
    assert summary["checkpoint_integrity"]["gate1"]
    assert summary["checkpoint_integrity"]["gate2"]
    assert summary["checkpoint_integrity"]["final_blind"]["pass"]
    assert not summary["checkpoint_integrity"]["final_blind"]["opened"]
