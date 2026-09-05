import json

import torch

from evaluation.analyze_foundation_v32_maturity import (
    attractor_classification,
    classify_scaling,
    per_million_trends,
)
from evaluation.evaluate_foundation_v32_maturity import MODES, _topic_retention
from training import run_foundation_v32_maturity_pilot as pilot


def _rows():
    return [
        {"tokens": 5_000_000, "validation_loss": 5.2, "top1": .18, "semantic": .30,
         "greedy_repetition_1": .95, "median_loop_onset": 6},
        {"tokens": 10_000_000, "validation_loss": 4.7, "top1": .23, "semantic": .40,
         "greedy_repetition_1": .93, "median_loop_onset": 14},
        {"tokens": 15_000_000, "validation_loss": 4.45, "top1": .25, "semantic": .55,
         "greedy_repetition_1": .92, "median_loop_onset": 20},
    ]


def test_scaling_classifies_slowing_but_healthy():
    assert classify_scaling(_rows()) == "SLOWING_BUT_HEALTHY"
    trends = per_million_trends(_rows())
    assert trends[0]["loss_improvement_per_million"] == .1
    assert trends[-1]["top1_percentage_point_improvement_per_million"] > 0


def test_attractor_uses_repetition_and_onset_not_runaway_alone():
    assert attractor_classification(_rows()) == "WEAKENING"


def test_pilot_is_isolated_and_conservative():
    target = pilot.checkpoint_path()
    assert target != pilot.SOURCE
    assert "experimental" in target.parts
    assert pilot.EOS_WEIGHT == 1.5
    assert pilot.DEFAULT_BUDGET <= 512_000


def test_checkpoint_integrity_requires_full_resume_state():
    payload = {
        "optimizer_state": {"state": {}},
        "scheduler_state": {"global_step": 30500},
        "random_state": {"python": 1, "numpy": 2, "torch_cpu": 3, "torch_cuda": 4},
        "permutation": torch.tensor([1]),
        "tokens_processed": 15_616_000,
        "device_metadata": {"device": "cuda:0"},
        "precision_mode": "fp32",
        "experimental": True,
        "phase": 43,
    }
    assert pilot.verify_payload(payload, end_update=30500, end_tokens=15_616_000)["pass"]
    del payload["random_state"]["torch_cuda"]
    assert not pilot.verify_payload(payload, end_update=30500, end_tokens=15_616_000)["pass"]


def test_generation_comparison_has_requested_decoders():
    assert set(MODES) == {"greedy", "temperature_0.7", "temperature_0.8", "top_p_0.90"}
    assert _topic_retention([1, 2, 9], [2, 3, 4]) == 1 / 3


def test_measured_decision_requires_short_gates_not_architecture_replacement():
    summary = json.loads(
        (pilot.ROOT / "evaluation/foundation-v32-base-maturity-decision-summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["scaling_classification"] == "SLOWING_BUT_HEALTHY"
    assert summary["greedy_attractor"] == "WEAKENING"
    assert summary["architecture_defect_evidence"] is False
    assert summary["next_phase_gate"] == "CONTINUE_SHORT_GPU_GATES"
    assert summary["continue_20m_gpu_permission"] is False
    assert summary["permission_checks"]["context_maintained"] is False
    assert summary["parallel_cpu_evaluation"] == "DISABLED"
    assert summary["foundation_base_complete"] is False


def test_phase43_integrity_and_blind_guards_are_recorded():
    summary = json.loads(
        (pilot.ROOT / "evaluation/foundation-v32-base-maturity-decision-summary.json").read_text(
            encoding="utf-8"
        )
    )
    integrity = summary["checkpoint_integrity"]
    assert integrity["official_15_360m"]["source_unchanged_after_pilot"]
    assert integrity["experimental_pilot"]["integrity"]["pass"]
    assert integrity["final_blind"] == {
        "opened": False,
        "expected_sha256": "fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b",
        "actual_sha256": "fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b",
        "pass": True,
    }
