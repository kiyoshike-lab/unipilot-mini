from __future__ import annotations

import json
from pathlib import Path

from training.train_foundation_v21_ab import load_json


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2026)
MILESTONES = (1_024_000, 1_280_000, 1_536_000, 1_792_000, 2_048_000)


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_v25_keeps_current_architecture_and_exact_token_budget() -> None:
    settings = load_json("configs/unipilot-foundation-v25.json")
    assert settings["parameter_target"] == 19_514_880
    assert settings["training"]["token_budget"] == 2_048_000
    assert settings["training"]["milestone_tokens"] == list(MILESTONES)
    assert settings["maximum_allowed_tokens_per_run"] == 2_048_000
    assert settings["training"]["schedule_after_warmup"] == "constant"


def test_v25_preflight_resumes_every_seed_from_exactly_1024k() -> None:
    preflight = read("evaluation/foundation-v25-resume-preflight.json")
    assert all(audit["tokens_processed"] == 1_024_000 for audit in preflight["audits"].values())
    assert all(audit["status"] == "PASS" for audit in preflight["audits"].values())
    assert all(audit["duplicate_data_prevented"] for audit in preflight["audits"].values())


def test_v25_runs_include_all_fixed_milestones_and_outside_top_one_percent_metrics() -> None:
    for seed in SEEDS:
        result = read(f"evaluation/foundation-v25-runs/current-seed-{seed}.json")
        rows = {row["tokens_processed"]: row for row in result["training"]["history"]}
        assert all(tokens in rows for tokens in MILESTONES)
        assert result["final"]["tokens_processed"] == 2_048_000
        assert result["parameters"] == 19_514_880
        validation = rows[2_048_000]["validation"]
        assert "top_1_percent_outside_top_5_accuracy" in validation
        assert "top_1_percent_outside_top_10_accuracy" in validation


def test_v25_generation_diagnostics_reuse_the_fixed_held_out_prefixes() -> None:
    identifiers = []
    for tokens in MILESTONES:
        payload = read(f"evaluation/foundation-v23-generation-diagnostics-{tokens}.json")
        rows = payload["validation_document_prefix"]["items"]
        assert len(rows) == 200
        assert payload["validation_sentence_prefix"]["metrics"]["examples"] == 50
        identifiers.append([row["id"] for row in rows])
    assert all(rows == identifiers[0] for rows in identifiers[1:])


def test_v25_checkpoint_integrity_and_bitwise_resume_reproducibility_pass() -> None:
    verification = read("evaluation/foundation-v25-checkpoint-verification.json")
    assert verification["expected_checkpoints"] == 12
    assert verification["verified_checkpoints"] == 12
    assert verification["integrity_pass"] is True
    assert verification["resume_reproducibility"]["status"] == "PASS"


def test_v25_gate_records_learning_improvement_with_generation_lag() -> None:
    summary = read("evaluation/foundation-v25-summary.json")
    assert summary["gate"] == "CONTINUE_5M_GENERATION_LAG"
    assert summary["language_emergence"] == "PARTIAL"
    assert summary["repetition_trend"] == "SLOW_IMPROVEMENT"
    assert summary["next_token_budget"] == "3M intermediate checkpoint toward 5M"
    assert summary["foundation_base_complete"] is False
    assert summary["final_blind"]["content_opened"] is False
    assert all(summary["gate_checks"][key] for key in (
        "validation_improved", "top_k_improved", "teacher_forced_improved",
        "frequency_learning_improved", "context_maintained", "generation_direction",
        "training_stable", "checkpoints_pass", "resume_reproducibility_pass",
        "synthetic_smoke_pass", "generation_lag",
    ))
