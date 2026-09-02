from __future__ import annotations

import json
from pathlib import Path

from training.train_foundation_v21_ab import load_json


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2026)
MILESTONES = (512_000, 640_000, 768_000, 896_000, 1_024_000)


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_v24_configuration_keeps_current_architecture_and_exact_budget() -> None:
    settings = load_json("configs/unipilot-foundation-v24.json")
    assert settings["variants"] == [{
        "name": "current",
        "residual_projection_init_scale": 1.0,
        "formal_name": "Current",
    }]
    assert settings["parameter_target"] == 19_514_880
    assert settings["training"]["token_budget"] == 1_024_000
    assert settings["training"]["milestone_tokens"] == list(MILESTONES)
    assert settings["training"]["schedule_after_warmup"] == "constant"


def test_v24_resume_preflight_preserves_seed_specific_start_points() -> None:
    preflight = read("evaluation/foundation-v24-resume-preflight.json")
    assert preflight["audits"]["42"]["tokens_processed"] == 640_000
    assert preflight["audits"]["123"]["tokens_processed"] == 512_000
    assert preflight["audits"]["2026"]["tokens_processed"] == 512_000
    assert all(row["status"] == "PASS" for row in preflight["audits"].values())
    assert all(row["duplicate_data_prevented"] for row in preflight["audits"].values())


def test_all_three_v24_runs_have_the_fixed_learning_curve_and_new_outside_metrics() -> None:
    for seed in SEEDS:
        result = read(f"evaluation/foundation-v24-runs/current-seed-{seed}.json")
        rows = {row["tokens_processed"]: row for row in result["training"]["history"]}
        assert all(tokens in rows for tokens in MILESTONES)
        assert result["final"]["tokens_processed"] == 1_024_000
        assert result["parameters"] == 19_514_880
        for tokens in (768_000, 896_000, 1_024_000):
            validation = rows[tokens]["validation"]
            assert "top_1_percent_outside_top_5_accuracy" in validation
            assert "top_1_percent_outside_top_10_accuracy" in validation


def test_v24_generation_diagnostics_keep_identical_held_out_prefixes() -> None:
    identifiers = []
    for tokens in MILESTONES:
        payload = read(f"evaluation/foundation-v23-generation-diagnostics-{tokens}.json")
        rows = payload["validation_document_prefix"]["items"]
        assert len(rows) == 200
        assert payload["validation_sentence_prefix"]["metrics"]["examples"] == 50
        identifiers.append([row["id"] for row in rows])
    assert all(rows == identifiers[0] for rows in identifiers[1:])
    final_rows = read("evaluation/foundation-v23-generation-diagnostics-1024000.json")["validation_document_prefix"]["items"]
    assert all("runaway_onset_token" in row["generation"] for row in final_rows)


def test_v24_checkpoint_integrity_and_bitwise_resume_reproducibility_pass() -> None:
    verification = read("evaluation/foundation-v24-checkpoint-verification.json")
    assert verification["expected_checkpoints"] == 11
    assert verification["verified_checkpoints"] == 11
    assert verification["integrity_pass"] is True
    assert verification["resume_reproducibility"]["status"] == "PASS"
    assert verification["resume_reproducibility"]["bitwise_equal_parameters"] is True


def test_v24_frequency_punctuation_context_and_smoke_evidence_are_complete() -> None:
    summary = read("evaluation/foundation-v24-summary.json")
    final = summary["training_curve"][-1]
    assert set(final["frequency_buckets"]) == {
        "top_1_percent", "top_5_percent_excluding_top_1",
        "top_20_percent_excluding_top_5", "middle_20_to_80_percent",
        "rare_bottom_20_percent",
    }
    assert set(final["outside_top_1_percent"]) == {
        "top_1_accuracy", "top_5_accuracy", "top_10_accuracy",
    }
    assert set(summary["generation_curve"][-1]["punctuation_distribution"]) == {
        "。", "、", "の", "に", "は", "を", "が", "と", "で",
    }
    assert summary["gate_checks"]["context_maintained"] is True
    assert summary["synthetic_smoke"]["gate_pass"] is True


def test_v24_final_gate_records_generation_lag_without_claiming_completion() -> None:
    summary = read("evaluation/foundation-v24-summary.json")
    assert summary["gate"] == "CONTINUE_2M_GENERATION_LAG"
    assert summary["one_point_024m"] == "PASS"
    assert summary["language_emergence"] == "PARTIAL"
    assert summary["next_token_budget"] == "2M"
    assert summary["full_training_continuation"] == "YES"
    assert summary["foundation_base_complete"] is False
    assert summary["final_blind"]["content_opened"] is False
