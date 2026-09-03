from __future__ import annotations

import json
from pathlib import Path

from training.train_foundation_v21_ab import load_json


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2026)
MILESTONES = (2_048_000, 2_560_000, 3_072_000, 3_584_000, 4_096_000, 4_608_000, 5_120_000)


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_v26_keeps_current_architecture_and_hard_5120k_cap() -> None:
    settings = load_json("configs/unipilot-foundation-v26.json")
    assert settings["parameter_target"] == 19_514_880
    assert settings["training"]["token_budget"] == 5_120_000
    assert settings["training"]["milestone_tokens"] == list(MILESTONES)
    assert settings["maximum_allowed_tokens_per_run"] == 5_120_000


def test_v26_preflights_pass_before_each_resume_stage() -> None:
    for path, expected_tokens in (("evaluation/foundation-v26-gate-resume-preflight.json", 2_048_000), ("evaluation/foundation-v26-final-resume-preflight.json", 3_072_000)):
        payload = read(path)
        assert all(audit["tokens_processed"] == expected_tokens for audit in payload["audits"].values())
        assert all(audit["status"] == "PASS" for audit in payload["audits"].values())


def test_v26_intermediate_gate_authorized_the_5120k_continuation() -> None:
    gate = read("evaluation/foundation-v26-intermediate-gate.json")
    assert gate["outcome"] == "CONTINUE_TO_5M"
    assert all(gate["checks"].values())


def test_v26_runs_cover_the_fixed_checkpoints_for_all_seeds() -> None:
    for seed in SEEDS:
        result = read(f"evaluation/foundation-v26-runs/current-seed-{seed}.json")
        rows = {int(row["tokens_processed"]): row for row in result["training"]["history"]}
        assert all(tokens in rows for tokens in MILESTONES)
        assert result["final"]["tokens_processed"] == 5_120_000
        assert result["parameters"] == 19_514_880


def test_v26_generation_diagnostics_keep_fixed_validation_prefixes() -> None:
    identifiers = []
    for tokens in MILESTONES:
        payload = read(f"evaluation/foundation-v23-generation-diagnostics-{tokens}.json")
        rows = payload["validation_document_prefix"]["items"]
        assert len(rows) == 200
        assert payload["validation_sentence_prefix"]["metrics"]["examples"] == 50
        identifiers.append([row["id"] for row in rows])
    assert all(rows == identifiers[0] for rows in identifiers[1:])


def test_v26_final_evidence_keeps_base_incomplete_and_final_blind_closed() -> None:
    verification = read("evaluation/foundation-v26-checkpoint-verification.json")
    summary = read("evaluation/foundation-v26-summary.json")
    assert verification["expected_checkpoints"] == 18
    assert verification["verified_checkpoints"] == 18
    assert verification["integrity_pass"] is True
    assert verification["resume_reproducibility"]["status"] == "PASS"
    assert summary["gate"] in {"CONTINUE_10M", "CONTINUE_10M_GENERATION_LAG", "GENERATION_PLATEAU_INVESTIGATE"}
    assert summary["foundation_base_complete"] is False
    assert summary["final_blind"]["content_opened"] is False
