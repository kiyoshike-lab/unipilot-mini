from __future__ import annotations

from pathlib import Path

from training.train_foundation_v21_ab import load_json, stateless_scheduler_state
from training.train_foundation_v22_current import default_resume_path, preflight_resume


ROOT = Path(__file__).resolve().parents[1]


def test_v22_current_configuration_is_the_selected_20m_architecture() -> None:
    settings = load_json(ROOT / "configs/unipilot-foundation-v22.json")
    assert settings["variants"] == [{
        "name": "current",
        "residual_projection_init_scale": 1.0,
        "formal_name": "Current",
    }]
    assert settings["parameter_target"] == 19_514_880
    assert settings["training"]["token_budget"] == 512_000
    assert settings["training"]["milestone_tokens"] == [256_000, 320_000, 384_000, 448_000, 512_000]


def test_v22_stateless_schedule_stays_constant_after_the_existing_warmup() -> None:
    settings = load_json(ROOT / "configs/unipilot-foundation-v22.json")
    assert stateless_scheduler_state(settings, 500)["learning_rate"] == 1e-4
    assert stateless_scheduler_state(settings, 501)["learning_rate"] == 1e-4


def test_v22_preflight_accepts_all_verified_v21_current_checkpoints() -> None:
    settings = load_json(ROOT / "configs/unipilot-foundation-v22.json")
    for seed in settings["seeds"]:
        audit = preflight_resume(settings, seed, default_resume_path(seed))
        assert audit["status"] == "PASS"
        assert audit["tokens_processed"] == 256_000
        assert audit["next_update"] == 501
        assert audit["scheduler_state_source"] == "derived_legacy_stateless_schedule"
