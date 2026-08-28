from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_v13_config_is_strictly_scratch_250_and_forbids_later_phases():
    config = load("configs/unipilot-foundation-v13.json")
    assert config["initialization"] == "scratch-no-resume-no-pretrained-model"
    assert config["max_steps"] == config["schedule_steps"] == 250
    assert config["warmup_steps"] == 20
    assert config["learning_rate"] == 1e-4
    assert config["checkpoint_steps"] == [50, 100, 150, 200, 250]
    assert not any(config[key] for key in (
        "foundation_500_enabled", "foundation_1000_enabled", "standard_46m_enabled",
        "campus_pretraining_enabled", "instruction_tuning_enabled", "dpo_enabled",
    ))


def test_v13_training_curve_has_all_required_metrics_and_is_scratch():
    report = load("evaluation/foundation-v13-training-curve.json")
    expected = [0, 10, 20, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    assert report["status"] == "COMPLETED"
    assert report["scratch_start"] is True
    assert report["resumed_from"] is None
    assert [row["step"] for row in report["history"]] == expected
    assert report["max_steps"] == 250
    assert report["parameters"] == 19_514_880
    assert not report["nan_or_inf"] and not report["diverged"]
    for row in report["history"]:
        assert {"validation_loss", "perplexity", "learning_rate", "peak_ram_mb",
                "tokens_processed", "corpus_percentage", "epoch_equivalent"} <= row.keys()


def test_all_v13_checkpoints_pass_integrity_and_match_sha256():
    report = load("evaluation/foundation-v13-checkpoint-integrity.json")
    assert report["all_checkpoints"] == "PASS"
    assert report["expected_steps"] == [50, 100, 150, 200, 250]
    for row in report["results"]:
        assert row["status"] == "PASS"
        assert all(row["checks"].values())
        assert sha256(ROOT / row["path"]) == row["sha256"]


def test_v13_generation_uses_fixed_50_completion_prompts_without_penalty():
    report = load("evaluation/foundation-v13-generation.json")
    assert report["steps"] == [0, 50, 100, 150, 200, 250]
    assert report["prompts"] == 50
    assert report["repetition_penalty_used_in_base_evaluation"] is False
    assert report["final_blind_used"] is False
    for row in report["results"]:
        assert set(row["modes"]) == {
            "greedy_no_penalty", "sampling_t07_topk40_topp09_no_penalty"
        }
        assert all(mode["metrics"]["prompts"] == 50 for mode in row["modes"].values())


def test_v13_resume_roundtrip_and_summary_protection_pass():
    resume = load("evaluation/foundation-v13-resume-reproducibility.json")
    summary = load("evaluation/foundation-v13-summary.json")
    assert resume["resume_integrity"] == "PASS"
    assert all(resume["checks"].values())
    assert summary["verification"]["checkpoint_integrity"] == "PASS"
    assert summary["verification"]["resume_reproducibility"] == "PASS"
    assert summary["verification"]["tokenizer_roundtrip"] == "PASS"
    assert summary["gate"]["status"] in {"CONTINUE", "INVESTIGATE", "STOP"}
    assert summary["decisions"]["continue_to_500"] == (
        "YES" if summary["gate"]["status"] == "CONTINUE" else "NO"
    )
    assert summary["decisions"]["foundation_500_executed"] is False
    assert summary["decisions"]["foundation_1000_executed"] is False
    assert summary["protected"]["final_blind_content_opened"] is False
    assert summary["protected"]["final_blind_sha256"] == (
        "fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b"
    )
