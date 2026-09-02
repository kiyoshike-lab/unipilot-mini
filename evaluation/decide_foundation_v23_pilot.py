"""Issue the PHASE 34 640k pilot readiness decision before any added training."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> int:
    parity = read("evaluation/foundation-v23-inference-parity.json")
    phase33 = read("evaluation/foundation-v22-summary.json")
    old = read("evaluation/foundation-v23-generation-diagnostics-256000.json")
    new = read("evaluation/foundation-v23-generation-diagnostics-512000.json")
    old_metrics = old["validation_document_prefix"]["metrics"]
    new_metrics = new["validation_document_prefix"]["metrics"]
    old_teacher = old_metrics["teacher_forced_horizon"]["32"]
    new_teacher = new_metrics["teacher_forced_horizon"]["32"]
    old_free = old_metrics["free_running"]
    new_free = new_metrics["free_running"]
    checks = {
        "inference_parity_pass": parity["inference_parity"] == "PASS",
        "kv_cache_parity_pass": parity["kv_cache_parity"] == "PASS",
        "checkpoint_integrity_pass": phase33["checkpoint_verification"]["integrity_pass"],
        "architecture_smoke_pass": phase33["synthetic_smoke"]["gate_pass"],
        "validation_loss_improved": phase33["curve"][-1]["validation_loss"] < phase33["curve"][0]["validation_loss"],
        "validation_top_k_improved": all(
            phase33["curve"][-1][key] > phase33["curve"][0][key]
            for key in ("top_1_accuracy", "top_5_accuracy", "top_10_accuracy")
        ),
        "teacher_forced_horizon_loss_improved": new_teacher["loss"] < old_teacher["loss"],
        "teacher_forced_horizon_top_k_improved": all(
            new_teacher[key] > old_teacher[key]
            for key in ("top_1_accuracy", "top_5_accuracy", "top_10_accuracy")
        ),
        "correct_token_probability_improved": new_teacher["mean_correct_token_probability"] > old_teacher["mean_correct_token_probability"],
        "divergence_not_regressed": new_free["mean_divergence_position"] >= old_free["mean_divergence_position"] - 0.05,
        "repetition_direction_improved": new_free["ngram_repetition"]["1"] < old_free["ngram_repetition"]["1"],
        "frequency_js_direction_improved": new["token_distribution"]["jensen_shannon_divergence_nats"] < old["token_distribution"]["jensen_shannon_divergence_nats"],
        "corpus_exposure_below_two_percent": new["corpus_exposure"]["percentage"] < 2.0,
        "evaluator_threshold_unchanged": not new["natural_japanese_evaluator_audit"]["thresholds_changed"],
        "evaluator_has_50_readable_examples": len(new["natural_japanese_evaluator_audit"]["examples"]) >= 50,
        "residual_training_stable": phase33["gate_evidence"]["training_stable"],
    }
    pilot_allowed = all(checks.values())
    result = {
        "schema": "foundation-v23-pilot-readiness-v1",
        "phase": 34,
        "decision_before_training": "EXECUTE_640K_PILOT" if pilot_allowed else "DO_NOT_EXECUTE_640K_PILOT",
        "pilot_allowed": pilot_allowed,
        "checks": checks,
        "learning_direction": "HEALTHY" if all((
            checks["validation_loss_improved"], checks["validation_top_k_improved"],
            checks["teacher_forced_horizon_loss_improved"],
            checks["teacher_forced_horizon_top_k_improved"],
            checks["correct_token_probability_improved"], checks["divergence_not_regressed"],
        )) else "UNHEALTHY",
        "root_cause_hypothesis": [
            "token-limited undertraining: 512k is only 1.53% of the train corpus",
            "free-running exposure error after an early low-margin error",
            "generated Top-1 frequency collapse, dominated by high-frequency boundary/newline tokens",
            "EOS supervision is scarce (132-167 EOS targets per seed by 512k)",
            "decoding exposes some better candidates but does not establish model improvement",
        ],
        "architecture_changed": False,
        "corpus_changed": False,
        "tokenizer_changed": False,
        "one_million_training_started": False,
    }
    output = ROOT / "evaluation/foundation-v23-pilot-readiness.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pilot_allowed": pilot_allowed, "learning_direction": result["learning_direction"]}, indent=2))
    return 0 if pilot_allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())
