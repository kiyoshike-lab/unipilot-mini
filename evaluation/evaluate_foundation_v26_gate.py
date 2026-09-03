"""Evaluate the required PHASE 37 3.072M intermediate gate before 5.120M."""
from __future__ import annotations

import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2026)


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def mean(values) -> float:
    return statistics.fmean(float(value) for value in values)


def row_at(run: dict, tokens: int) -> dict:
    return next(row for row in run["training"]["history"] if int(row["tokens_processed"]) == tokens)


def main() -> int:
    baseline = read("evaluation/foundation-v25-summary.json")
    runs = [read(f"evaluation/foundation-v26-gate-runs/current-seed-{seed}.json") for seed in SEEDS]
    rows_2048 = [row_at(run, 2_048_000) for run in runs]
    rows_3072 = [row_at(run, 3_072_000) for run in runs]
    diag_2048 = read("evaluation/foundation-v23-generation-diagnostics-2048000.json")
    diag_3072 = read("evaluation/foundation-v23-generation-diagnostics-3072000.json")
    sample_2048 = diag_2048["decoding_comparison"]["temperature_0.7"]
    sample_3072 = diag_3072["decoding_comparison"]["temperature_0.7"]
    greedy_2048 = diag_2048["validation_document_prefix"]["metrics"]["free_running"]
    greedy_3072 = diag_3072["validation_document_prefix"]["metrics"]["free_running"]
    metrics = {
        "loss_2048": mean(row["validation"]["loss"] for row in rows_2048),
        "loss_3072": mean(row["validation"]["loss"] for row in rows_3072),
        "top_1_2048": mean(row["validation"]["top_1_accuracy"] for row in rows_2048),
        "top_1_3072": mean(row["validation"]["top_1_accuracy"] for row in rows_3072),
        "top_5_2048": mean(row["validation"]["top_5_accuracy"] for row in rows_2048),
        "top_5_3072": mean(row["validation"]["top_5_accuracy"] for row in rows_3072),
        "top_10_2048": mean(row["validation"]["top_10_accuracy"] for row in rows_2048),
        "top_10_3072": mean(row["validation"]["top_10_accuracy"] for row in rows_3072),
        "sampling_naturalness_2048": sample_2048["natural_japanese_proxy"],
        "sampling_naturalness_3072": sample_3072["natural_japanese_proxy"],
        "sampling_semantic_2048": sample_2048["semantic_local_syntax_proxy"],
        "sampling_semantic_3072": sample_3072["semantic_local_syntax_proxy"],
        "sampling_repetition_3gram_2048": sample_2048["mean_repetition_3gram"],
        "sampling_repetition_3gram_3072": sample_3072["mean_repetition_3gram"],
        "greedy_repetition_1gram_2048": greedy_2048["ngram_repetition"]["1"],
        "greedy_repetition_1gram_3072": greedy_3072["ngram_repetition"]["1"],
        "context_advantage_3072_min": min(row["context_utilization"]["full_vs_last_1_loss_advantage"] for row in rows_3072),
    }
    checks = {
        "validation_improved": metrics["loss_3072"] < metrics["loss_2048"],
        "top_k_maintained_or_improved": all(metrics[f"top_{k}_3072"] >= metrics[f"top_{k}_2048"] for k in (1, 5, 10)),
        "teacher_forced_improved": diag_3072["validation_document_prefix"]["metrics"]["teacher_forced_horizon"]["32"]["top_10_accuracy"] > diag_2048["validation_document_prefix"]["metrics"]["teacher_forced_horizon"]["32"]["top_10_accuracy"],
        "frequency_learning_improved": mean(row["validation"]["top_1_percent_outside_top_10_accuracy"] for row in rows_3072) >= mean(row["validation"]["top_1_percent_outside_top_10_accuracy"] for row in rows_2048),
        "context_maintained": metrics["context_advantage_3072_min"] > 0,
        "sampling_not_majorly_regressed": metrics["sampling_naturalness_3072"] >= metrics["sampling_naturalness_2048"] - 0.10 and metrics["sampling_semantic_3072"] >= metrics["sampling_semantic_2048"] - 0.10,
        "repetition_not_majorly_worse": metrics["greedy_repetition_1gram_3072"] <= metrics["greedy_repetition_1gram_2048"] + 0.02,
        "checkpoints_present": all((ROOT / f"checkpoints/foundation-v26-current/current/seed-{seed}/checkpoint-tokens-3072000.pt").is_file() for seed in SEEDS),
    }
    outcome = "CONTINUE_TO_5M" if all(checks.values()) else "INVESTIGATE_AT_3M"
    result = {"schema": "foundation-v26-intermediate-gate-v1", "phase": 37, "gate_tokens": 3_072_000, "metrics": metrics, "checks": checks, "outcome": outcome, "baseline_gate": baseline["gate"], "production_changed": False, "final_blind_used": False}
    (ROOT / "evaluation/foundation-v26-intermediate-gate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"outcome": outcome, "checks": checks}, indent=2))
    return 0 if outcome == "CONTINUE_TO_5M" else 2


if __name__ == "__main__":
    raise SystemExit(main())
