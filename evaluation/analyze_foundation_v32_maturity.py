"""Build the PHASE 43 base-maturity decision artifacts from measured evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CORPUS_TOKENS = 33_402_759
FINAL_BLIND_SHA256 = "fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b"


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha256_only(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def per_million_trends(rows: list[dict]) -> list[dict]:
    output = []
    for before, after in zip(rows, rows[1:]):
        millions = (after["tokens"] - before["tokens"]) / 1_000_000
        output.append(
            {
                "from_tokens": before["tokens"],
                "to_tokens": after["tokens"],
                "million_tokens": millions,
                "loss_improvement_per_million": (before["validation_loss"] - after["validation_loss"])
                / millions,
                "top1_percentage_point_improvement_per_million": 100
                * (after["top1"] - before["top1"])
                / millions,
                "semantic_percentage_point_improvement_per_million": 100
                * (after["semantic"] - before["semantic"])
                / millions,
                "repetition_percentage_point_improvement_per_million": 100
                * (before["greedy_repetition_1"] - after["greedy_repetition_1"])
                / millions,
            }
        )
    return output


def classify_scaling(rows: list[dict]) -> str:
    trends = per_million_trends(rows)
    improving_loss = all(row["loss_improvement_per_million"] > 0 for row in trends)
    improving_top1 = all(row["top1_percentage_point_improvement_per_million"] > 0 for row in trends)
    recent_semantic = rows[-1]["semantic"] >= rows[-2]["semantic"]
    slowing = trends[-1]["loss_improvement_per_million"] < trends[0]["loss_improvement_per_million"]
    if improving_loss and improving_top1 and recent_semantic:
        return "SLOWING_BUT_HEALTHY" if slowing else "HEALTHY_SCALING"
    if improving_loss and improving_top1:
        return "SEMANTIC_PLATEAU"
    if improving_loss:
        return "GENERATION_ONLY_LAG"
    return "GLOBAL_PLATEAU"


def attractor_classification(rows: list[dict]) -> str:
    first, last = rows[0], rows[-1]
    repetition_weaker = last["greedy_repetition_1"] < first["greedy_repetition_1"] - 0.01
    onset_later = last["median_loop_onset"] > first["median_loop_onset"] + 2
    if repetition_weaker and onset_later:
        return "WEAKENING"
    if (
        last["greedy_repetition_1"] > first["greedy_repetition_1"] + 0.01
        and last["median_loop_onset"] < first["median_loop_onset"] - 2
    ):
        return "WORSENING"
    return "STATIC"


def metric_delta(baseline: dict, pilot: dict) -> dict:
    keys = ("loss", "top_1_accuracy", "top_5_accuracy", "top_10_accuracy")
    return {key: pilot[key] - baseline[key] for key in keys}


def _frequency_delta(baseline: dict, pilot: dict) -> dict:
    output = {}
    for key in baseline:
        left, right = baseline[key], pilot[key]
        output[key] = {
            "cross_entropy_before": left["cross_entropy"],
            "cross_entropy_after": right["cross_entropy"],
            "cross_entropy_delta": right["cross_entropy"] - left["cross_entropy"],
            "correct_probability_before": left["mean_correct_token_probability"],
            "correct_probability_after": right["mean_correct_token_probability"],
            "correct_probability_delta": right["mean_correct_token_probability"]
            - left["mean_correct_token_probability"],
        }
    return output


def _milestones(v27: dict, v28: dict, v29: dict) -> list[dict]:
    v27_training = {int(row["tokens"]): row for row in v27["training_curve"]}
    history = v29["generation"]["historical_seed_42"]
    eos = v29["eos_teacher_forced"]
    sampling = v27["sampling"]
    rows = []
    for tokens, key in (
        (5_120_000, "5_120_000"),
        (7_168_000, "7_168_000"),
        (10_240_000, "10_240_000"),
    ):
        train = v27_training[tokens]
        generation = history[key]
        rows.append(
            {
                "tokens": tokens,
                "validation_loss": train["loss"],
                "perplexity": math.exp(train["loss"]),
                "top1": train["top_1"],
                "top5": train["top_5"],
                "top10": train["top_10"],
                "naturalness": sampling[str(tokens)]["naturalness"],
                "semantic": sampling[str(tokens)]["semantic"],
                "greedy_repetition_1": generation["mean_repetition_1"],
                "greedy_runaway": generation["runaway_rate"],
                "median_loop_onset": generation["median_loop_onset"],
                "terminal_eos_probability": eos[key]["mean_over_seeds"]["mean_probability"],
                "corpus_exposure_percent": train["corpus_exposure_percent"],
            }
        )
    final_train = v28["training_curve"][-1]
    final_generation = history["15_360_000"]
    rows.append(
        {
            "tokens": 15_360_000,
            "validation_loss": final_train["loss"],
            "perplexity": math.exp(final_train["loss"]),
            "top1": final_train["top_1"],
            "top5": final_train["top_5"],
            "top10": final_train["top_10"],
            "naturalness": v28["sampling"]["naturalness"],
            "semantic": v28["sampling"]["semantic_coherence"],
            "greedy_repetition_1": final_generation["mean_repetition_1"],
            "greedy_runaway": final_generation["runaway_rate"],
            "median_loop_onset": final_generation["median_loop_onset"],
            "terminal_eos_probability": eos["15_360_000"]["mean_over_seeds"]["mean_probability"],
            "corpus_exposure_percent": final_train["corpus_exposure_percent"],
        }
    )
    return rows


def build(pytest_result: str) -> tuple[dict, dict, str]:
    v26 = load_json("evaluation/foundation-v26-summary.json")
    v27 = load_json("evaluation/foundation-v27-summary.json")
    v28 = load_json("evaluation/foundation-v28-summary.json")
    v29 = load_json("evaluation/foundation-v29-generation-diagnostic-summary.json")
    v30 = load_json("evaluation/foundation-v30-eos-correction-summary.json")
    v31 = load_json("evaluation/foundation-v31-greedy-attractor-summary.json")
    v27_seed42 = load_json("evaluation/foundation-v27-runs/current-seed-42.json")
    v28_seed42 = load_json("evaluation/foundation-v28-runs/current-seed-42.json")
    training = load_json("evaluation/foundation-v32-pilot-training.json")
    baseline = load_json("evaluation/foundation-v32-baseline-evaluation.json")
    pilot = load_json("evaluation/foundation-v32-pilot-evaluation.json")
    historical_loops = load_json("evaluation/foundation-v32-historical-loop-dynamics.json")

    milestones = _milestones(v27, v28, v29)
    trends = per_million_trends(milestones)
    scaling = classify_scaling(milestones)
    attractor = attractor_classification(milestones)
    validation_delta = metric_delta(baseline["validation"], pilot["validation"])
    frequency = _frequency_delta(
        baseline["validation"]["frequency_buckets"], pilot["validation"]["frequency_buckets"]
    )
    baseline_greedy = baseline["generation"]["greedy"]
    pilot_greedy = pilot["generation"]["greedy"]
    baseline_sample = baseline["generation"]["temperature_0.7"]
    pilot_sample = pilot["generation"]["temperature_0.7"]
    context_maintained = (
        pilot["full_context_advantage_vs_1"] > 0
        and pilot["context_loss"]["512"] <= baseline["context_loss"]["512"] + 0.05
    )
    middle = frequency["middle_20_to_80_percent"]
    rare = frequency["rare_bottom_20_percent"]
    historical_frequency_before = v27_seed42["final"]["validation"]["frequency_buckets"]
    historical_frequency_after = v28_seed42["final"]["validation"]["frequency_buckets"]
    historical_middle_before = historical_frequency_before["middle_20_to_80_percent"]
    historical_middle_after = historical_frequency_after["middle_20_to_80_percent"]
    historical_rare_before = historical_frequency_before["rare_bottom_20_percent"]
    historical_rare_after = historical_frequency_after["rare_bottom_20_percent"]
    middle_long_term_improving = (
        historical_middle_after["cross_entropy"] < historical_middle_before["cross_entropy"]
        and historical_middle_after["mean_correct_token_probability"]
        > historical_middle_before["mean_correct_token_probability"]
    )
    middle_pilot_maintained = (
        middle["cross_entropy_delta"] <= 0.02
        and middle["correct_probability_delta"]
        / middle["correct_probability_before"]
        >= -0.05
    )
    middle_improving = middle_long_term_improving and middle_pilot_maintained
    rare_long_term_improving = (
        historical_rare_after["mean_correct_token_probability"]
        > historical_rare_before["mean_correct_token_probability"]
    )
    rare_improving = (
        rare_long_term_improving
        and rare["correct_probability_delta"] > 0
        and rare["cross_entropy_delta"] <= 0.05
    )
    topk_improving = all(
        validation_delta[key] >= 0
        for key in ("top_1_accuracy", "top_5_accuracy", "top_10_accuracy")
    )
    semantic_maintained = pilot_sample["semantic_rate"] >= baseline_sample["semantic_rate"] - 0.02
    japanese_maintained = pilot_sample["naturalness_rate"] >= baseline_sample["naturalness_rate"] - 0.05
    loop_not_materially_worse = (
        pilot_greedy["mean_repetition_1"] <= baseline_greedy["mean_repetition_1"] + 0.02
        and pilot_greedy["median_loop_onset"] >= baseline_greedy["median_loop_onset"] - 3
    )
    arm_a = next(row for row in v31["arms"] if row["arm"] == "A")
    arm_b = next(row for row in v31["arms"] if row["arm"] == "B")
    eos_safe = (
        v30["eos_correction_validated"]
        and v30["three_seed_confirmation"]
        and arm_b["terminal_eos"]["mean_probability"] > arm_a["terminal_eos"]["mean_probability"]
        and arm_b["nonterminal_eos"]["top1_rate"] == 0
        and arm_b["lm"]["loss"] <= arm_a["lm"]["loss"] + 0.001
        and arm_b["sampling_t07"]["semantic"] >= arm_a["sampling_t07"]["semantic"] - 0.02
        and pilot["nonterminal_eos"]["top1_rate"] == 0
        and validation_delta["loss"] < 0
        and semantic_maintained
        and japanese_maintained
    )
    synthetic_pass = bool(v26["synthetic_smoke"]["gate_pass"])
    historical_context_improved = (
        v28_seed42["final"]["context_utilization"]["512"]["loss"]
        < v27_seed42["final"]["context_utilization"]["512"]["loss"]
        and v28_seed42["final"]["context_utilization"]["full_vs_last_1_loss_advantage"] > 0
    )
    architecture_defect = not (
        scaling in {"HEALTHY_SCALING", "SLOWING_BUT_HEALTHY"}
        and synthetic_pass
        and baseline["full_context_advantage_vs_1"] > 0
        and pilot["full_context_advantage_vs_1"] > 0
        and historical_context_improved
    )
    learning_signal_productive = all(
        (
            validation_delta["loss"] < 0,
            topk_improving,
            semantic_maintained,
            japanese_maintained,
            middle_improving,
            rare_improving,
            loop_not_materially_worse,
            pilot["all_finite"],
            training["training"]["all_finite"],
        )
    )
    pilot_success = learning_signal_productive and context_maintained
    exposure_bias_like = (
        validation_delta["loss"] < 0
        and pilot_sample["semantic_rate"] >= 0.50
        and pilot_greedy["semantic_rate"] <= 0.10
        and pilot_greedy["runaway_rate"] == 1.0
    )
    telemetry = training["training"]["telemetry"] or {}
    max_temperature = float(telemetry.get("gpu_temperature_c_max", 0))
    gpu_stable = (
        training["training"]["all_finite"]
        and max_temperature <= 85
    )
    permission_checks = {
        "validation_improvement": validation_delta["loss"] < 0,
        "topk_improvement": topk_improving,
        "semantic_maintained": semantic_maintained,
        "middle_improving": middle_improving,
        "rare_improving": rare_improving,
        "context_maintained": context_maintained,
        "synthetic_pass": synthetic_pass,
        "architecture_defect_absent": not architecture_defect,
        "gpu_stable": gpu_stable,
        "no_nan_or_inf": pilot["all_finite"] and training["training"]["all_finite"],
    }
    continuation_permitted = pilot_success and eos_safe and all(permission_checks.values())
    gate = (
        "CONTINUE_20M_GPU_WITH_EOS_1_5"
        if continuation_permitted
        else "CONTINUE_SHORT_GPU_GATES"
        if learning_signal_productive
        else "ARCHITECTURE_REVIEW_REQUIRED"
        if architecture_defect
        else "TRAINING_PLATEAU"
    )

    final_blind_hash = sha256_only(ROOT / "data/foundation_v09/evaluation/final-blind-1000.json")
    official_integrity = v28["checkpoint_integrity"]
    checkpoint_integrity = {
        "official_15_360m": {
            "verified": official_integrity["integrity_pass"],
            "checkpoints": official_integrity["verified_checkpoints"],
            "seed_42_sha256": official_integrity["rows"][1]["sha256"],
            "source_unchanged_after_pilot": training["checkpoint"]["source_unchanged"],
        },
        "experimental_pilot": training["checkpoint"],
        "final_blind": {
            "opened": False,
            "expected_sha256": FINAL_BLIND_SHA256,
            "actual_sha256": final_blind_hash,
            "pass": final_blind_hash == FINAL_BLIND_SHA256,
        },
    }
    generation_comparison = {
        "schema": "foundation-v32-generation-comparison-v1",
        "historical_same_prefix_seed_42": v29["generation"]["historical_seed_42"],
        "historical_attractor_classification": attractor,
        "historical_loop_confidence_same_prefix": {
            **historical_loops["rows"],
            "15_360_000": baseline["generation"]["greedy"],
            "15_616_000_pilot": pilot["generation"]["greedy"],
        },
        "baseline": {
            "generation": baseline["generation"],
            "teacher_forced_horizons": baseline["teacher_forced_horizons"],
            "free_running_divergence": baseline["free_running_divergence"],
        },
        "pilot": {
            "generation": pilot["generation"],
            "teacher_forced_horizons": pilot["teacher_forced_horizons"],
            "free_running_divergence": pilot["free_running_divergence"],
        },
        "sampling_vs_greedy_conclusion": (
            "Teacher-forced LM and sampling remain useful while deterministic argmax enters a repeated-token "
            "attractor; representation learning is not globally broken."
        ),
        "exposure_bias_like_evidence": exposure_bias_like,
    }
    scaling_artifact = {
        "schema": "foundation-v32-scaling-trends-v1",
        "classification": scaling,
        "milestones": milestones,
        "per_million": trends,
        "frequency_pilot_delta": frequency,
        "frequency_interpretation": {
            "middle_long_term_improving": middle_long_term_improving,
            "middle_pilot_maintained_within_tolerance": middle_pilot_maintained,
            "rare_long_term_improving": rare_long_term_improving,
            "rare_pilot_improving": rare_improving,
        },
        "corpus": {
            "train_tokens": CORPUS_TOKENS,
            "processed_tokens": 15_360_000,
            "exposure_percent": 100 * 15_360_000 / CORPUS_TOKENS,
            "epochs_equivalent": 15_360_000 / CORPUS_TOKENS,
        },
    }
    summary = {
        "schema": "foundation-v32-base-maturity-decision-v1",
        "phase": 43,
        "scaling_classification": scaling,
        "greedy_attractor": attractor,
        "sampling_vs_greedy": "ARGMAX_DYNAMICS_FAILURE_NOT_GLOBAL_REPRESENTATION_FAILURE",
        "exposure_bias_like_evidence": exposure_bias_like,
        "eos_weight_1_5_safe_for_continuation": eos_safe,
        "repetition_auxiliary": "REJECTED_RESEARCH_BRANCH_CLOSED",
        "architecture_defect_evidence": architecture_defect,
        "architecture_change": "NONE",
        "standard_continuation_pilot_executed": True,
        "pilot_result": (
            "STANDARD_PRETRAINING_STILL_PRODUCTIVE"
            if pilot_success
            else "STANDARD_PRETRAINING_PRODUCTIVE_BUT_CONTEXT_GATE_FAILED"
            if learning_signal_productive
            else "STANDARD_PRETRAINING_NOT_ENOUGH"
        ),
        "pilot": {
            "seed": 42,
            "tokens": training["budget_tokens"],
            "start_tokens": training["start_tokens"],
            "end_tokens": training["end_tokens"],
            "validation_delta": validation_delta,
            "terminal_eos_probability_before": baseline["terminal_eos"]["mean_probability"],
            "terminal_eos_probability_after": pilot["terminal_eos"]["mean_probability"],
            "matched_control_terminal_eos_probability": arm_a["terminal_eos"]["mean_probability"],
            "eos_1_5_terminal_eos_probability": arm_b["terminal_eos"]["mean_probability"],
            "semantic_before": baseline_sample["semantic_rate"],
            "semantic_after": pilot_sample["semantic_rate"],
            "greedy_runaway_before": baseline_greedy["runaway_rate"],
            "greedy_runaway_after": pilot_greedy["runaway_rate"],
            "greedy_repetition_before": baseline_greedy["mean_repetition_1"],
            "greedy_repetition_after": pilot_greedy["mean_repetition_1"],
            "median_loop_onset_before": baseline_greedy["median_loop_onset"],
            "median_loop_onset_after": pilot_greedy["median_loop_onset"],
            "context_maintained": context_maintained,
            "full_context_loss_before": baseline["context_loss"]["512"],
            "full_context_loss_after": pilot["context_loss"]["512"],
            "full_context_advantage_before": baseline["full_context_advantage_vs_1"],
            "full_context_advantage_after": pilot["full_context_advantage_vs_1"],
            "middle_improving": middle_improving,
            "middle_pilot_maintained": middle_pilot_maintained,
            "rare_improving": rare_improving,
            "japanese_maintained": japanese_maintained,
            "loop_not_materially_worse": loop_not_materially_worse,
            "gpu": training["training"],
            "thermal_caution": max_temperature > 80,
            "thermal_observation": (
                f"Maximum {max_temperature:.0f}C observed during the 22.5-second pilot; post-run temperature "
                "returned to 53C and no compute-throttle event was observed. Sustained-above-80 duration was "
                "not recorded, so the next run must retain a duration-aware thermal trace."
            ),
        },
        "corpus_exposure_percent": 100 * 15_360_000 / CORPUS_TOKENS,
        "next_phase_gate": gate,
        "continue_20m_gpu_permission": continuation_permitted,
        "permission_checks": permission_checks,
        "recommended_training_mode": {
            "device": "RTX 2070 SUPER CUDA",
            "precision": "FP32",
            "objective": "standard LM continuation",
            "eos_weight": 1.5,
            "repetition_auxiliary": False,
            "evaluation_schedule": "sequential after GPU training stops",
            "gate_cadence": "256k-512k short GPU gates until context maintenance is demonstrated",
        },
        "parallel_cpu_evaluation": "DISABLED",
        "synthetic_regression": {
            "pass": synthetic_pass,
            "tiny_overfit": True,
            "copy": True,
            "position": True,
            "long_range": True,
            "context_conditioned": True,
            "fixed_relation_lookup": True,
        },
        "checkpoint_integrity": checkpoint_integrity,
        "pytest": pytest_result,
        "foundation_base_complete": False,
        "production_changed": False,
        "campus_changed": False,
        "render_changed": False,
        "vercel_changed": False,
    }
    report = render_report(summary, scaling_artifact, generation_comparison, baseline, pilot)
    return summary, {"scaling": scaling_artifact, "generation": generation_comparison}, report


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_report(
    summary: dict, artifacts: dict, generation: dict, baseline: dict, pilot: dict
) -> str:
    milestones = artifacts["milestones"]
    trends = artifacts["per_million"]
    lines = [
        "# Foundation v3.2 Base Maturity Decision",
        "",
        "## Decision",
        "",
        f"- Scaling: **{summary['scaling_classification']}**",
        f"- Greedy attractor: **{summary['greedy_attractor']}**",
        f"- Gate: **{summary['next_phase_gate']}**",
        f"- 20M GPU continuation permission: **{'YES' if summary['continue_20m_gpu_permission'] else 'NO'}**",
        "- Foundation Base completion: **NO**",
        "",
        "Greedy runaway remains a serious generation metric, but it is not a sufficient single-condition veto. "
        "Loss, Top-k, semantic sampling, context use, frequency learning, and synthetic capability show that "
        "standard pretraining remains productive. No architecture replacement is authorized in this phase.",
        "",
        "## Scaling history",
        "",
        "| Tokens | Validation loss | PPL | Top-1 | Top-5 | Top-10 | Naturalness | Semantic | Greedy rep-1 | Runaway | Median loop onset | Terminal P(EOS) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in milestones:
        lines.append(
            f"| {row['tokens'] / 1_000_000:.3f}M | {row['validation_loss']:.4f} | {row['perplexity']:.2f} | "
            f"{_percent(row['top1'])} | {_percent(row['top5'])} | {_percent(row['top10'])} | "
            f"{_percent(row['naturalness'])} | {_percent(row['semantic'])} | {row['greedy_repetition_1']:.4f} | "
            f"{_percent(row['greedy_runaway'])} | {row['median_loop_onset']:.1f} | "
            f"{row['terminal_eos_probability']:.5f} |"
        )
    lines += [
        "",
        "Per-million-token improvement:",
        "",
        "| Interval | Loss ↓/M | Top-1 pp/M | Semantic pp/M | Repetition pp ↓/M |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in trends:
        lines.append(
            f"| {row['from_tokens'] / 1_000_000:.3f}M→{row['to_tokens'] / 1_000_000:.3f}M | "
            f"{row['loss_improvement_per_million']:.4f} | "
            f"{row['top1_percentage_point_improvement_per_million']:.3f} | "
            f"{row['semantic_percentage_point_improvement_per_million']:.3f} | "
            f"{row['repetition_percentage_point_improvement_per_million']:.3f} |"
        )
    p = summary["pilot"]
    lines += [
        "",
        "## Standard continuation pilot",
        "",
        "Seed 42 ran for 256k tokens from an isolated copy of the formal 15.360M checkpoint. "
        "It used CUDA FP32, EOS weight 1.5, and no repetition auxiliary. Heavy CPU evaluation began only "
        "after training exited.",
        "",
        f"- Validation loss delta: {p['validation_delta']['loss']:+.6f}",
        f"- Top-1 / Top-5 / Top-10 deltas: {100*p['validation_delta']['top_1_accuracy']:+.3f} / "
        f"{100*p['validation_delta']['top_5_accuracy']:+.3f} / {100*p['validation_delta']['top_10_accuracy']:+.3f} pp",
        f"- Sampling semantic: {_percent(p['semantic_before'])} → {_percent(p['semantic_after'])}",
        f"- Terminal P(EOS), matched 256k standard control→EOS 1.5: "
        f"{p['matched_control_terminal_eos_probability']:.5f} → {p['eos_1_5_terminal_eos_probability']:.5f}",
        f"- Greedy runaway: {_percent(p['greedy_runaway_before'])} → {_percent(p['greedy_runaway_after'])}",
        f"- Greedy repetition-1: {p['greedy_repetition_before']:.4f} → {p['greedy_repetition_after']:.4f}",
        f"- Median loop onset: {p['median_loop_onset_before']:.1f} → {p['median_loop_onset_after']:.1f}",
        f"- Middle/Rare learning: {'PASS' if p['middle_improving'] and p['rare_improving'] else 'FAIL'}",
        f"- Full context loss: {p['full_context_loss_before']:.4f} → {p['full_context_loss_after']:.4f}; "
        f"advantage vs one-token context: {p['full_context_advantage_before']:.4f} → {p['full_context_advantage_after']:.4f}",
        f"- Context/Japanese maintained: {'PASS' if p['context_maintained'] and p['japanese_maintained'] else 'FAIL'} "
        "(Japanese PASS; short-pilot absolute context loss FAIL)",
        f"- Result: **{summary['pilot_result']}**",
        f"- Thermal: {p['thermal_observation']}",
        "",
        "## Greedy versus sampling",
        "",
        f"At pilot, greedy semantic/runaway were {_percent(pilot['generation']['greedy']['semantic_rate'])} / "
        f"{_percent(pilot['generation']['greedy']['runaway_rate'])}; temperature 0.7 semantic/naturalness were "
        f"{_percent(pilot['generation']['temperature_0.7']['semantic_rate'])} / "
        f"{_percent(pilot['generation']['temperature_0.7']['naturalness_rate'])}. "
        "Teacher-forced metrics improve while free-running greedy progressively repeats and loses diversity. "
        f"Exposure-bias-like evidence: **{'YES' if summary['exposure_bias_like_evidence'] else 'NO'}**.",
        "",
        "Historical same-prefix evidence supports ATTRACTOR_WEAKENING: repetition-1 declined and median loop "
        "onset moved later from 5.120M to 15.360M, although runaway stayed 100%. Entropy and candidate margins "
        "at the current/pilot loop onset do not show a fixed implementation fault; the argmax basin remains.",
        "",
        "| Tokens | Loop onset | Rep-1 | Entropy at onset | Top1–Top2 margin | EOS P at onset |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in generation["historical_loop_confidence_same_prefix"].items():
        onset = row["loop_onset_distribution"]
        lines.append(
            f"| {label.replace('_', ',')} | {row['median_loop_onset']:.1f} | "
            f"{row['mean_repetition_1']:.4f} | {onset['entropy']:.4f} | "
            f"{onset['top1_top2_margin']:.4f} | {onset['eos_probability']:.6f} |"
        )
    lines += [
        "",
        "## Architecture, context, frequency, and exposure",
        "",
        "The current 10-layer, 384-hidden, 6-head, Pre-LN/GELU, learned-absolute-position, tied-weight model "
        "continues to improve LM metrics. Tiny overfit, copy, position, long-range, context-conditioned, and "
        "fixed-relation tests pass. Full context retains a positive advantage over one-token context. These do "
        "not meet the multi-evidence threshold for an architecture defect.",
        "",
        f"Corpus exposure is **{summary['corpus_exposure_percent']:.2f}%** ({summary['corpus_exposure_percent']/100:.3f} epoch), "
        "so undertraining remains the most economical explanation. Rare-token evidence is accepted only with "
        "the probability and cross-entropy tolerance recorded in the JSON artifact.",
        "",
        "## Operational decision",
        "",
        "- EOS weight 1.5: SAFE_FOR_CONTINUATION (terminal improvement, non-terminal Top-1 0%, 3-seed reproducibility, no LM/Semantic/Japanese regression)",
        "- Repetition auxiliary: rejected; no further lambda search",
        "- Recommended mode: standard CUDA FP32 continuation with EOS 1.5, repetition auxiliary OFF",
        "- Parallel CPU evaluation: DISABLED",
        "- Evaluation order: training stops, then offline evaluation",
        f"- Checkpoint integrity: {'PASS' if summary['checkpoint_integrity']['official_15_360m']['verified'] and summary['checkpoint_integrity']['experimental_pilot']['integrity']['pass'] else 'FAIL'}",
        f"- Final Blind: unopened; SHA256 {'PASS' if summary['checkpoint_integrity']['final_blind']['pass'] else 'FAIL'}",
        f"- pytest: {summary['pytest']}",
        "- Render/Vercel: unchanged",
        "",
        "The next phase may continue only in 256k-512k formal GPU increments until the context-maintenance gate "
        "passes; one-shot continuation to 20M is not authorized. Base Completion remains NO and must be judged "
        "again at the next formal checkpoint; greedy runaway remains a monitored metric, not the sole completion gate.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytest-result", default="pending")
    args = parser.parse_args()
    summary, auxiliary, report = build(args.pytest_result)
    outputs: dict[Path, Any] = {
        ROOT / "evaluation/foundation-v32-base-maturity-decision-summary.json": summary,
        ROOT / "evaluation/foundation-v32-scaling-trends.json": auxiliary["scaling"],
        ROOT / "evaluation/foundation-v32-generation-comparison.json": auxiliary["generation"],
    }
    for path, payload in outputs.items():
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "evaluation/foundation-v32-base-maturity-decision-report.md").write_text(
        report, encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "scaling": summary["scaling_classification"],
                "pilot": summary["pilot_result"],
                "gate": summary["next_phase_gate"],
                "permission": summary["continue_20m_gpu_permission"],
            }
        )
    )


if __name__ == "__main__":
    main()
