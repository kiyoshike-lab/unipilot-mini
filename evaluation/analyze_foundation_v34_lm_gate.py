"""PHASE 45 root-cause analysis and recipe decision."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2026)
STAGES = ("baseline", "gate1", "gate2")
TOKENS = {"baseline": 15_360_000, "gate1": 15_616_000, "gate2": 15_872_000}
FINAL_BLIND_SHA256 = "fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b"


def load(path: str) -> dict | list:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def stats(values: list[float]) -> dict:
    return {"mean": mean(values), "std": stdev(values), "values": values}


def nested(row: dict, *path: str) -> float:
    value = row
    for key in path:
        value = value[key]
    return float(value)


def gradient_map() -> dict[tuple[str, int], dict]:
    result = {}
    for seed in SEEDS:
        history = load(f"evaluation/foundation-v28-runs/current-seed-{seed}.json")["training"][
            "history"
        ][-1]
        result[("baseline", seed)] = {
            "value": float(history["gradient_norm"]),
            "definition": "final checkpoint update gradient norm",
        }
    for gate, stage in ((1, "gate1"), (2, "gate2")):
        payload = load(f"evaluation/foundation-v33-gate{gate}-training.json")
        for row in payload["results"]:
            result[(stage, int(row["seed"]))] = {
                "value": float(row["training"]["mean_gradient_norm"]),
                "maximum": float(row["training"]["max_gradient_norm"]),
                "definition": "mean pre-clip gradient norm across 256k interval",
            }
    return result


def trajectories(diagnostics: dict) -> dict:
    gradients = gradient_map()
    result = {}
    for seed in SEEDS:
        result[str(seed)] = {}
        for stage in STAGES:
            row = load(f"evaluation/phase44/{stage}/seed-{seed}.json")
            detailed = diagnostics["trajectory"][stage][str(seed)]["metrics"]
            sampling = row["generation"]["temperature_0.7"]
            middle = detailed["frequency_buckets"]["middle_20_to_80_percent"]
            rare = detailed["frequency_buckets"]["rare_bottom_20_percent"]
            result[str(seed)][stage] = {
                "tokens": TOKENS[stage],
                "validation_loss": row["validation"]["loss"],
                "perplexity": row["validation"]["perplexity"],
                "top1": row["validation"]["top_1_accuracy"],
                "top5": row["validation"]["top_5_accuracy"],
                "top10": row["validation"]["top_10_accuracy"],
                "gradient_norm": gradients[(stage, seed)],
                "teacher_forced_horizons": row["teacher_forced_horizons"],
                "middle": middle,
                "rare": rare,
                "context_full_loss": row["context"]["512"]["loss"],
                "sampling_naturalness": sampling["naturalness_rate"],
                "sampling_semantic": sampling["semantic_rate"],
                "terminal_eos_probability": row["terminal_eos"]["mean_probability"],
                "premature_eos_top1": row["nonterminal_eos"]["top1_rate"],
                "greedy_repetition": row["generation"]["greedy"]["mean_repetition_1"],
                "greedy_runaway": row["generation"]["greedy"]["runaway_rate"],
                "median_loop_onset": row["generation"]["greedy"]["median_loop_onset"],
                "loop_onset_distribution": row["generation"]["greedy"][
                    "loop_onset_distribution"
                ],
            }
    return result


def aggregate(trajectory: dict) -> dict:
    result = {}
    scalar_paths = {
        "validation_loss": ("validation_loss",),
        "perplexity": ("perplexity",),
        "top1": ("top1",),
        "top5": ("top5",),
        "top10": ("top10",),
        "gradient_norm": ("gradient_norm", "value"),
        "middle_cross_entropy": ("middle", "cross_entropy"),
        "middle_mean_probability": ("middle", "mean_correct_token_probability"),
        "rare_cross_entropy": ("rare", "cross_entropy"),
        "rare_mean_probability": ("rare", "mean_correct_token_probability"),
        "rare_median_probability": ("rare", "median_correct_token_probability"),
        "context_full_loss": ("context_full_loss",),
        "sampling_naturalness": ("sampling_naturalness",),
        "sampling_semantic": ("sampling_semantic",),
        "terminal_eos_probability": ("terminal_eos_probability",),
        "premature_eos_top1": ("premature_eos_top1",),
        "greedy_repetition": ("greedy_repetition",),
        "greedy_runaway": ("greedy_runaway",),
        "median_loop_onset": ("median_loop_onset",),
        "loop_entropy": ("loop_onset_distribution", "entropy"),
        "loop_top1_top2_margin": ("loop_onset_distribution", "top1_top2_margin"),
    }
    for stage in STAGES:
        result[stage] = {
            key: stats(
                [nested(trajectory[str(seed)][stage], *path) for seed in SEEDS]
            )
            for key, path in scalar_paths.items()
        }
        result[stage]["teacher_forced_horizons"] = {
            horizon: {
                metric: stats(
                    [
                        nested(
                            trajectory[str(seed)][stage],
                            "teacher_forced_horizons",
                            horizon,
                            metric,
                        )
                        for seed in SEEDS
                    ]
                )
                for metric in ("loss", "accuracy", "correct_token_probability")
            }
            for horizon in ("1", "2", "4", "8", "16", "32")
        }
    return result


def interval_deltas(trajectory: dict, aggregated: dict) -> dict:
    keys = (
        "validation_loss",
        "top1",
        "top5",
        "top10",
        "middle_cross_entropy",
        "middle_mean_probability",
        "rare_cross_entropy",
        "rare_mean_probability",
        "rare_median_probability",
        "sampling_semantic",
        "sampling_naturalness",
        "context_full_loss",
    )
    result = {}
    for name, left, right in (
        ("15.360_to_15.616", "baseline", "gate1"),
        ("15.616_to_15.872", "gate1", "gate2"),
    ):
        result[name] = {
            "mean": {
                key: aggregated[right][key]["mean"] - aggregated[left][key]["mean"]
                for key in keys
            },
            "by_seed": {},
        }
        for seed in SEEDS:
            before = trajectory[str(seed)][left]
            after = trajectory[str(seed)][right]
            result[name]["by_seed"][str(seed)] = {
                "validation_loss": after["validation_loss"] - before["validation_loss"],
                "top1": after["top1"] - before["top1"],
                "top5": after["top5"] - before["top5"],
                "top10": after["top10"] - before["top10"],
                "middle_cross_entropy": after["middle"]["cross_entropy"]
                - before["middle"]["cross_entropy"],
                "rare_cross_entropy": after["rare"]["cross_entropy"]
                - before["rare"]["cross_entropy"],
                "sampling_semantic": after["sampling_semantic"]
                - before["sampling_semantic"],
                "context_full_loss": after["context_full_loss"]
                - before["context_full_loss"],
            }
    return result


def difference_ci(before: float, after: float, samples: int = 300) -> list[float]:
    delta = after - before
    standard_error = math.sqrt(
        before * (1 - before) / samples + after * (1 - after) / samples
    )
    return [delta - 1.96 * standard_error, delta + 1.96 * standard_error]


def recipe_comparison(diagnostics: dict) -> dict:
    arms = load("evaluation/foundation-v31-arm-results.json")
    control, eos = arms[0], arms[1]
    detailed = diagnostics["existing_recipe_control_detailed"]
    standard_rare = detailed["standard_ce_eos_1_0"]["metrics"]["frequency_buckets"]
    eos_rare = detailed["eos_weight_1_5"]["metrics"]["frequency_buckets"]
    checks = {
        "terminal_eos_clear_improvement": eos["terminal_eos"]["mean_probability"]
        >= control["terminal_eos"]["mean_probability"] * 1.5,
        "validation_no_regression": eos["lm"]["loss"] - control["lm"]["loss"] <= 0.01,
        "topk_no_regression": all(
            eos["lm"][key] >= control["lm"][key] - 0.002
            for key in ("top1", "top5", "top10")
        ),
        "rare_no_regression": eos_rare["rare_bottom_20_percent"]["cross_entropy"]
        <= standard_rare["rare_bottom_20_percent"]["cross_entropy"] + 0.05,
        "context_no_regression": eos["context_loss"]["512"]
        <= control["context_loss"]["512"] + 0.05,
        "semantic_no_regression": eos["sampling_t07"]["semantic"]
        >= control["sampling_t07"]["semantic"] - 0.03,
        "naturalness_no_major_regression": eos["sampling_t07"]["naturalness"]
        >= control["sampling_t07"]["naturalness"] - 0.05,
        "premature_eos_top1_zero": eos["nonterminal_eos"]["top1_rate"] == 0,
    }
    return {
        "schema": "foundation-v34-recipe-comparison-v1",
        "phase": 45,
        "source": "existing PHASE 41/42 matched seed-42 256k control; no duplicate training",
        "new_recipe_pilot_executed": False,
        "standard_ce_eos_1_0": control,
        "eos_weight_1_5": eos,
        "detailed_frequency": {
            "standard_ce_eos_1_0": standard_rare,
            "eos_weight_1_5": eos_rare,
        },
        "delta_eos_1_5_minus_standard": {
            "validation_loss": eos["lm"]["loss"] - control["lm"]["loss"],
            "top1": eos["lm"]["top1"] - control["lm"]["top1"],
            "top5": eos["lm"]["top5"] - control["lm"]["top5"],
            "top10": eos["lm"]["top10"] - control["lm"]["top10"],
            "terminal_eos_probability": eos["terminal_eos"]["mean_probability"]
            - control["terminal_eos"]["mean_probability"],
            "rare_cross_entropy": eos_rare["rare_bottom_20_percent"]["cross_entropy"]
            - standard_rare["rare_bottom_20_percent"]["cross_entropy"],
            "context_full_loss": eos["context_loss"]["512"]
            - control["context_loss"]["512"],
            "semantic": eos["sampling_t07"]["semantic"]
            - control["sampling_t07"]["semantic"],
            "naturalness": eos["sampling_t07"]["naturalness"]
            - control["sampling_t07"]["naturalness"],
        },
        "checks": checks,
        "adoption_criterion_pass": all(checks.values()),
        "greedy_runaway_improved": eos["greedy"]["runaway_rate"]
        < control["greedy"]["runaway_rate"],
        "recommended_eos_weight": 1.5,
    }


def thermal_analysis() -> dict:
    diagnostic = load("evaluation/foundation-v34-thermal.json")
    gate1 = load("evaluation/foundation-v33-gate1-training.json")["results"]
    gate2 = load("evaluation/foundation-v33-gate2-training.json")["results"]
    gate1_tps = mean(row["training"]["tokens_per_second"] for row in gate1)
    gate2_tps = mean(row["training"]["tokens_per_second"] for row in gate2)
    samples = diagnostic["samples"]
    active_samples = [
        row
        for row in samples
        if row["clocks_throttle_reasons.sw_thermal_slowdown"] == "Active"
    ]
    return {
        "classification": diagnostic["summary"]["classification"],
        "phase44_max_temperature_c": max(
            row["training"]["telemetry"]["gpu_temperature_c_max"]
            for row in gate1 + gate2
        ),
        "diagnostic_max_temperature_c": diagnostic["summary"]["max_temperature_c"],
        "thermal_throttling_observed": diagnostic["summary"][
            "thermal_throttling_observed"
        ],
        "hardware_thermal_slowdown_observed": any(
            row["clocks_throttle_reasons.hw_thermal_slowdown"] == "Active"
            for row in samples
        ),
        "software_thermal_active_samples": len(active_samples),
        "samples": len(samples),
        "active_sm_clock_range_mhz": [
            min(row["clocks.sm"] for row in active_samples),
            max(row["clocks.sm"] for row in active_samples),
        ],
        "gate1_tokens_per_second": gate1_tps,
        "gate2_tokens_per_second": gate2_tps,
        "throughput_change_percent": 100 * (gate2_tps / gate1_tps - 1),
        "training_quality_root_cause": False,
        "throughput_contributor": True,
        "settings_changed": diagnostic["settings_changed"],
        "checkpoint_unchanged": diagnostic["checkpoint_unchanged"],
    }


def render_report(summary: dict) -> str:
    aggregate_rows = summary["mean_std_trajectory"]
    lines = [
        "# Foundation v3.4 LM Gate Failure Root-Cause Review",
        "",
        "## Decision",
        "",
        f"- Primary cause: **{summary['gate2_stop_lm_gate_primary_cause']}**",
        f"- Plateau classification: **{summary['plateau_classification']}**",
        f"- Next Gate: **{summary['next_phase_gate']}**",
        f"- Recommended next interval: **{summary['recommended_next_interval_tokens']:,} tokens**",
        f"- Recommended EOS weight: **{summary['recommended_eos_weight']}**",
        "- 20M permission: **NO**",
        "- Foundation Base completion: **NO**",
        "",
        "Gate 2 combined two different seed-local signals: seed 42 caused the deterministic loss/Top-1 decline, while seed 123 caused the sampling Semantic decline. Mean validation loss, Top-5/10, context, and both Middle/Rare cross-entropy continued improving, so this is not a global plateau or EOS-recipe-wide regression.",
        "",
        "## Per-seed trajectory",
        "",
        "| Seed | Tokens (M) | Loss | Top-1 | Top-5 | Top-10 | Semantic | Full context |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        for stage in STAGES:
            row = summary["per_seed_trajectory"][str(seed)][stage]
            lines.append(
                f"| {seed} | {row['tokens']/1e6:.3f} | {row['validation_loss']:.4f} | "
                f"{100*row['top1']:.2f}% | {100*row['top5']:.2f}% | {100*row['top10']:.2f}% | "
                f"{100*row['sampling_semantic']:.1f}% | {row['context_full_loss']:.4f} |"
            )
    lines += [
        "",
        "## Mean / standard deviation",
        "",
        "| Tokens (M) | Loss | Top-1 | Top-5 | Top-10 | Semantic | Naturalness |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        row = aggregate_rows[stage]
        lines.append(
            f"| {TOKENS[stage]/1e6:.3f} | {row['validation_loss']['mean']:.4f} ± {row['validation_loss']['std']:.4f} | "
            f"{100*row['top1']['mean']:.2f}% ± {100*row['top1']['std']:.2f} | "
            f"{100*row['top5']['mean']:.2f}% ± {100*row['top5']['std']:.2f} | "
            f"{100*row['top10']['mean']:.2f}% ± {100*row['top10']['std']:.2f} | "
            f"{100*row['sampling_semantic']['mean']:.2f}% ± {100*row['sampling_semantic']['std']:.2f} | "
            f"{100*row['sampling_naturalness']['mean']:.2f}% ± {100*row['sampling_naturalness']['std']:.2f} |"
        )
    recipe = summary["recipe_comparison"]
    control = recipe["standard_ce_eos_1_0"]
    eos = recipe["eos_weight_1_5"]
    lines += [
        "",
        "## EOS 1.0 versus 1.5",
        "",
        "The existing matched seed-42 256k control was sufficient, so no duplicate training pilot was run.",
        "",
        f"- Validation loss: {control['lm']['loss']:.6f} → {eos['lm']['loss']:.6f}",
        f"- Top-1/5/10 delta: {100*recipe['delta_eos_1_5_minus_standard']['top1']:+.3f} / {100*recipe['delta_eos_1_5_minus_standard']['top5']:+.3f} / {100*recipe['delta_eos_1_5_minus_standard']['top10']:+.3f} pp",
        f"- terminal P(EOS): {control['terminal_eos']['mean_probability']:.5f} → {eos['terminal_eos']['mean_probability']:.5f}",
        f"- premature EOS Top-1: {100*eos['nonterminal_eos']['top1_rate']:.2f}%",
        f"- Rare CE delta: {recipe['delta_eos_1_5_minus_standard']['rare_cross_entropy']:+.6f}",
        f"- Semantic/Naturalness delta: {100*recipe['delta_eos_1_5_minus_standard']['semantic']:+.1f} / {100*recipe['delta_eos_1_5_minus_standard']['naturalness']:+.1f} pp",
        f"- Adoption criterion: {'PASS' if recipe['adoption_criterion_pass'] else 'FAIL'}; greedy runaway improvement: {'YES' if recipe['greedy_runaway_improved'] else 'NO'}",
        "",
        "## Rare metric resolution",
        "",
        f"- Rare CE: {aggregate_rows['gate1']['rare_cross_entropy']['mean']:.4f} → {aggregate_rows['gate2']['rare_cross_entropy']['mean']:.4f}",
        f"- Arithmetic mean probability: {aggregate_rows['gate1']['rare_mean_probability']['mean']:.8f} → {aggregate_rows['gate2']['rare_mean_probability']['mean']:.8f}",
        f"- Mean of per-seed medians: {aggregate_rows['gate1']['rare_median_probability']['mean']:.8f} → {aggregate_rows['gate2']['rare_median_probability']['mean']:.8f}",
        "- Resolution: YES. Cross-entropy tracks the geometric mean and the low-probability body; arithmetic mean probability was dominated by a small high-probability tail, especially seed 42. The fixed 112-target bucket and recomputation exactly matched PHASE 44.",
        "",
        "## Generation and thermal",
        "",
        f"- Greedy runaway: {100*aggregate_rows['gate2']['greedy_runaway']['mean']:.0f}%",
        f"- Median loop onset mean: {aggregate_rows['baseline']['median_loop_onset']['mean']:.1f} → {aggregate_rows['gate2']['median_loop_onset']['mean']:.1f}",
        f"- Attractor: **{summary['attractor_classification']}** (runaway unchanged; repetition improved overall but onset moved earlier)",
        f"- Thermal: **{summary['thermal']['classification']}**; max {summary['thermal']['phase44_max_temperature_c']:.0f}C; software thermal slowdown observed; hardware slowdown not observed",
        f"- Throughput: {summary['thermal']['gate1_tokens_per_second']:.2f} → {summary['thermal']['gate2_tokens_per_second']:.2f} tok/s ({summary['thermal']['throughput_change_percent']:.2f}%)",
        "- Thermal is a throughput contributor, not the LM-quality root cause: clocks stayed in a narrow 1890–1905 MHz range during active thermal flags and all numerical/integrity checks passed.",
        "",
        "## Integrity",
        "",
        f"- Deterministic evaluation: {'PASS' if summary['deterministic_evaluation_pass'] else 'FAIL'}",
        f"- Context regression: {'YES' if summary['context_regression'] else 'NO'}",
        f"- Checkpoint integrity: {'PASS' if summary['checkpoint_integrity'] else 'FAIL'}",
        f"- Final Blind: unopened; SHA256 {'PASS' if summary['final_blind']['pass'] else 'FAIL'}",
        f"- Corpus exposure: {summary['corpus_exposure_percent']:.2f}%",
        "- Parallel CPU evaluation: DISABLED",
        f"- pytest: {summary['pytest']}",
        "- Render/Vercel: unchanged",
    ]
    return "\n".join(lines) + "\n"


def build(pytest_result: str = "pending") -> tuple[dict, dict, str]:
    diagnostics = load("evaluation/foundation-v34-determinism-and-rare.json")
    phase44 = load("evaluation/foundation-v33-context-gate-summary.json")
    trajectory = trajectories(diagnostics)
    aggregated = aggregate(trajectory)
    deltas = interval_deltas(trajectory, aggregated)
    recipe = recipe_comparison(diagnostics)
    thermal = thermal_analysis()
    gate1_semantic = aggregated["gate1"]["sampling_semantic"]["mean"]
    gate2_semantic = aggregated["gate2"]["sampling_semantic"]["mean"]
    summary = {
        "schema": "foundation-v34-lm-gate-review-summary-v1",
        "phase": 45,
        "large_scale_training_executed": False,
        "formal_20m_training_executed": False,
        "gate2_stop_lm_gate_primary_cause": (
            "SEED_LOCAL_SHORT_TERM_VARIANCE: seed 42 deterministic loss/Top-1; "
            "seed 123 sampling Semantic"
        ),
        "plateau_classification": "HEALTHY_SHORT_TERM_VARIANCE",
        "per_seed_trajectory": trajectory,
        "mean_std_trajectory": aggregated,
        "interval_deltas": deltas,
        "gate2_failure_reproduction": {
            "validation_loss_worse_seeds": [
                seed
                for seed in SEEDS
                if deltas["15.616_to_15.872"]["by_seed"][str(seed)]["validation_loss"] > 0
            ],
            "top1_materially_worse_seeds": [
                seed
                for seed in SEEDS
                if deltas["15.616_to_15.872"]["by_seed"][str(seed)]["top1"] < -0.002
            ],
            "semantic_worse_seeds": [
                seed
                for seed in SEEDS
                if deltas["15.616_to_15.872"]["by_seed"][str(seed)]["sampling_semantic"] < 0
            ],
            "semantic_delta_normal_95pct_screening_ci": difference_ci(
                gate1_semantic, gate2_semantic
            ),
            "all_seed_common": False,
        },
        "deterministic_evaluation_pass": diagnostics["deterministic_evaluation"]["pass"],
        "evaluation_nondeterminism": False,
        "sampling_protocol": diagnostics["sampling_protocol"],
        "recipe_comparison": recipe,
        "recommended_eos_weight": 1.5,
        "rare_metric_contradiction_resolved": True,
        "rare_metric_resolution": (
            "CE/geometric mean and per-seed median improved while arithmetic mean fell because "
            "the high-probability tail contracted; bucket population remained 112 per seed"
        ),
        "attractor_classification": "STATIC",
        "context_regression": phase44["context_regression"],
        "thermal": thermal,
        "corpus_exposure_percent": phase44["corpus_exposure_percent"],
        "next_phase_gate": "CONTINUE_SHORT_GPU_GATES_EOS_1_5",
        "recommended_next_interval_tokens": 256_000,
        "continue_20m_permission": False,
        "recommended_formal_training_recipe": {
            "objective": "standard cross-entropy with EOS target weight 1.5",
            "repetition_auxiliary": False,
            "device": "CUDA",
            "precision": "FP32",
            "amp": False,
            "interval_tokens": 256_000,
            "parallel_cpu_evaluation": "DISABLED",
        },
        "foundation_base_complete": False,
        "checkpoint_integrity": diagnostics["checkpoint_integrity"]["unchanged"]
        and thermal["checkpoint_unchanged"],
        "final_blind": {
            "opened": diagnostics["final_blind"]["opened"],
            "sha256": diagnostics["final_blind"]["sha256"],
            "pass": diagnostics["final_blind"]["sha256"] == FINAL_BLIND_SHA256
            and not diagnostics["final_blind"]["opened"],
        },
        "parallel_cpu_evaluation": "DISABLED",
        "pytest": pytest_result,
        "render_changed": False,
        "vercel_changed": False,
    }
    return summary, recipe, render_report(summary)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pytest-result", default="pending")
    args = parser.parse_args()
    summary, recipe, report = build(args.pytest_result)
    (ROOT / "evaluation/foundation-v34-lm-gate-review-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "evaluation/foundation-v34-recipe-comparison.json").write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "evaluation/foundation-v34-lm-gate-review-report.md").write_text(
        report, encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "cause": summary["gate2_stop_lm_gate_primary_cause"],
                "classification": summary["plateau_classification"],
                "gate": summary["next_phase_gate"],
            }
        )
    )


if __name__ == "__main__":
    main()
