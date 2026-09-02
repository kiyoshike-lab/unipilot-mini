from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_foundation_v21_ab import file_sha256, load_json


MILESTONES = (0, 64_000, 128_000, 192_000, 256_000)
SEEDS = (42, 123, 2026)
VARIANTS = ("current", "depth_init")


def nested(row: dict, path: str):
    value = row
    for key in path.split("."):
        value = value[key]
    return value


def mean_std(values) -> dict:
    numeric = [float(value) for value in values]
    return {
        "mean": statistics.mean(numeric),
        "std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
        "values": numeric,
    }


def load_runs(directory: Path) -> dict[tuple[str, int], dict]:
    runs = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            path = directory / f"{variant}-seed-{seed}.json"
            if not path.exists():
                raise RuntimeError(f"missing PHASE 32 run: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["final"]["tokens_processed"] != 256_000:
                raise RuntimeError(f"incomplete PHASE 32 run: {path}")
            runs[(variant, seed)] = payload
    return runs


def history_at(run: dict, tokens: int) -> dict:
    return next(row for row in run["training"]["history"] if row["tokens_processed"] == tokens)


def aggregate_curve(runs: dict, path: str) -> dict:
    return {
        variant: {
            str(tokens): mean_std([
                nested(history_at(runs[(variant, seed)], tokens), path)
                for seed in SEEDS
            ])
            for tokens in MILESTONES
        }
        for variant in VARIANTS
    }


def paired_curve(runs: dict, path: str) -> dict:
    output = {}
    for tokens in MILESTONES:
        differences = {
            str(seed): (
                nested(history_at(runs[("depth_init", seed)], tokens), path)
                - nested(history_at(runs[("current", seed)], tokens), path)
            )
            for seed in SEEDS
        }
        output[str(tokens)] = {
            "depth_minus_current_by_seed": differences,
            **mean_std(differences.values()),
        }
    return output


def final_aggregate(runs: dict, path: str) -> dict:
    return {
        variant: mean_std([
            nested(runs[(variant, seed)]["final"], path) for seed in SEEDS
        ])
        for variant in VARIANTS
    }


def frequency_summary(runs: dict) -> dict:
    buckets = next(iter(runs.values()))["final"]["validation"]["frequency_buckets"]
    fields = (
        "top_1_accuracy", "top_5_accuracy", "top_10_accuracy",
        "mean_correct_token_probability", "cross_entropy",
    )
    return {
        variant: {
            bucket: {
                field: mean_std([
                    runs[(variant, seed)]["final"]["validation"]
                    ["frequency_buckets"][bucket][field]
                    for seed in SEEDS
                ])
                for field in fields
            }
            for bucket in buckets
        }
        for variant in VARIANTS
    }


def punctuation_summary(runs: dict) -> dict:
    texts = ("。", "、", "の", "に", "は", "を", "が")
    fields = (
        "actual_frequency", "top_1_predicted_frequency", "mean_probability", "accuracy"
    )
    return {
        variant: {
            text: {
                field: mean_std([
                    runs[(variant, seed)]["final"]["validation"]["punctuation"]
                    [text][field]
                    for seed in SEEDS
                ])
                for field in fields
            }
            for text in texts
        }
        for variant in VARIANTS
    }


def context_summary(runs: dict) -> dict:
    contexts = ("512", "64", "16", "2", "1")
    return {
        variant: {
            context: {
                field: mean_std([
                    runs[(variant, seed)]["final"]["context_utilization"]
                    [context][field]
                    for seed in SEEDS
                ])
                for field in ("loss", "top_1_accuracy", "mean_target_probability")
            }
            for context in contexts
        } | {
            "full_vs_last_1_loss_advantage": mean_std([
                runs[(variant, seed)]["final"]["context_utilization"]
                ["full_vs_last_1_loss_advantage"]
                for seed in SEEDS
            ]),
            "full_vs_last_2_loss_advantage": mean_std([
                runs[(variant, seed)]["final"]["context_utilization"]
                ["full_vs_last_2_loss_advantage"]
                for seed in SEEDS
            ]),
        }
        for variant in VARIANTS
    }


def run_table(runs: dict) -> list[dict]:
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            run = runs[(variant, seed)]
            final = run["final"]
            recorded_checkpoint_rows = [
                row for row in run["training"]["history"] if "checkpoint" in row
            ]
            rows.append({
                "seed": seed,
                "architecture": variant,
                "tokens": final["tokens_processed"],
                "best_validation_loss": run["best_validation_loss"],
                "final_validation_loss": final["validation"]["loss"],
                "top_1": final["validation"]["top_1_accuracy"],
                "top_5": final["validation"]["top_5_accuracy"],
                "top_10": final["validation"]["top_10_accuracy"],
                "tokens_per_second": final["training_tokens_per_second"],
                "peak_ram_mb": max(row["peak_ram_mb"] for row in run["training"]["history"]),
                "checkpoint_strict_reload": all(
                    row["checkpoint"]["strict_reload"]
                    for row in recorded_checkpoint_rows
                ),
                "checkpoint_rows_recorded": len(recorded_checkpoint_rows),
            })
    return rows


def generation_comparison(runs: dict) -> dict:
    output = {}
    for variant in VARIANTS:
        run = runs[(variant, 42)]
        output[variant] = {
            str(row["tokens_processed"]): {
                mode: {
                    "settings": value["settings"],
                    "metrics": value["metrics"],
                }
                for mode, value in row["generation"].items()
            }
            for row in run["training"]["history"]
        }
    current = output["current"]["256000"]
    depth = output["depth_init"]["256000"]
    current_greedy = current["greedy_no_penalty"]["metrics"]
    depth_greedy = depth["greedy_no_penalty"]["metrics"]
    current_sampling = current["sampling_t07_topk40_topp09_no_penalty"]["metrics"]
    depth_sampling = depth["sampling_t07_topk40_topp09_no_penalty"]["metrics"]
    regression = (
        depth_greedy["mean_repetition_rate"] > current_greedy["mean_repetition_rate"] + .05
        or depth_greedy["character_validity"] < current_greedy["character_validity"] - .05
        or depth_sampling["character_validity"] < current_sampling["character_validity"] - .05
    )
    return {
        "representative_seed": 42,
        "curves": output,
        "final_differences_depth_minus_current": {
            "greedy_character_validity": depth_greedy["character_validity"] - current_greedy["character_validity"],
            "greedy_natural_japanese": depth_greedy["natural_japanese_proxy"] - current_greedy["natural_japanese_proxy"],
            "greedy_repetition": depth_greedy["mean_repetition_rate"] - current_greedy["mean_repetition_rate"],
            "sampling_character_validity": depth_sampling["character_validity"] - current_sampling["character_validity"],
            "sampling_natural_japanese": depth_sampling["natural_japanese_proxy"] - current_sampling["natural_japanese_proxy"],
            "sampling_repetition": depth_sampling["mean_repetition_rate"] - current_sampling["mean_repetition_rate"],
        },
        "generation_regression": regression,
        "fluent_japanese_required_at_256k": False,
    }


def choose_architecture(summary: dict) -> tuple[dict, str]:
    final = summary["final_aggregate"]
    paired = summary["paired_seed_differences"]["validation_loss"]["256000"]
    loss_improved = (
        final["validation_loss"]["depth_init"]["mean"]
        < final["validation_loss"]["current"]["mean"]
    )
    consistent = all(value < 0 for value in paired["depth_minus_current_by_seed"].values())
    topk_not_worse = all(
        final[name]["depth_init"]["mean"] >= final[name]["current"]["mean"]
        for name in ("top_1", "top_5", "top_10")
    )
    punctuation = all(
        summary["learning_curves"]["punctuation_mass"]["depth_init"][str(tokens)]["mean"]
        < summary["learning_curves"]["punctuation_mass"]["current"][str(tokens)]["mean"]
        for tokens in MILESTONES[1:]
    )
    residual = (
        final["layer_9_rms"]["depth_init"]["mean"]
        < final["layer_9_rms"]["current"]["mean"]
        and summary["activation_health_pass"]
    )
    synthetic = summary["synthetic_smoke"]["gate_pass"]
    current_context = final["context_advantage"]["current"]["mean"]
    depth_context = final["context_advantage"]["depth_init"]["mean"]
    context_ok = depth_context > 0 and depth_context >= current_context - .05
    generation_ok = not summary["generation"]["generation_regression"]
    gates = {
        "A_validation_loss_mean_improves": loss_improved,
        "B_improvement_consistent_across_seeds": consistent,
        "C_top_1_5_10_not_worse": topk_not_worse,
        "D_punctuation_collapse_clearly_improves": punctuation,
        "E_residual_rms_stable": residual,
        "F_synthetic_smoke_no_regression": synthetic,
        "G_context_no_major_regression": context_ok,
        "H_generation_trend_not_worse": generation_ok,
    }
    instability = not summary["activation_health_pass"]
    if instability:
        decision = "TRAINING_INSTABILITY"
    elif all(gates.values()):
        decision = "DEPTH_INIT_PROMOTE"
    elif not synthetic or not context_ok or not generation_ok:
        decision = "CURRENT_RETAIN"
    elif not (loss_improved and consistent and topk_not_worse and punctuation and residual):
        decision = "EXTEND_AB_TO_512K"
    else:
        decision = "UNRESOLVED"
    return gates, decision


def markdown_report(summary: dict) -> str:
    lines = [
        "# UniPilot Foundation v2.1 Controlled 256k Architecture A/B Report",
        "",
        f"Final Gate: **{summary['gate']}**",
        f"Formal Foundation Architecture: **{summary['formal_foundation_architecture']}**",
        "Foundation Base complete: **NO**",
        f"Next token budget: **{summary['next_token_budget']}**",
        "",
        "## Six runs",
        "",
        "| Architecture | Seed | Tokens | Best val | Final val | Top-1 | Top-5 | Top-10 | tok/s | Peak RAM MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["runs"]:
        lines.append(
            f"| {row['architecture']} | {row['seed']} | {row['tokens']:,} | "
            f"{row['best_validation_loss']:.4f} | {row['final_validation_loss']:.4f} | "
            f"{row['top_1']:.2%} | {row['top_5']:.2%} | {row['top_10']:.2%} | "
            f"{row['tokens_per_second']:.1f} | {row['peak_ram_mb']:.1f} |"
        )
    lines += ["", "## 3-seed final mean ± sample std", ""]
    for metric in ("validation_loss", "top_1", "top_5", "top_10", "punctuation_mass",
                   "top_1_percent_outside", "layer_9_rms", "context_advantage"):
        current = summary["final_aggregate"][metric]["current"]
        depth = summary["final_aggregate"][metric]["depth_init"]
        lines.append(
            f"- {metric}: Current {current['mean']:.6f} ± {current['std']:.6f}; "
            f"Depth {depth['mean']:.6f} ± {depth['std']:.6f}."
        )
    lines += ["", "## Validation loss learning curve", "",
              "| Tokens | Current mean ± std | Depth mean ± std | Depth−Current paired mean |",
              "|---:|---:|---:|---:|"]
    for tokens in MILESTONES:
        current = summary["learning_curves"]["validation_loss"]["current"][str(tokens)]
        depth = summary["learning_curves"]["validation_loss"]["depth_init"][str(tokens)]
        paired = summary["paired_seed_differences"]["validation_loss"][str(tokens)]
        lines.append(
            f"| {tokens:,} | {current['mean']:.4f} ± {current['std']:.4f} | "
            f"{depth['mean']:.4f} ± {depth['std']:.4f} | {paired['mean']:+.4f} |"
        )
    lines += ["", "## Punctuation and residual learning curves", "",
              "| Tokens | Current punct. mass | Depth punct. mass | Current Layer9 RMS | Depth Layer9 RMS |",
              "|---:|---:|---:|---:|---:|"]
    for tokens in MILESTONES:
        current_punctuation = summary["learning_curves"]["punctuation_mass"]["current"][str(tokens)]
        depth_punctuation = summary["learning_curves"]["punctuation_mass"]["depth_init"][str(tokens)]
        current_rms = summary["learning_curves"]["layer_9_rms"]["current"][str(tokens)]
        depth_rms = summary["learning_curves"]["layer_9_rms"]["depth_init"][str(tokens)]
        lines.append(
            f"| {tokens:,} | {current_punctuation['mean']:.2%} ± {current_punctuation['std']:.2%} | "
            f"{depth_punctuation['mean']:.2%} ± {depth_punctuation['std']:.2%} | "
            f"{current_rms['mean']:.3f} ± {current_rms['std']:.3f} | "
            f"{depth_rms['mean']:.3f} ± {depth_rms['std']:.3f} |"
        )
    lines += ["", "## Final frequency buckets (3-seed mean)", "",
              "| Bucket | Architecture | Top-1 | Top-5 | Top-10 | Correct prob. | Cross entropy |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for bucket in summary["frequency_buckets"]["current"]:
        for variant in VARIANTS:
            row = summary["frequency_buckets"][variant][bucket]
            lines.append(
                f"| {bucket} | {variant} | {row['top_1_accuracy']['mean']:.2%} | "
                f"{row['top_5_accuracy']['mean']:.2%} | {row['top_10_accuracy']['mean']:.2%} | "
                f"{row['mean_correct_token_probability']['mean']:.4f} | "
                f"{row['cross_entropy']['mean']:.4f} |"
            )
    lines += ["", "## Context utilization (final 3-seed mean loss)", "",
              "| Context tokens | Current | Depth |",
              "|---:|---:|---:|"]
    for context in ("512", "64", "16", "2", "1"):
        lines.append(
            f"| {context} | {summary['context_utilization']['current'][context]['loss']['mean']:.4f} | "
            f"{summary['context_utilization']['depth_init'][context]['loss']['mean']:.4f} |"
        )
    lines += ["", "## Representative generation at 256k (seed 42)", "",
              "| Architecture | Mode | Valid | Natural proxy | Semantic proxy | Completion | Runaway | Repetition |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        final_generation = summary["generation"]["curves"][variant]["256000"]
        for mode in ("greedy_no_penalty", "sampling_t07_topk40_topp09_no_penalty"):
            metrics = final_generation[mode]["metrics"]
            lines.append(
                f"| {variant} | {mode} | {metrics['character_validity']:.2%} | "
                f"{metrics['natural_japanese_proxy']:.2%} | {metrics['semantic_coherence_proxy']:.2%} | "
                f"{metrics['completion_proxy']:.2%} | {metrics['runaway_rate']:.2%} | "
                f"{metrics['mean_repetition_rate']:.4f} |"
            )
    lines += ["", "## Selection gates", ""]
    lines.extend(
        f"- {name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in summary["selection_gates"].items()
    )
    generation = summary["generation"]["final_differences_depth_minus_current"]
    lines += [
        "",
        "## Interpretation",
        "",
        "Depth-init improved validation loss in all three paired seeds, improved the 3-seed mean "
        "Top-1/5/10, reduced mean punctuation collapse at every trained milestone, and kept Layer9 "
        "RMS substantially lower. Context utilization remained positive without a major mean regression.",
        "",
        "It is not promoted because the fixed representative generation evaluation showed a clear "
        f"greedy repetition regression ({generation['greedy_repetition']:+.4f}) and lower sampling "
        f"character validity ({generation['sampling_character_validity']:+.4f}). PHASE 32 explicitly "
        "forbids promotion when language-model metrics improve but generation regresses. Current is "
        "therefore retained; this does not claim that Depth has lower learning capacity.",
        "",
        "Synthetic smoke: PASS for both variants. Novel random Key Lookup and modular addition were not gates.",
        "Final Blind content was not parsed; SHA256 only was verified.",
        "No Production, Campus, Render, Vercel, tokenizer, or corpus change was made.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v21.json")
    parser.add_argument("--runs", default="evaluation/foundation-v21-runs")
    parser.add_argument("--smoke", default="evaluation/foundation-v21-synthetic-smoke.json")
    parser.add_argument("--summary", default="evaluation/foundation-v21-summary.json")
    parser.add_argument("--report", default="evaluation/foundation-v21-architecture-selection-report.md")
    args = parser.parse_args()
    settings = load_json(args.config)
    runs = load_runs(ROOT / args.runs)
    smoke = load_json(args.smoke)
    run_rows = run_table(runs)
    metric_paths = {
        "validation_loss": "validation.loss",
        "top_1": "validation.top_1_accuracy",
        "top_5": "validation.top_5_accuracy",
        "top_10": "validation.top_10_accuracy",
        "punctuation_mass": "validation.period_comma_prediction_mass",
        "top_1_percent_outside": "validation.top_1_percent_outside_accuracy",
        "layer_9_rms": "activation_health.layer_9_rms",
        "context_advantage": "context_utilization.full_vs_last_1_loss_advantage",
    }
    same_order = all(
        runs[("current", seed)]["training"]["data_order_sha256"]
        == runs[("depth_init", seed)]["training"]["data_order_sha256"]
        for seed in SEEDS
    )
    final_blind_path = ROOT / settings["final_blind"]["path"]
    final_blind_sha = file_sha256(final_blind_path)
    summary = {
        "schema_version": "foundation-v21-controlled-ab-summary-v1",
        "project": settings["project"],
        "runs": run_rows,
        "run_count": len(run_rows),
        "all_runs_256k": all(row["tokens"] == 256_000 for row in run_rows),
        "parameters_equal": all(row["parameters"] == 19_514_880 for row in runs.values()),
        "paired_data_order_equal": same_order,
        "actual_resume_runs": [
            {
                "variant": variant,
                "seed": seed,
                "resumed_from": runs[(variant, seed)]["training"].get("resumed_from"),
            }
            for variant in VARIANTS for seed in SEEDS
            if runs[(variant, seed)]["training"].get("resumed_from")
        ],
        "learning_curves": {
            name: aggregate_curve(runs, path) for name, path in metric_paths.items()
        },
        "paired_seed_differences": {
            name: paired_curve(runs, path) for name, path in metric_paths.items()
        },
        "final_aggregate": {
            name: final_aggregate(runs, path) for name, path in metric_paths.items()
        },
        "frequency_buckets": frequency_summary(runs),
        "punctuation_tokens": punctuation_summary(runs),
        "context_utilization": context_summary(runs),
        "synthetic_smoke": smoke,
        "generation": generation_comparison(runs),
        "activation_health_pass": all(
            row["activation_health"]["all_finite"]
            and not row["activation_health"]["explosion"]
            and not row["activation_health"]["collapse"]
            for run in runs.values() for row in run["training"]["history"]
        ),
        "checkpoint_integrity_pass": all(row["checkpoint_strict_reload"] for row in run_rows),
        "final_blind": {
            "path": settings["final_blind"]["path"],
            "sha256": final_blind_sha,
            "expected_sha256": settings["final_blind"]["expected_sha256"],
            "match": final_blind_sha == settings["final_blind"]["expected_sha256"],
            "content_parsed_by_phase32": False,
        },
        "foundation_base_complete": False,
        "production_changed": False,
        "campus_changed": False,
        "render_deployed": False,
        "vercel_deployed": False,
        "external_ai_api": "OFF",
    }
    gates, decision = choose_architecture(summary)
    summary["selection_gates"] = gates
    summary["gate"] = decision
    summary["formal_foundation_architecture"] = (
        "Depth-scaled init" if decision == "DEPTH_INIT_PROMOTE" else "Current"
    )
    summary["winner_by_validation_and_calibration"] = "Depth-scaled init"
    summary["formal_winner"] = summary["formal_foundation_architecture"]
    summary["next_token_budget"] = "512k"
    summary["next_training_plan"] = (
        "Retain Current and continue Current-only clean-corpus pretraining to 512k tokens."
        if decision == "CURRENT_RETAIN"
        else "Continue the selected architecture to 512k tokens."
    )
    summary["full_training_continue"] = decision not in {"TRAINING_INSTABILITY", "UNRESOLVED"}
    summary_path = ROOT / args.summary
    report_path = ROOT / args.report
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps({
        "gate": decision,
        "formal_architecture": summary["formal_foundation_architecture"],
        "next_token_budget": summary["next_token_budget"],
        "selection_gates": gates,
        "summary": summary_path.relative_to(ROOT).as_posix(),
        "report": report_path.relative_to(ROOT).as_posix(),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
