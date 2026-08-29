from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(values: list[float]) -> dict:
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "std_population": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def path_value(value: dict, path: tuple):
    for key in path:
        value = value[key]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="checkpoints/foundation-v16-reproduction")
    parser.add_argument("--output", default="evaluation/foundation-v16-reproduction-summary.json")
    args = parser.parse_args()
    directory = ROOT / args.input_dir
    seeds = [42, 123, 2026]
    variants = ("current_unscaled", "sqrt_scaled_a")
    runs = {
        variant: [load(directory / f"{variant}-seed-{seed}.json") for seed in seeds]
        for variant in variants
    }
    final_paths = {
        "validation_loss": ("validation", "loss"),
        "top_1_accuracy": ("validation", "top_1_accuracy"),
        "top_5_accuracy": ("validation", "top_5_accuracy"),
        "top_10_accuracy": ("validation", "top_10_accuracy"),
        "context_sensitivity_score": ("context_sensitivity", "context_sensitivity_score"),
        "layer_9_output_rms": ("probe", "layers", 9, "output", "rms"),
        "logit_mean": ("probe", "logits", "mean"),
        "logit_std": ("probe", "logits", "std"),
        "max_logit": ("probe", "logits", "max"),
        "softmax_entropy": ("probe", "logits", "mean_softmax_entropy"),
        "normalized_softmax_entropy": ("probe", "logits", "normalized_softmax_entropy"),
        "top_1_probability": ("probe", "logits", "mean_top_1_probability"),
        "top_5_probability_mass": ("probe", "logits", "mean_top_5_probability_mass"),
        "gradient_norm": ("gradient_norm",),
        "period_top_1_frequency": ("frequency", "tokens", "。", "top_1_predicted_frequency"),
        "comma_top_1_frequency": ("frequency", "tokens", "、", "top_1_predicted_frequency"),
        "period_comma_top_1_mass": ("frequency", "period_comma_top1_mass"),
    }
    comparison = {}
    for variant, variant_runs in runs.items():
        comparison[variant] = {}
        for name, path in final_paths.items():
            comparison[variant][name] = aggregate([
                float(path_value(run["final"], path)) for run in variant_runs
            ])
        for accuracy_kind in ("top_1_accuracy", "top_5_accuracy"):
            values = []
            for run in variant_runs:
                buckets = run["final"]["frequency"]["buckets"]
                values.append(statistics.fmean(
                    row[accuracy_kind]
                    for name, row in buckets.items()
                    if name != "top_1_percent"
                ))
            comparison[variant][f"non_top_1_percent_macro_{accuracy_kind}"] = aggregate(values)

    milestone_summary = {}
    for variant, variant_runs in runs.items():
        milestone_summary[variant] = {}
        for history_index, reference in enumerate(variant_runs[0]["history"]):
            update = str(reference["update"])
            rows = [run["history"][history_index] for run in variant_runs]
            milestone_summary[variant][update] = {
                "embedding": {
                    name: aggregate([
                        float(row["probe"]["embedding"][name]["rms"])
                        for row in rows
                    ])
                    for name in ("raw_token", "scaled_token", "position", "combined")
                },
                "scaled_to_position_rms_ratio": aggregate([
                    float(row["probe"]["embedding"]["scaled_to_position_rms_ratio"])
                    for row in rows
                ]),
                "layer_9_output_rms": aggregate([
                    float(row["probe"]["layers"][9]["output"]["rms"])
                    for row in rows
                ]),
                "logits": {
                    name: aggregate([
                        float(row["probe"]["logits"][name]) for row in rows
                    ])
                    for name in (
                        "mean", "std", "max", "mean_softmax_entropy",
                        "mean_top_1_probability", "mean_top_5_probability_mass",
                    )
                },
                "layers": [
                    {
                        "layer": layer,
                        **{
                            component: {
                                statistic: aggregate([
                                    float(row["probe"]["layers"][layer][component][statistic])
                                    for row in rows
                                ])
                                for statistic in ("mean", "std", "rms")
                            }
                            for component in (
                                "input", "attention", "residual", "mlp", "output"
                            )
                        },
                    }
                    for layer in range(10)
                ],
            }
    paired = {}
    for name in final_paths:
        current_values = comparison["current_unscaled"][name]["values"]
        scaled_values = comparison["sqrt_scaled_a"][name]["values"]
        paired[name] = aggregate([
            scaled - current for current, scaled in zip(current_values, scaled_values)
        ])
    checks = {
        "three_seed_loss_improved": all(
            scaled < current for current, scaled in zip(
                comparison["current_unscaled"]["validation_loss"]["values"],
                comparison["sqrt_scaled_a"]["validation_loss"]["values"],
            )
        ),
        "three_seed_top_1_improved": all(
            scaled > current for current, scaled in zip(
                comparison["current_unscaled"]["top_1_accuracy"]["values"],
                comparison["sqrt_scaled_a"]["top_1_accuracy"]["values"],
            )
        ),
        "mean_top_5_improved": (
            comparison["sqrt_scaled_a"]["top_5_accuracy"]["mean"]
            > comparison["current_unscaled"]["top_5_accuracy"]["mean"]
        ),
        "mean_context_sensitivity_not_regressed": (
            comparison["sqrt_scaled_a"]["context_sensitivity_score"]["mean"]
            >= comparison["current_unscaled"]["context_sensitivity_score"]["mean"]
        ),
        "mean_layer_9_rms_reduced": (
            comparison["sqrt_scaled_a"]["layer_9_output_rms"]["mean"]
            < comparison["current_unscaled"]["layer_9_output_rms"]["mean"]
        ),
        "all_checkpoints_strict_reload": all(
            run["checkpoint"]["strict_reload"]
            for variant_runs in runs.values() for run in variant_runs
        ),
    }
    result = {
        "schema_version": "foundation-v16-reproduction-summary-v1",
        "seeds": seeds,
        "standard_deviation": "population standard deviation across the three requested seeds",
        "comparison": comparison,
        "paired_sqrt_minus_current": paired,
        "milestones": milestone_summary,
        "checks": checks,
        "same_data_optimizer_lr_batch_evaluation_within_seed_pair": True,
        "external_ai_api": "OFF",
        "production_changed": False,
        "final_blind_used": False,
    }
    (ROOT / args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"comparison": comparison, "checks": checks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
