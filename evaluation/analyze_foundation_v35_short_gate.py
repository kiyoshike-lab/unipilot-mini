"""PHASE 46 preregistered gate decisions and rolling trend analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2026)
CONTEXTS = (512, 256, 128, 64, 32, 16, 8, 2, 1)
FREQUENCY_BUCKETS = (
    "top_1_percent",
    "top_5_percent_excluding_top_1",
    "top_20_percent_excluding_top_5",
    "middle_20_to_80_percent",
    "rare_bottom_20_percent",
)
BLIND_SHA = "fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b"
THRESHOLDS = {
    "schema": "foundation-v35-preregistered-thresholds-v1",
    "phase": 46,
    "defined_before_training": True,
    "validation_mean_loss_max_increase": 0.03,
    "validation_single_seed_loss_max_increase": 0.05,
    "topk_mean_max_drop": 0.005,
    "context_mean_full_loss_max_increase": 0.05,
    "context_minimum_full_advantage": 0.10,
    "frequency_ce_max_increase": 0.05,
    "rare_median_probability_min_ratio": 0.80,
    "sampling_semantic_max_drop": 0.08,
    "sampling_naturalness_max_drop": 0.08,
    "eos_terminal_min_ratio": 0.90,
    "premature_eos_top1_required": 0.0,
    "teacher_forced_loss_max_increase": 0.10,
    "hardware_thermal_throttling_is_stability_failure": True,
    "source": "PHASE 43-45 multi-seed standard deviations and PHASE 45 HEALTHY_SHORT_TERM_VARIANCE classification",
}


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(stage: str) -> list[dict]:
    return [read(f"evaluation/phase46/{stage}/seed-{seed}.json") for seed in SEEDS]


def metric(rows_: list[dict], *keys: str) -> list[float]:
    values = []
    for row in rows_:
        value = row
        for key in keys:
            value = value[key]
        values.append(float(value))
    return values


def aggregate(rows_: list[dict]) -> dict:
    result = {}
    fields = {
        "validation_loss": ("validation", "loss"), "perplexity": ("validation", "perplexity"),
        "top1": ("validation", "top_1_accuracy"), "top5": ("validation", "top_5_accuracy"), "top10": ("validation", "top_10_accuracy"),
        "semantic": ("generation", "temperature_0.7", "semantic_rate"), "naturalness": ("generation", "temperature_0.7", "naturalness_rate"),
        "terminal_eos_probability": ("terminal_eos", "mean_probability"), "premature_eos_top1": ("nonterminal_eos", "top1_rate"),
        "greedy_runaway": ("generation", "greedy", "runaway_rate"), "greedy_repetition_1": ("generation", "greedy", "repetition", "1"),
        "median_loop_onset": ("generation", "greedy", "median_loop_onset"),
        "loop_entropy": ("generation", "greedy", "loop_onset_distribution", "entropy"),
        "loop_margin": ("generation", "greedy", "loop_onset_distribution", "top1_top2_margin"),
    }
    for name, path in fields.items():
        values = metric(rows_, *path)
        result[name] = {"mean": mean(values), "std": stdev(values), "by_seed": dict(zip(map(str, SEEDS), values))}
    result["context"] = {}
    for context in CONTEXTS:
        values = metric(rows_, "context", str(context), "loss")
        result["context"][str(context)] = {"mean": mean(values), "std": stdev(values), "by_seed": dict(zip(map(str, SEEDS), values))}
    advantage = [row["context"]["full_context_advantage_vs_1"] for row in rows_]
    result["full_context_advantage"] = {"mean": mean(advantage), "std": stdev(advantage), "by_seed": dict(zip(map(str, SEEDS), advantage))}
    result["frequency"] = {}
    for bucket in FREQUENCY_BUCKETS:
        result["frequency"][bucket] = {}
        for field in ("cross_entropy", "mean_correct_token_probability", "median_correct_token_probability", "top_1_accuracy", "top_5_accuracy", "top_10_accuracy"):
            values = metric(rows_, "validation", "frequency_buckets", bucket, field)
            result["frequency"][bucket][field] = {"mean": mean(values), "std": stdev(values), "by_seed": dict(zip(map(str, SEEDS), values))}
    result["teacher_forced_horizons"] = {}
    for horizon in ("1", "2", "4", "8", "16", "32"):
        result["teacher_forced_horizons"][horizon] = {}
        for field in ("loss", "accuracy", "correct_token_probability"):
            values = metric(rows_, "teacher_forced_horizons", horizon, field)
            result["teacher_forced_horizons"][horizon][field] = {
                "mean": mean(values),
                "std": stdev(values),
                "by_seed": dict(zip(map(str, SEEDS), values)),
            }
    return result


def training(gate: int) -> dict:
    payload = read(f"evaluation/foundation-v35-gate{gate}-training.json")["results"]
    checks = {
        "three_seeds": [row["seed"] for row in payload] == list(SEEDS),
        "all_finite": all(row["training"]["all_finite"] for row in payload),
        "checkpoint_integrity": all(row["checkpoint"]["integrity"]["pass"] for row in payload),
        "resume_preflight": all(row["resume_preflight"]["pass"] for row in payload),
        "parallel_cpu_evaluation_disabled": all(row["parallel_cpu_evaluation"] == "DISABLED" for row in payload),
        "settings_unchanged": all(not row["settings_changed"] for row in payload),
    }
    thermal = [row["training"]["telemetry"] for row in payload]
    return {
        "checks": checks, "pass": all(checks.values()), "rows": payload,
        "mean_tokens_per_second": mean(row["training"]["tokens_per_second"] for row in payload),
        "peak_vram_mib": max(row["training"]["peak_vram_mib"] for row in payload),
        "max_temperature_c": max(row["gpu_temperature_c_max"] for row in thermal),
        "thermal_classifications": [row["thermal_classification"] for row in thermal],
        "hardware_thermal_throttling": any(row["hardware_thermal_slowdown"] for row in thermal),
        "software_thermal_slowdown": any(row["software_thermal_slowdown"] for row in thermal),
        "cooldown_all_reached": all(row["cooldown"]["target_reached"] for row in payload),
    }


def gate_decision(baseline: list[dict], candidate: list[dict], gate: int) -> dict:
    before, after = aggregate(baseline), aggregate(candidate)
    loss_delta = after["validation_loss"]["mean"] - before["validation_loss"]["mean"]
    top_deltas = {name: after[name]["mean"] - before[name]["mean"] for name in ("top1", "top5", "top10")}
    context_delta = after["context"]["512"]["mean"] - before["context"]["512"]["mean"]
    middle = after["frequency"]["middle_20_to_80_percent"]
    rare = after["frequency"]["rare_bottom_20_percent"]
    before_middle = before["frequency"]["middle_20_to_80_percent"]
    before_rare = before["frequency"]["rare_bottom_20_percent"]
    teacher = all(
        after["teacher_forced_horizons"][horizon]["loss"]["mean"]
        <= before["teacher_forced_horizons"][horizon]["loss"]["mean"] + THRESHOLDS["teacher_forced_loss_max_increase"]
        for horizon in after["teacher_forced_horizons"]
    )
    training_result = training(gate)
    checks = {
        "validation": loss_delta <= THRESHOLDS["validation_mean_loss_max_increase"] and sum(
            after["validation_loss"]["by_seed"][str(seed)] - before["validation_loss"]["by_seed"][str(seed)] > THRESHOLDS["validation_single_seed_loss_max_increase"] for seed in SEEDS
        ) < 2,
        "topk": all(delta >= -THRESHOLDS["topk_mean_max_drop"] for delta in top_deltas.values()),
        "context": context_delta <= THRESHOLDS["context_mean_full_loss_max_increase"] and all(value >= THRESHOLDS["context_minimum_full_advantage"] for value in after["full_context_advantage"]["by_seed"].values()),
        "frequency": middle["cross_entropy"]["mean"] <= before_middle["cross_entropy"]["mean"] + THRESHOLDS["frequency_ce_max_increase"] and rare["cross_entropy"]["mean"] <= before_rare["cross_entropy"]["mean"] + THRESHOLDS["frequency_ce_max_increase"] and rare["median_correct_token_probability"]["mean"] >= before_rare["median_correct_token_probability"]["mean"] * THRESHOLDS["rare_median_probability_min_ratio"],
        "sampling": after["semantic"]["mean"] >= before["semantic"]["mean"] - THRESHOLDS["sampling_semantic_max_drop"] and after["naturalness"]["mean"] >= before["naturalness"]["mean"] - THRESHOLDS["sampling_naturalness_max_drop"],
        "eos": after["terminal_eos_probability"]["mean"] >= before["terminal_eos_probability"]["mean"] * THRESHOLDS["eos_terminal_min_ratio"] and after["premature_eos_top1"]["mean"] == THRESHOLDS["premature_eos_top1_required"],
        "teacher_forced": teacher,
        "training": training_result["pass"],
        "gpu": not training_result["hardware_thermal_throttling"],
    }
    passed = all(checks.values())
    if passed:
        decision = "CONTINUE_TO_16_384M" if gate == 1 else "GATE2_PASS"
    elif not checks["context"]:
        decision = "STOP_CONTEXT_GATE"
    elif not checks["sampling"]:
        decision = "STOP_GENERATION_GATE"
    elif not checks["frequency"]:
        decision = "STOP_FREQUENCY_GATE"
    elif not checks["eos"]:
        decision = "STOP_EOS_GATE"
    elif not checks["training"] or not checks["gpu"]:
        decision = "STOP_STABILITY_GATE"
    else:
        decision = "STOP_LM_GATE"
    return {"gate": gate, "baseline": before, "candidate": after, "deltas": {"validation_loss": loss_delta, "topk": top_deltas, "full_context_loss": context_delta}, "checks": checks, "training": training_result, "pass": passed, "decision": decision}


def historical_rows() -> list[tuple[int, dict]]:
    # Phase 44 predates the detailed greedy-repetition fields introduced by
    # Phase 46.  Rolling gates use only validation loss and top-1, so retain
    # exactly those shared measurements rather than inventing missing values.
    def historical_aggregate(stage_rows: list[dict]) -> dict:
        result = {}
        for name, path in (
            ("validation_loss", ("validation", "loss")),
            ("top1", ("validation", "top_1_accuracy")),
        ):
            values = metric(stage_rows, *path)
            result[name] = {
                "mean": mean(values),
                "std": stdev(values),
                "by_seed": dict(zip(map(str, SEEDS), values)),
            }
        return result

    rows_ = []
    for tokens, stage in ((15_360_000, "baseline"), (15_616_000, "gate1"), (15_872_000, "gate2")):
        stage_rows = [read(f"evaluation/phase44/{stage}/seed-{seed}.json") for seed in SEEDS]
        rows_.append((tokens, historical_aggregate(stage_rows)))
    return rows_


def classify_attractor(before: dict, after: dict) -> dict:
    """Classify direction from the preregistered greedy tracking signals."""
    weakening, worsening = [], []
    if after["median_loop_onset"]["mean"] >= before["median_loop_onset"]["mean"] + 2.0:
        weakening.append("later_loop_onset")
    elif after["median_loop_onset"]["mean"] <= before["median_loop_onset"]["mean"] - 2.0:
        worsening.append("earlier_loop_onset")
    if after["greedy_repetition_1"]["mean"] <= before["greedy_repetition_1"]["mean"] - 0.01:
        weakening.append("lower_repetition")
    elif after["greedy_repetition_1"]["mean"] >= before["greedy_repetition_1"]["mean"] + 0.01:
        worsening.append("higher_repetition")
    if after["loop_margin"]["mean"] <= before["loop_margin"]["mean"] - 0.01:
        weakening.append("lower_confidence_margin")
    elif after["loop_margin"]["mean"] >= before["loop_margin"]["mean"] + 0.01:
        worsening.append("higher_confidence_margin")
    if after["loop_entropy"]["mean"] >= before["loop_entropy"]["mean"] + 0.10:
        weakening.append("higher_loop_entropy")
    elif after["loop_entropy"]["mean"] <= before["loop_entropy"]["mean"] - 0.10:
        worsening.append("lower_loop_entropy")
    if len(weakening) >= 2 and len(weakening) > len(worsening):
        label = "WEAKENING"
    elif len(worsening) >= 2 and len(worsening) > len(weakening):
        label = "WORSENING"
    else:
        label = "STATIC"
    return {"label": label, "weakening_signals": weakening, "worsening_signals": worsening}


def final_artifacts(pytest_result: str) -> tuple[dict, str]:
    baseline = rows("baseline")
    gate1_rows = rows("gate1")
    gate1 = gate_decision(baseline, gate1_rows, 1)
    gate2_paths = [ROOT / f"evaluation/phase46/gate2/seed-{seed}.json" for seed in SEEDS]
    gate2 = None
    final = gate1_rows
    if all(path.exists() for path in gate2_paths):
        gate2_rows = rows("gate2")
        gate2 = gate_decision(gate1_rows, gate2_rows, 2)
        final = gate2_rows
    history = historical_rows() + [(16_128_000, aggregate(gate1_rows))]
    if gate2:
        history.append((16_384_000, aggregate(final)))
    final_metrics = aggregate(final)
    rolling = {}
    if len(history) >= 3:
        rolling["512k"] = {"validation_loss_change": history[-1][1]["validation_loss"]["mean"] - history[-3][1]["validation_loss"]["mean"], "top1_change": history[-1][1]["top1"]["mean"] - history[-3][1]["top1"]["mean"]}
    if len(history) >= 5:
        rolling["1024k"] = {"validation_loss_change": history[-1][1]["validation_loss"]["mean"] - history[-5][1]["validation_loss"]["mean"], "top1_change": history[-1][1]["top1"]["mean"] - history[-5][1]["top1"]["mean"]}
    if not gate1["pass"]:
        final_gate = {"STOP_CONTEXT_GATE": "CONTEXT_REVIEW_REQUIRED", "STOP_GENERATION_GATE": "GENERATION_TRADEOFF_REVIEW", "STOP_STABILITY_GATE": "THERMAL_REVIEW_REQUIRED"}.get(gate1["decision"], "LM_PLATEAU_REVIEW")
    elif gate2 is None:
        final_gate = "CONTINUE_SHORT_GPU_GATES_EOS_1_5"
    elif not gate2["pass"]:
        final_gate = {"STOP_CONTEXT_GATE": "CONTEXT_REVIEW_REQUIRED", "STOP_GENERATION_GATE": "GENERATION_TRADEOFF_REVIEW", "STOP_STABILITY_GATE": "THERMAL_REVIEW_REQUIRED"}.get(gate2["decision"], "LM_PLATEAU_REVIEW")
    elif rolling.get("512k", {}).get("validation_loss_change", 0) < -0.01 and rolling.get("1024k", {}).get("validation_loss_change", 0) < -0.02 and all(value >= 0 for value in (rolling.get("512k", {}).get("top1_change", -1), rolling.get("1024k", {}).get("top1_change", -1))):
        final_gate = "CONTINUE_TO_18M_EOS_1_5"
    else:
        final_gate = "CONTINUE_SHORT_GPU_GATES_EOS_1_5"
    blind = sha(ROOT / "data/foundation_v09/evaluation/final-blind-1000.json")
    baseline_metrics = aggregate(baseline)
    summary = {"schema": "foundation-v35-short-gate-summary-v1", "phase": 46, "thresholds": THRESHOLDS, "baseline_15_872m": baseline_metrics, "gate1_executed": True, "gate1": gate1, "gate2_executed": gate2 is not None, "gate2": gate2, "final_metrics": final_metrics, "historical_trend": {str(tokens): values for tokens, values in history}, "rolling_trend": rolling, "context_regression": not gate1["checks"]["context"] or (gate2 is not None and not gate2["checks"]["context"]), "attractor": classify_attractor(baseline_metrics, final_metrics), "final_gate": final_gate, "recommended_next_token_target": 18_000_000 if final_gate == "CONTINUE_TO_18M_EOS_1_5" else (16_640_000 if final_gate == "CONTINUE_SHORT_GPU_GATES_EOS_1_5" else None), "continue_20m_permission": False, "formal_recipe": {"device": "CUDA", "precision": "FP32", "eos_weight": 1.5, "repetition_auxiliary": False, "amp": False}, "parallel_cpu_evaluation": "DISABLED", "corpus_exposure_percent": 100 * final[0]["tokens_processed"] / 33_402_759, "foundation_base_complete": False, "checkpoint_integrity": gate1["training"]["checks"]["checkpoint_integrity"] and (gate2 is None or gate2["training"]["checks"]["checkpoint_integrity"]), "final_blind": {"opened": False, "sha256": blind, "pass": blind == BLIND_SHA}, "pytest": pytest_result, "render_changed": False, "vercel_changed": False}
    report = render(summary)
    return summary, report


def render(summary: dict) -> str:
    final = summary["final_metrics"]
    lines = ["# Foundation v3.5 Thermal-Aware Short GPU Continuation", "", "## Decision", "", f"- Gate 1: **{summary['gate1']['decision']}**", f"- Gate 2 executed: **{'YES' if summary['gate2_executed'] else 'NO'}**", f"- Gate 2: **{summary['gate2']['decision'] if summary['gate2'] else 'NOT_EXECUTED'}**", f"- Final Gate: **{summary['final_gate']}**", "- 20M permission: **NO**", "- Foundation Base completion: **NO**", "", "## Final metrics", "", f"- Validation loss / std: {final['validation_loss']['mean']:.4f} / {final['validation_loss']['std']:.4f}", f"- Top-1 / Top-5 / Top-10: {100*final['top1']['mean']:.2f}% / {100*final['top5']['mean']:.2f}% / {100*final['top10']['mean']:.2f}%", f"- Sampling Naturalness / Semantic: {100*final['naturalness']['mean']:.2f}% / {100*final['semantic']['mean']:.2f}%", f"- terminal P(EOS): {final['terminal_eos_probability']['mean']:.5f}; premature EOS Top-1: {100*final['premature_eos_top1']['mean']:.2f}%", f"- Greedy runaway / loop onset / repetition-1: {100*final['greedy_runaway']['mean']:.2f}% / {final['median_loop_onset']['mean']:.1f} / {final['greedy_repetition_1']['mean']:.4f}", f"- Attractor: {summary['attractor']['label']} ({summary['attractor']})", f"- Full context loss / advantage: {final['context']['512']['mean']:.4f} / {final['full_context_advantage']['mean']:.4f}", f"- Context regression: {'YES' if summary['context_regression'] else 'NO'}", "", "## Operations", "", f"- Rolling 512k: {summary['rolling_trend'].get('512k')}", f"- Rolling 1.024M: {summary['rolling_trend'].get('1024k')}", f"- Corpus exposure: {summary['corpus_exposure_percent']:.2f}%", "- CUDA FP32, EOS weight 1.5, repetition auxiliary OFF, AMP OFF", "- Parallel CPU evaluation: DISABLED", f"- Checkpoint integrity: {'PASS' if summary['checkpoint_integrity'] else 'FAIL'}", f"- Final Blind: unopened; SHA256 {'PASS' if summary['final_blind']['pass'] else 'FAIL'}", f"- pytest: {summary['pytest']}", "- Render/Vercel: unchanged"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preregister", "final"), required=True)
    parser.add_argument("--pytest-result", default="pending")
    args = parser.parse_args()
    if args.mode == "preregister":
        (ROOT / "evaluation/foundation-v35-gate-thresholds.json").write_text(json.dumps(THRESHOLDS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"preregistered": True, "thresholds": THRESHOLDS}))
        return
    summary, report = final_artifacts(args.pytest_result)
    (ROOT / "evaluation/foundation-v35-short-gate-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "evaluation/foundation-v35-short-gate-report.md").write_text(report, encoding="utf-8")
    (ROOT / "evaluation/foundation-v35-rolling-trends.json").write_text(json.dumps(summary["historical_trend"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gate": summary["final_gate"], "gate1": summary["gate1"]["decision"], "gate2": summary["gate2"]["decision"] if summary["gate2"] else None}))


if __name__ == "__main__":
    main()
