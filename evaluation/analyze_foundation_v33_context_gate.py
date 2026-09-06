"""PHASE 44 multi-seed context gates and final continuation decision."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2026)
CONTEXT_LENGTHS = (512, 256, 128, 64, 32, 16, 8, 2, 1)
MATERIAL_ABSOLUTE_DELTA = 0.05
MATERIAL_RELATIVE_DELTA = 0.01
CONFIDENCE_Z = 1.96
MINIMUM_FULL_ADVANTAGE = 0.10
CORPUS_TOKENS = 33_402_759
FINAL_BLIND_SHA256 = "fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b"


def load_json(path: str | Path) -> dict:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return json.loads(candidate.read_text(encoding="utf-8"))


def sha256_only(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_rows(stage: str) -> list[dict]:
    return [load_json(f"evaluation/phase44/{stage}/seed-{seed}.json") for seed in SEEDS]


def aggregate_context(rows: list[dict]) -> dict:
    result = {}
    for context in CONTEXT_LENGTHS:
        values = [row["context"][str(context)]["loss"] for row in rows]
        result[str(context)] = {
            "mean_loss": mean(values),
            "std_across_seeds": stdev(values),
            "seed_losses": {str(row["seed"]): value for row, value in zip(rows, values)},
        }
    advantages = [row["context"]["full_context_advantage_vs_1"] for row in rows]
    result["full_context_advantage_vs_1"] = {
        "mean": mean(advantages),
        "std_across_seeds": stdev(advantages),
        "by_seed": {str(row["seed"]): value for row, value in zip(rows, advantages)},
    }
    return result


def context_gate(baseline: list[dict], candidate: list[dict]) -> dict:
    baseline_mean = mean(row["context"]["512"]["loss"] for row in baseline)
    candidate_mean = mean(row["context"]["512"]["loss"] for row in candidate)
    seed_deltas = {
        str(left["seed"]): right["context"]["512"]["loss"]
        - left["context"]["512"]["loss"]
        for left, right in zip(baseline, candidate)
    }
    paired = [
        after - before
        for left, right in zip(baseline, candidate)
        for before, after in zip(
            left["context"]["512"]["per_target_losses"],
            right["context"]["512"]["per_target_losses"],
        )
    ]
    paired_mean = mean(paired)
    paired_sem = stdev(paired) / math.sqrt(len(paired))
    confidence = [paired_mean - CONFIDENCE_Z * paired_sem, paired_mean + CONFIDENCE_Z * paired_sem]
    absolute_delta = candidate_mean - baseline_mean
    relative_delta = absolute_delta / baseline_mean
    material_threshold = max(
        MATERIAL_ABSOLUTE_DELTA, MATERIAL_RELATIVE_DELTA * baseline_mean
    )
    positive_seed_deltas = sum(value > 0 for value in seed_deltas.values())
    material = absolute_delta > material_threshold and relative_delta > MATERIAL_RELATIVE_DELTA
    statistical = confidence[0] > 0
    multi_seed = positive_seed_deltas >= 2
    regression = material and statistical and multi_seed
    single_seed_anomaly = (
        positive_seed_deltas < 2
        and any(value > max(0.10, 2 * material_threshold) for value in seed_deltas.values())
    )
    advantages = [row["context"]["full_context_advantage_vs_1"] for row in candidate]
    advantage_maintained = all(value >= MINIMUM_FULL_ADVANTAGE for value in advantages)
    aggregated = aggregate_context(candidate)
    full = aggregated["512"]["mean_loss"]
    relationship_checks = {
        "full_not_materially_worse_than_64": full <= aggregated["64"]["mean_loss"] + 0.05,
        "full_not_materially_worse_than_16": full <= aggregated["16"]["mean_loss"] + 0.05,
        "full_better_than_2": full < aggregated["2"]["mean_loss"],
        "full_better_than_1": full < aggregated["1"]["mean_loss"],
    }
    sanity = all(row["sanity"]["pass"] and row["checkpoint_unchanged"] for row in candidate)
    passed = (
        not regression
        and not single_seed_anomaly
        and advantage_maintained
        and all(relationship_checks.values())
        and sanity
    )
    return {
        "threshold": {
            "material_absolute_delta": MATERIAL_ABSOLUTE_DELTA,
            "material_relative_delta": MATERIAL_RELATIVE_DELTA,
            "confidence_z": CONFIDENCE_Z,
            "minimum_full_advantage": MINIMUM_FULL_ADVANTAGE,
            "regression_requires": (
                "mean delta above max(0.05, 1% baseline), paired 95% CI lower bound above zero, "
                "and positive deltas in at least two of three seeds"
            ),
        },
        "baseline_full_mean": baseline_mean,
        "candidate_full_mean": candidate_mean,
        "absolute_delta": absolute_delta,
        "relative_delta": relative_delta,
        "seed_deltas": seed_deltas,
        "positive_seed_deltas": positive_seed_deltas,
        "paired_targets": len(paired),
        "paired_delta_mean": paired_mean,
        "paired_delta_sem": paired_sem,
        "paired_delta_95pct_ci": confidence,
        "material_regression": material,
        "statistically_positive": statistical,
        "multi_seed_reproduction": multi_seed,
        "single_seed_anomaly": single_seed_anomaly,
        "context_regression": regression,
        "full_advantage_by_seed": {
            str(row["seed"]): value for row, value in zip(candidate, advantages)
        },
        "full_advantage_maintained": advantage_maintained,
        "relationship_checks": relationship_checks,
        "sanity_pass": sanity,
        "pass": passed,
    }


def _mean(rows: list[dict], *path: str) -> float:
    values = []
    for row in rows:
        value = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return mean(values)


def lm_gate(baseline: list[dict], candidate: list[dict]) -> dict:
    baseline_loss = _mean(baseline, "validation", "loss")
    candidate_loss = _mean(candidate, "validation", "loss")
    top_checks = {
        key: _mean(candidate, "validation", key)
        >= _mean(baseline, "validation", key) - 0.002
        for key in ("top_1_accuracy", "top_5_accuracy", "top_10_accuracy")
    }
    semantic_before = _mean(baseline, "generation", "temperature_0.7", "semantic_rate")
    semantic_after = _mean(candidate, "generation", "temperature_0.7", "semantic_rate")
    natural_before = _mean(baseline, "generation", "temperature_0.7", "naturalness_rate")
    natural_after = _mean(candidate, "generation", "temperature_0.7", "naturalness_rate")
    frequency = {}
    for bucket in ("middle_20_to_80_percent", "rare_bottom_20_percent"):
        before_ce = _mean(baseline, "validation", "frequency_buckets", bucket, "cross_entropy")
        after_ce = _mean(candidate, "validation", "frequency_buckets", bucket, "cross_entropy")
        before_probability = _mean(
            baseline,
            "validation",
            "frequency_buckets",
            bucket,
            "mean_correct_token_probability",
        )
        after_probability = _mean(
            candidate,
            "validation",
            "frequency_buckets",
            bucket,
            "mean_correct_token_probability",
        )
        maintained = after_ce <= before_ce + 0.05 or after_probability >= before_probability * 0.95
        frequency[bucket] = {
            "cross_entropy_before": before_ce,
            "cross_entropy_after": after_ce,
            "probability_before": before_probability,
            "probability_after": after_probability,
            "maintained_or_improved": maintained,
        }
    horizon_checks = {
        horizon: _mean(candidate, "teacher_forced_horizons", horizon, "loss")
        <= _mean(baseline, "teacher_forced_horizons", horizon, "loss") + 0.10
        for horizon in ("1", "2", "4", "8", "16", "32")
    }
    checks = {
        "validation_loss_improved": candidate_loss < baseline_loss,
        "topk_maintained_or_improved": all(top_checks.values()),
        "semantic_maintained": semantic_after >= semantic_before - 0.03,
        "naturalness_japanese_maintained": natural_after >= natural_before - 0.05,
        "middle_maintained_or_improved": frequency["middle_20_to_80_percent"]["maintained_or_improved"],
        "rare_maintained_or_improved": frequency["rare_bottom_20_percent"]["maintained_or_improved"],
        "teacher_forced_horizons_maintained": all(horizon_checks.values()),
        "finite": all(
            math.isfinite(_mean(candidate, "validation", key))
            for key in ("loss", "perplexity")
        ),
    }
    return {
        "validation_loss_before": baseline_loss,
        "validation_loss_after": candidate_loss,
        "validation_loss_delta": candidate_loss - baseline_loss,
        "topk_checks": top_checks,
        "semantic_before": semantic_before,
        "semantic_after": semantic_after,
        "naturalness_before": natural_before,
        "naturalness_after": natural_after,
        "frequency": frequency,
        "teacher_forced_horizon_checks": horizon_checks,
        "checks": checks,
        "pass": all(checks.values()),
    }


def eos_gate(baseline: list[dict], candidate: list[dict]) -> dict:
    terminal_before = _mean(baseline, "terminal_eos", "mean_probability")
    terminal_after = _mean(candidate, "terminal_eos", "mean_probability")
    premature_top1 = max(
        row["nonterminal_eos"]["top1_rate"] for row in candidate
    )
    return {
        "terminal_probability_before": terminal_before,
        "terminal_probability_after": terminal_after,
        "terminal_maintained": terminal_after >= terminal_before * 0.90,
        "premature_eos_top1": premature_top1,
        "premature_eos_safe": premature_top1 == 0,
        "pass": terminal_after >= terminal_before * 0.90 and premature_top1 == 0,
    }


def training_gate(gate: int) -> dict:
    payload = load_json(f"evaluation/foundation-v33-gate{gate}-training.json")
    results = payload["results"]
    checks = {
        "three_seeds": [row["seed"] for row in results] == list(SEEDS),
        "all_finite": all(row["training"]["all_finite"] for row in results),
        "checkpoint_integrity": all(row["checkpoint"]["integrity"]["pass"] for row in results),
        "parallel_cpu_evaluation_disabled": all(
            row["parallel_cpu_evaluation"] == "DISABLED" for row in results
        ),
    }
    telemetry = [row["training"]["telemetry"] for row in results]
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "mean_tokens_per_second": mean(row["training"]["tokens_per_second"] for row in results),
        "peak_vram_mib": max(row["training"]["peak_vram_mib"] for row in results),
        "max_gpu_temperature_c": max(row.get("gpu_temperature_c_max", 0) for row in telemetry),
        "max_longest_above_80_seconds": max(
            row.get("longest_above_80_seconds", 0) for row in telemetry
        ),
        "thermal_attention": any(row.get("thermal_attention", False) for row in telemetry),
        "rows": results,
    }


def gate_decision(gate: int, baseline: list[dict], candidate: list[dict]) -> dict:
    context = context_gate(baseline, candidate)
    lm = lm_gate(baseline, candidate)
    eos = eos_gate(baseline, candidate)
    training = training_gate(gate)
    passed = context["pass"] and lm["pass"] and eos["pass"] and training["pass"]
    if passed:
        decision = "CONTINUE_PLUS_256K"
    elif not context["pass"]:
        decision = "STOP_CONTEXT_REGRESSION"
    elif not lm["pass"]:
        decision = "STOP_LM_GATE"
    elif not eos["pass"]:
        decision = "STOP_EOS_GATE"
    else:
        decision = "STOP_TRAINING_INTEGRITY"
    return {
        "gate": gate,
        "context": context,
        "lm": lm,
        "eos": eos,
        "training": training,
        "pass": passed,
        "decision": decision,
    }


def aggregate_endpoint(rows: list[dict]) -> dict:
    return {
        "validation_loss": _mean(rows, "validation", "loss"),
        "perplexity": _mean(rows, "validation", "perplexity"),
        "top1": _mean(rows, "validation", "top_1_accuracy"),
        "top5": _mean(rows, "validation", "top_5_accuracy"),
        "top10": _mean(rows, "validation", "top_10_accuracy"),
        "sampling_naturalness": _mean(
            rows, "generation", "temperature_0.7", "naturalness_rate"
        ),
        "semantic_coherence": _mean(
            rows, "generation", "temperature_0.7", "semantic_rate"
        ),
        "sampling_repetition": _mean(
            rows, "generation", "temperature_0.7", "mean_repetition_1"
        ),
        "sampling_completion": _mean(
            rows, "generation", "temperature_0.7", "sentence_completion_rate"
        ),
        "terminal_eos_probability": _mean(rows, "terminal_eos", "mean_probability"),
        "premature_eos_top1": max(row["nonterminal_eos"]["top1_rate"] for row in rows),
        "greedy_repetition": _mean(rows, "generation", "greedy", "mean_repetition_1"),
        "greedy_runaway": _mean(rows, "generation", "greedy", "runaway_rate"),
        "median_loop_onset": mean(
            row["generation"]["greedy"]["median_loop_onset"] for row in rows
        ),
        "context": aggregate_context(rows),
    }


def final_artifacts(pytest_result: str) -> tuple[dict, dict, str]:
    baseline = stage_rows("baseline")
    gate1_rows = stage_rows("gate1")
    gate1 = gate_decision(1, baseline, gate1_rows)
    gate2_available = all(
        (ROOT / f"evaluation/phase44/gate2/seed-{seed}.json").exists() for seed in SEEDS
    )
    gate2 = None
    final_rows = gate1_rows
    if gate2_available:
        gate2_rows = stage_rows("gate2")
        gate2 = gate_decision(2, gate1_rows, gate2_rows)
        final_rows = gate2_rows
    baseline_endpoint = aggregate_endpoint(baseline)
    gate1_endpoint = aggregate_endpoint(gate1_rows)
    final_endpoint = aggregate_endpoint(final_rows)
    final_tokens = int(final_rows[0]["tokens_processed"])
    overall_context = context_gate(baseline, final_rows)
    overall_lm = lm_gate(baseline, final_rows)
    overall_eos = eos_gate(baseline, final_rows)
    phase43 = load_json("evaluation/foundation-v32-base-maturity-decision-summary.json")
    training_stable = gate1["training"]["pass"] and (
        gate2 is None or gate2["training"]["pass"]
    )
    twenty_m_requirements = {
        "three_seed_context_maintained": overall_context["pass"],
        "validation_improved": final_endpoint["validation_loss"]
        < baseline_endpoint["validation_loss"],
        "topk_improved": all(
            final_endpoint[key] >= baseline_endpoint[key]
            for key in ("top1", "top5", "top10")
        ),
        "semantic_maintained_or_improved": final_endpoint["semantic_coherence"]
        >= baseline_endpoint["semantic_coherence"],
        "middle_rare_maintained_or_improved": (
            overall_lm["checks"]["middle_maintained_or_improved"]
            and overall_lm["checks"]["rare_maintained_or_improved"]
        ),
        "eos_safe": overall_eos["pass"],
        "checkpoint_stable": training_stable,
        "no_architecture_defect_evidence": not phase43["architecture_defect_evidence"],
        "no_nan_or_inf": overall_lm["checks"]["finite"],
        "both_256k_intervals_passed": gate2 is not None and gate1["pass"] and gate2["pass"],
    }
    permission = all(twenty_m_requirements.values())
    context_regression = overall_context["context_regression"]
    if not training_stable or not overall_eos["pass"]:
        final_gate = "STOP_AND_INVESTIGATE"
    elif not overall_context["pass"]:
        final_gate = (
            "CONTEXT_CAPABILITY_TRADEOFF"
            if overall_lm["pass"]
            else "CONTEXT_REGRESSION_INVESTIGATE"
        )
    elif permission:
        final_gate = "CONTINUE_20M_GPU_WITH_EOS_1_5"
    elif overall_lm["pass"]:
        final_gate = "CONTINUE_SHORT_GPU_GATES"
    elif final_endpoint["validation_loss"] >= baseline_endpoint["validation_loss"]:
        final_gate = "TRAINING_PLATEAU"
    else:
        final_gate = "STOP_AND_INVESTIGATE"

    history_available = all(
        (ROOT / f"evaluation/phase44/history-10240/seed-{seed}.json").exists()
        for seed in SEEDS
    )
    context_curves = {
        "schema": "foundation-v33-context-curves-v1",
        "threshold": gate1["context"]["threshold"],
        "stages": {
            "15_360_000": aggregate_context(baseline),
            "15_616_000": aggregate_context(gate1_rows),
        },
        "gate1_delta": gate1["context"],
    }
    if history_available:
        context_curves["stages"]["10_240_000"] = aggregate_context(
            stage_rows("history-10240")
        )
    if gate2 is not None:
        context_curves["stages"]["15_872_000"] = aggregate_context(final_rows)
        context_curves["gate2_delta"] = gate2["context"]

    blind_hash = sha256_only(ROOT / "data/foundation_v09/evaluation/final-blind-1000.json")
    thermal = gate2["training"] if gate2 is not None else gate1["training"]
    summary = {
        "schema": "foundation-v33-context-gate-summary-v1",
        "phase": 44,
        "baseline_context": aggregate_context(baseline),
        "context_targets_per_seed": 256,
        "context_target_positions_sha256": baseline[0]["context_target_positions_sha256"],
        "context_metric_sanity": all(row["sanity"]["pass"] for row in baseline + final_rows),
        "gate1": gate1,
        "gate1_executed": True,
        "gate1_decision": gate1["decision"],
        "gate2_executed": gate2 is not None,
        "gate2": gate2,
        "overall_512k": {
            "context": overall_context,
            "lm": overall_lm,
            "eos": overall_eos,
        },
        "baseline_endpoint": baseline_endpoint,
        "final_endpoint": final_endpoint,
        "scaling_per_256k": {
            "gate1": {
                "loss_improvement": baseline_endpoint["validation_loss"]
                - gate1_endpoint["validation_loss"],
                "top1_improvement": gate1_endpoint["top1"] - baseline_endpoint["top1"],
                "semantic_improvement": gate1_endpoint["semantic_coherence"]
                - baseline_endpoint["semantic_coherence"],
                "full_context_delta": gate1["context"]["absolute_delta"],
            }
        },
        "context_regression": context_regression,
        "context_vs_lm_tradeoff": final_gate == "CONTEXT_CAPABILITY_TRADEOFF",
        "architecture_defect_evidence": phase43["architecture_defect_evidence"],
        "eos_weight": 1.5,
        "repetition_auxiliary": False,
        "final_tokens": final_tokens,
        "corpus_exposure_percent": 100 * final_tokens / CORPUS_TOKENS,
        "gpu": thermal,
        "parallel_cpu_evaluation": "DISABLED",
        "final_gate": final_gate,
        "continue_20m_permission": permission,
        "twenty_m_requirements": twenty_m_requirements,
        "foundation_base_complete": False,
        "checkpoint_integrity": {
            "gate1": gate1["training"]["checks"]["checkpoint_integrity"],
            "gate2": gate2["training"]["checks"]["checkpoint_integrity"]
            if gate2 is not None
            else None,
            "formal_15_360m_sources_unchanged": all(
                row["checkpoint_unchanged"] for row in baseline
            ),
            "final_blind": {
                "opened": False,
                "expected_sha256": FINAL_BLIND_SHA256,
                "actual_sha256": blind_hash,
                "pass": blind_hash == FINAL_BLIND_SHA256,
            },
        },
        "pytest": pytest_result,
        "render_changed": False,
        "vercel_changed": False,
    }
    if gate2 is not None:
        summary["scaling_per_256k"]["gate2"] = {
            "loss_improvement": gate1_endpoint["validation_loss"]
            - final_endpoint["validation_loss"],
            "top1_improvement": final_endpoint["top1"] - gate1_endpoint["top1"],
            "semantic_improvement": final_endpoint["semantic_coherence"]
            - gate1_endpoint["semantic_coherence"],
            "full_context_delta": gate2["context"]["absolute_delta"],
        }
    if history_available:
        summary["historical_context_10_240m"] = aggregate_context(
            stage_rows("history-10240")
        )
    report = render_report(summary)
    return summary, context_curves, report


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_report(summary: dict) -> str:
    baseline = summary["baseline_context"]
    endpoint = summary["final_endpoint"]
    gate1 = summary["gate1"]
    gate2 = summary["gate2"]
    overall = summary["overall_512k"]
    lines = [
        "# Foundation v3.3 Context Gate Recovery",
        "",
        "## Decision",
        "",
        f"- Gate 1: **{summary['gate1_decision']}**",
        f"- Gate 2 executed: **{'YES' if summary['gate2_executed'] else 'NO'}**",
        f"- Gate 2 interval: **{gate2['decision'] if gate2 else 'NOT_EXECUTED'}**",
        f"- Final Gate: **{summary['final_gate']}**",
        f"- 20M permission: **{'YES' if summary['continue_20m_permission'] else 'NO'}**",
        "- Foundation Base completion: **NO**",
        "",
        "The pre-registered context regression rule requires an absolute and relative material delta, a paired "
        "95% confidence interval above zero, and reproduction in at least two seeds. A positive Full Context "
        "Advantage and natural ordering against short contexts are separate mandatory checks.",
        "",
        "## Expanded multi-seed baseline",
        "",
        "| Context | Mean loss | Seed std |",
        "|---:|---:|---:|",
    ]
    for context in CONTEXT_LENGTHS:
        row = baseline[str(context)]
        lines.append(f"| {context} | {row['mean_loss']:.4f} | {row['std_across_seeds']:.4f} |")
    lines += [
        "",
        f"Full Context Advantage vs 1: {baseline['full_context_advantage_vs_1']['mean']:.4f} "
        f"± {baseline['full_context_advantage_vs_1']['std_across_seeds']:.4f} across seeds.",
        "",
        "## Gate 1",
        "",
        f"- Full loss: {gate1['context']['baseline_full_mean']:.4f} → {gate1['context']['candidate_full_mean']:.4f} "
        f"(delta {gate1['context']['absolute_delta']:+.4f}, {100*gate1['context']['relative_delta']:+.2f}%)",
        f"- Paired 95% CI: [{gate1['context']['paired_delta_95pct_ci'][0]:+.4f}, "
        f"{gate1['context']['paired_delta_95pct_ci'][1]:+.4f}] over {gate1['context']['paired_targets']} targets",
        f"- Context regression: {'YES' if gate1['context']['context_regression'] else 'NO'}",
        f"- LM Gate: {'PASS' if gate1['lm']['pass'] else 'FAIL'}; EOS Gate: {'PASS' if gate1['eos']['pass'] else 'FAIL'}",
    ]
    if gate2 is not None:
        lines += [
            "",
            "## Gate 2",
            "",
            f"- Full loss: {gate2['context']['baseline_full_mean']:.4f} → {gate2['context']['candidate_full_mean']:.4f} "
            f"(delta {gate2['context']['absolute_delta']:+.4f}, {100*gate2['context']['relative_delta']:+.2f}%)",
            f"- Paired 95% CI: [{gate2['context']['paired_delta_95pct_ci'][0]:+.4f}, "
            f"{gate2['context']['paired_delta_95pct_ci'][1]:+.4f}] over {gate2['context']['paired_targets']} targets",
            f"- Context regression: {'YES' if gate2['context']['context_regression'] else 'NO'}",
            f"- LM Gate: {'PASS' if gate2['lm']['pass'] else 'FAIL'}; EOS Gate: {'PASS' if gate2['eos']['pass'] else 'FAIL'}",
            f"- Interval Top-1 check: {'PASS' if gate2['lm']['topk_checks']['top_1_accuracy'] else 'FAIL'}; "
            f"Semantic: {gate2['lm']['semantic_before']:.4f} → {gate2['lm']['semantic_after']:.4f}",
        ]
    lines += [
        "",
        "## Overall 15.360M → final",
        "",
        f"- Context / LM / EOS: {'PASS' if overall['context']['pass'] else 'FAIL'} / "
        f"{'PASS' if overall['lm']['pass'] else 'FAIL'} / {'PASS' if overall['eos']['pass'] else 'FAIL'}",
        f"- Full-context delta: {overall['context']['absolute_delta']:+.4f}; "
        f"context regression: {'YES' if summary['context_regression'] else 'NO'}",
        "- 20M remains denied because every required condition must pass; "
        f"Gate-2 interval pass={'YES' if summary['twenty_m_requirements']['both_256k_intervals_passed'] else 'NO'}, "
        f"strict Semantic maintenance={'YES' if summary['twenty_m_requirements']['semantic_maintained_or_improved'] else 'NO'}.",
        "",
        "## Final endpoint",
        "",
        f"- Tokens: {summary['final_tokens']:,}",
        f"- Validation loss / PPL: {endpoint['validation_loss']:.4f} / {endpoint['perplexity']:.2f}",
        f"- Top-1 / Top-5 / Top-10: {_pct(endpoint['top1'])} / {_pct(endpoint['top5'])} / {_pct(endpoint['top10'])}",
        f"- Sampling Naturalness / Semantic: {_pct(endpoint['sampling_naturalness'])} / {_pct(endpoint['semantic_coherence'])}",
        f"- terminal P(EOS): {endpoint['terminal_eos_probability']:.5f}; premature EOS Top-1: {_pct(endpoint['premature_eos_top1'])}",
        f"- Greedy runaway / repetition / median onset: {_pct(endpoint['greedy_runaway'])} / "
        f"{endpoint['greedy_repetition']:.4f} / {endpoint['median_loop_onset']:.1f}",
        f"- Full context loss / advantage: {endpoint['context']['512']['mean_loss']:.4f} / "
        f"{endpoint['context']['full_context_advantage_vs_1']['mean']:.4f}",
        f"- Middle CE: {overall['lm']['frequency']['middle_20_to_80_percent']['cross_entropy_before']:.4f} → "
        f"{overall['lm']['frequency']['middle_20_to_80_percent']['cross_entropy_after']:.4f}; "
        f"Rare CE: {overall['lm']['frequency']['rare_bottom_20_percent']['cross_entropy_before']:.4f} → "
        f"{overall['lm']['frequency']['rare_bottom_20_percent']['cross_entropy_after']:.4f}",
        f"- Corpus exposure: {summary['corpus_exposure_percent']:.2f}%",
    ]
    if "historical_context_10_240m" in summary:
        history = summary["historical_context_10_240m"]
        lines += [
            "",
            "## Historical context comparison",
            "",
            f"- 10.240M full loss / advantage: {history['512']['mean_loss']:.4f} / "
            f"{history['full_context_advantage_vs_1']['mean']:.4f}",
            f"- 15.360M full loss / advantage: {baseline['512']['mean_loss']:.4f} / "
            f"{baseline['full_context_advantage_vs_1']['mean']:.4f}",
            f"- 15.872M full loss / advantage: {endpoint['context']['512']['mean_loss']:.4f} / "
            f"{endpoint['context']['full_context_advantage_vs_1']['mean']:.4f}",
        ]
    lines += [
        "",
        "## Operations",
        "",
        f"- GPU mean throughput: {summary['gpu']['mean_tokens_per_second']:.2f} tok/s",
        f"- Peak VRAM: {summary['gpu']['peak_vram_mib']:.2f} MiB",
        f"- Max temperature: {summary['gpu']['max_gpu_temperature_c']:.0f}C; longest >80C: "
        f"{summary['gpu']['max_longest_above_80_seconds']:.2f}s; "
        f"THERMAL_ATTENTION={'YES' if summary['gpu']['thermal_attention'] else 'NO'}",
        "- CUDA FP32, AMP OFF, EOS weight 1.5, repetition auxiliary OFF",
        "- Parallel CPU evaluation: DISABLED",
        f"- Checkpoint integrity: {'PASS' if summary['checkpoint_integrity']['gate1'] and (summary['checkpoint_integrity']['gate2'] is not False) and summary['checkpoint_integrity']['formal_15_360m_sources_unchanged'] else 'FAIL'}",
        f"- Final Blind: unopened; SHA256 {'PASS' if summary['checkpoint_integrity']['final_blind']['pass'] else 'FAIL'}",
        f"- pytest: {summary['pytest']}",
        "- Render/Vercel: unchanged",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gate1", "final"), required=True)
    parser.add_argument("--pytest-result", default="pending")
    args = parser.parse_args()
    baseline = stage_rows("baseline")
    gate1_rows = stage_rows("gate1")
    gate1 = gate_decision(1, baseline, gate1_rows)
    (ROOT / "evaluation/foundation-v33-gate1-decision.json").write_text(
        json.dumps(gate1, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"gate1": gate1["decision"], "pass": gate1["pass"]}), flush=True)
    if args.mode == "gate1":
        return
    summary, curves, report = final_artifacts(args.pytest_result)
    (ROOT / "evaluation/foundation-v33-context-gate-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "evaluation/foundation-v33-context-curves.json").write_text(
        json.dumps(curves, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "evaluation/foundation-v33-context-gate-report.md").write_text(
        report, encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "final_gate": summary["final_gate"],
                "permission": summary["continue_20m_permission"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
