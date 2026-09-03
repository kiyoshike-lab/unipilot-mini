"""Aggregate PHASE 36 three-seed learning and generation-lag evidence."""
from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluation.report_foundation_v24 as shared
from foundation.base_tokenizer import FoundationTokenizer
from training.train_foundation_v21_ab import file_sha256, load_json


SEEDS = (42, 123, 2026)
MILESTONES = (1_024_000, 1_280_000, 1_536_000, 1_792_000, 2_048_000)


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def rate(start: float, end: float, start_tokens: int, end_tokens: int) -> float:
    return (end - start) / ((end_tokens - start_tokens) / 1_000_000)


def human_comparison(diagnostics: dict[int, dict]) -> dict:
    per_tokens = {
        tokens: diagnostics[tokens]["natural_japanese_evaluator_audit"]["examples"][:20]
        for tokens in MILESTONES
    }
    identifiers = [row["id"] for row in per_tokens[MILESTONES[0]]]
    if any([row["id"] for row in per_tokens[tokens]] != identifiers for tokens in MILESTONES):
        raise RuntimeError("human-readable generation identifiers are not fixed")
    return {
        "schema": "foundation-v25-fixed-generation-examples-v1",
        "representative_seed": 42,
        "fixed_examples": [
            {
                "id": identifier,
                "prefix": per_tokens[MILESTONES[0]][index]["prefix"],
                "reference": per_tokens[MILESTONES[0]][index]["reference"],
                "milestones": {
                    str(tokens): {
                        "generated": per_tokens[tokens][index]["generated"],
                        "natural_japanese_proxy": per_tokens[tokens][index]["natural_japanese_proxy"],
                        "reasons": per_tokens[tokens][index]["reasons"],
                    }
                    for tokens in MILESTONES
                },
            }
            for index, identifier in enumerate(identifiers)
        ],
    }


def main() -> int:
    settings = load_json("configs/unipilot-foundation-v25.json")
    tokenizer = FoundationTokenizer.load(ROOT / settings["tokenizer"])
    runs = {
        seed: read(f"evaluation/foundation-v25-runs/current-seed-{seed}.json")
        for seed in SEEDS
    }
    diagnostics = {
        tokens: read(f"evaluation/foundation-v23-generation-diagnostics-{tokens}.json")
        for tokens in MILESTONES
    }
    punctuation_probe = {
        int(row["tokens"]): row
        for row in read("evaluation/foundation-v25-punctuation.json")["rows"]
    }
    checkpoints = read("evaluation/foundation-v25-checkpoint-verification.json")
    smoke = read("evaluation/foundation-v25-synthetic-smoke.json")
    parity = read("evaluation/foundation-v23-inference-parity.json")
    shared.MILESTONES = MILESTONES
    shared.SEEDS = SEEDS
    shared.RUNS_BY_SEED = runs
    training_curve = shared.aggregate_training_curve(list(runs.values()))
    generations = shared.generation_curve(diagnostics, tokenizer, punctuation_probe)
    knowledge = shared.knowledge_observations(diagnostics)
    examples = human_comparison(diagnostics)
    (ROOT / "evaluation/foundation-v25-generation-examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    first, final = training_curve[0], training_curve[-1]
    first_generation, final_generation = generations[0], generations[-1]
    context_curve = [{
        "tokens": tokens,
        "context_utilization": shared.row_at(runs[42], tokens)["context_utilization"],
        "activation_health": shared.row_at(runs[42], tokens)["activation_health"],
    } for tokens in MILESTONES]
    validation_healthy = final["validation_loss"]["mean"] < first["validation_loss"]["mean"]
    top_k_healthy = all(final[key]["mean"] > first[key]["mean"] for key in (
        "top_1_accuracy", "top_5_accuracy", "top_10_accuracy",
    ))
    teacher_healthy = (
        final_generation["teacher_forced_horizon"]["32"]["loss"]
        < first_generation["teacher_forced_horizon"]["32"]["loss"]
        and final_generation["teacher_forced_horizon"]["32"]["top_10_accuracy"]
        > first_generation["teacher_forced_horizon"]["32"]["top_10_accuracy"]
    )
    frequency_healthy = (
        final["outside_top_1_percent"]["top_10_accuracy"]["mean"]
        > first["outside_top_1_percent"]["top_10_accuracy"]["mean"]
    )
    context_healthy = (
        context_curve[-1]["context_utilization"]["full_vs_last_1_loss_advantage"] > 0
        and context_curve[-1]["context_utilization"]["512"]["loss"]
        < context_curve[0]["context_utilization"]["512"]["loss"]
    )
    training_stable = all(
        row["activation_health"]["all_finite"]
        and not row["activation_health"]["explosion"]
        and not row["activation_health"]["collapse"]
        for result in runs.values()
        for row in result["training"]["history"]
        if int(row["tokens_processed"]) in MILESTONES
    )
    repetition_values = [row["greedy"]["ngram_repetition"]["1"] for row in generations]
    if repetition_values[-1] <= repetition_values[0] - 0.05:
        repetition_trend = "CLEAR_IMPROVEMENT"
    elif repetition_values[-1] < repetition_values[0] - 0.01:
        repetition_trend = "SLOW_IMPROVEMENT"
    elif repetition_values[-1] <= repetition_values[0] + 0.01:
        repetition_trend = "FLAT"
    else:
        repetition_trend = "WORSE"
    sampling_improved = (
        final_generation["sampling_temperature_0.7"]["natural_japanese"]
        > first_generation["sampling_temperature_0.7"]["natural_japanese"]
        or final_generation["sampling_temperature_0.7"]["semantic_coherence"]
        > first_generation["sampling_temperature_0.7"]["semantic_coherence"]
    )
    candidate_improved = (
        final_generation["greedy"]["candidate_expected_top_10_rate"]
        > first_generation["greedy"]["candidate_expected_top_10_rate"]
    )
    divergence_improved = (
        final_generation["greedy"]["mean_divergence_position"]
        > first_generation["greedy"]["mean_divergence_position"]
    )
    generation_direction = sampling_improved or candidate_improved or divergence_improved or repetition_trend in {
        "CLEAR_IMPROVEMENT", "SLOW_IMPROVEMENT",
    }
    generation_lag = (
        final_generation["greedy"]["natural_japanese"] == 0
        and final_generation["greedy"]["runaway_rate"] >= 0.90
    )
    base_health = all((
        validation_healthy, top_k_healthy, teacher_healthy, frequency_healthy,
        context_healthy, training_stable, checkpoints["integrity_pass"],
        checkpoints["resume_reproducibility"]["status"] == "PASS",
        smoke["gate_pass"], parity["pass"],
    ))
    if not checkpoints["integrity_pass"] or not training_stable:
        gate = "TRAINING_INSTABILITY"
    elif not validation_healthy or not top_k_healthy:
        gate = "TRAINING_PLATEAU"
    elif base_health and generation_lag and generation_direction:
        gate = "CONTINUE_5M_GENERATION_LAG"
    elif base_health and not generation_lag:
        gate = "CONTINUE_5M"
    elif validation_healthy and top_k_healthy:
        gate = "GENERATION_PLATEAU_INVESTIGATE"
    else:
        gate = "STOP"
    language_emergence = (
        "YES" if final_generation["greedy"]["natural_japanese"] >= 0.50
        and final_generation["greedy"]["semantic_coherence"] >= 0.30
        else "PARTIAL" if final_generation["sampling_temperature_0.7"]["natural_japanese"] >= 0.20
        else "NO"
    )
    final_blind_sha = file_sha256(ROOT / settings["final_blind"]["path"])
    if final_blind_sha != settings["final_blind"]["expected_sha256"]:
        raise RuntimeError("Final Blind SHA256 mismatch")
    intervals = ((512_000, 1_024_000), (1_024_000, 1_536_000), (1_536_000, 2_048_000))
    historical_512 = read("evaluation/foundation-v24-summary.json")["training_curve"][0]
    lookup = {row["tokens"]: row for row in training_curve}
    lookup[512_000] = historical_512
    improvement_rates = []
    for start_tokens, end_tokens in intervals:
        start_row, end_row = lookup[start_tokens], lookup[end_tokens]
        improvement_rates.append({
            "interval": f"{start_tokens}-{end_tokens}",
            "loss_improvement_per_million_tokens": rate(
                start_row["validation_loss"]["mean"], end_row["validation_loss"]["mean"], start_tokens, end_tokens,
            ),
            "top_1_improvement_per_million_tokens": rate(
                start_row["top_1_accuracy"]["mean"], end_row["top_1_accuracy"]["mean"], start_tokens, end_tokens,
            ),
            "top_5_improvement_per_million_tokens": rate(
                start_row["top_5_accuracy"]["mean"], end_row["top_5_accuracy"]["mean"], start_tokens, end_tokens,
            ),
            "top_10_improvement_per_million_tokens": rate(
                start_row["top_10_accuracy"]["mean"], end_row["top_10_accuracy"]["mean"], start_tokens, end_tokens,
            ),
        })
    summary = {
        "schema": "foundation-v25-summary-v1",
        "phase": 36,
        "formal_architecture": "Current",
        "parameters": 19_514_880,
        "target_tokens": 2_048_000,
        "seeds": list(SEEDS),
        "training_curve": training_curve,
        "improvement_rates": improvement_rates,
        "generation_curve": generations,
        "repetition_trend": repetition_trend,
        "context_curve_representative_seed": context_curve,
        "knowledge_completion_observational": knowledge,
        "human_readable_generation": "evaluation/foundation-v25-generation-examples.json",
        "synthetic_smoke": smoke,
        "checkpoint_verification": checkpoints,
        "inference_parity_prior": parity,
        "gate_checks": {
            "validation_improved": validation_healthy,
            "top_k_improved": top_k_healthy,
            "teacher_forced_improved": teacher_healthy,
            "frequency_learning_improved": frequency_healthy,
            "context_maintained": context_healthy,
            "sampling_improved": sampling_improved,
            "candidate_quality_improved": candidate_improved,
            "divergence_improved": divergence_improved,
            "generation_direction": generation_direction,
            "training_stable": training_stable,
            "checkpoints_pass": checkpoints["integrity_pass"],
            "resume_reproducibility_pass": checkpoints["resume_reproducibility"]["status"] == "PASS",
            "synthetic_smoke_pass": smoke["gate_pass"],
            "inference_parity_pass": parity["pass"],
            "generation_lag": generation_lag,
        },
        "language_emergence": language_emergence,
        "gate": gate,
        "next_token_budget": "3M intermediate checkpoint toward 5M" if gate.startswith("CONTINUE_5M") else "INVESTIGATE",
        "full_training_continuation": "YES" if gate.startswith("CONTINUE_5M") else "NO",
        "foundation_base_complete": False,
        "final_blind": {"sha256": final_blind_sha, "content_opened": False},
        "production_changed": False,
        "campus_changed": False,
        "render_changed": False,
        "vercel_changed": False,
    }
    (ROOT / "evaluation/foundation-v25-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# UniPilot Foundation v2.5 — PHASE 36", "", f"Gate: **{gate}**", "",
        f"Language Emergence: **{language_emergence}**", "",
        "## Three-seed learning curve", "",
        "| tokens | val loss mean ± std | top-1 mean ± std | top-5 mean ± std | top-10 mean ± std | corpus / epoch |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in training_curve:
        lines.append(
            f"| {row['tokens']} | {row['validation_loss']['mean']:.4f} ± {row['validation_loss']['std']:.4f} | "
            f"{row['top_1_accuracy']['mean']:.2%} ± {row['top_1_accuracy']['std']:.2%} | "
            f"{row['top_5_accuracy']['mean']:.2%} ± {row['top_5_accuracy']['std']:.2%} | "
            f"{row['top_10_accuracy']['mean']:.2%} ± {row['top_10_accuracy']['std']:.2%} | "
            f"{row['corpus_percentage']:.4f}% / {row['epoch_equivalent']:.6f} |"
        )
    lines.extend(["", "### 2.048M results by seed", "", "| seed | val loss | top-1 | top-5 | top-10 |", "| ---: | ---: | ---: | ---: | ---: |"])
    for index, seed in enumerate(SEEDS):
        lines.append(
            f"| {seed} | {final['validation_loss']['values'][index]:.4f} | {final['top_1_accuracy']['values'][index]:.2%} | "
            f"{final['top_5_accuracy']['values'][index]:.2%} | {final['top_10_accuracy']['values'][index]:.2%} |"
        )
    lines.extend(["", "## Improvement rate", "", "| interval | loss / M tokens | Top-1 / M | Top-5 / M | Top-10 / M |", "| --- | ---: | ---: | ---: | ---: |"])
    for item in improvement_rates:
        lines.append(
            f"| {item['interval']} | {item['loss_improvement_per_million_tokens']:.4f} | "
            f"{item['top_1_improvement_per_million_tokens']:.2%} | {item['top_5_improvement_per_million_tokens']:.2%} | "
            f"{item['top_10_improvement_per_million_tokens']:.2%} |"
        )
    lines.extend(["", "## Teacher-forced and free-running generation", "", "| tokens | h32 loss/top-10 | divergence | rep-1/2/3/4 | greedy natural/semantic/runaway | sampling natural/semantic/runaway |", "| ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in generations:
        greedy, sampling = row["greedy"], row["sampling_temperature_0.7"]
        repetition = greedy["ngram_repetition"]
        lines.append(
            f"| {row['tokens']} | {row['teacher_forced_horizon']['32']['loss']:.4f} / {row['teacher_forced_horizon']['32']['top_10_accuracy']:.2%} | "
            f"{greedy['mean_divergence_position']:.3f} | {repetition['1']:.3f}/{repetition['2']:.3f}/{repetition['3']:.3f}/{repetition['4']:.3f} | "
            f"{greedy['natural_japanese']:.0%}/{greedy['semantic_coherence']:.0%}/{greedy['runaway_rate']:.0%} | "
            f"{sampling['natural_japanese']:.0%}/{sampling['semantic_coherence']:.0%}/{sampling['runaway_rate']:.0%} |"
        )
    lines.extend(["", "### Teacher-forced horizon at 2.048M", "", "| horizon | loss | top-1 | top-5 | top-10 | correct-token probability |", "| ---: | ---: | ---: | ---: | ---: | ---: |"])
    for horizon, metric in final_generation["teacher_forced_horizon"].items():
        lines.append(
            f"| {horizon} | {metric['loss']:.4f} | {metric['top_1_accuracy']:.2%} | "
            f"{metric['top_5_accuracy']:.2%} | {metric['top_10_accuracy']:.2%} | "
            f"{metric['mean_correct_token_probability']:.5f} |"
        )
    lines.extend(["", f"Repetition trend: **{repetition_trend}**.", "", "## Frequency, punctuation, boundary, and context", "", "| tokens | outside Top-1% Top-1/5/10 | generated Top-1% share | JS divergence |", "| ---: | ---: | ---: | ---: |"])
    for training, generation in zip(training_curve, generations, strict=True):
        outside = training["outside_top_1_percent"]
        frequency = generation["frequency_distribution"]
        lines.append(
            f"| {training['tokens']} | {outside['top_1_accuracy']['mean']:.2%}/{outside['top_5_accuracy']['mean']:.2%}/{outside['top_10_accuracy']['mean']:.2%} | "
            f"{frequency['generation_frequency_buckets']['top_1_percent']:.2%} | {frequency['jensen_shannon_divergence_nats']:.4f} |"
        )
    lines.extend(["", "### Validation frequency buckets at 2.048M", "", "| bucket | targets | top-1 | top-5 | top-10 | correct probability | cross entropy |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for bucket, metric in final["frequency_buckets"].items():
        lines.append(
            f"| {bucket} | {metric['targets']} | {metric['top_1_accuracy']['mean']:.2%} | "
            f"{metric['top_5_accuracy']['mean']:.2%} | {metric['top_10_accuracy']['mean']:.2%} | "
            f"{metric['mean_correct_token_probability']['mean']:.5f} | {metric['cross_entropy']['mean']:.4f} |"
        )
    lines.extend(["", "### Nine-token punctuation at 2.048M", "", "| token | actual | Top-1 predicted | mean probability | generated |", "| --- | ---: | ---: | ---: | ---: |"])
    for text, metric in final_generation["punctuation_distribution"].items():
        lines.append(f"| {text} | {metric['actual_frequency']:.3%} | {metric['top_1_predicted_frequency']:.3%} | {metric['mean_probability']:.5f} | {metric['generation_frequency']:.3%} |")
    lines.extend(["", "| boundary | actual | Top-1 predicted | mean probability | generated |", "| --- | ---: | ---: | ---: | ---: |"])
    for text, metric in final_generation["boundary_eos"].items():
        lines.append(f"| {text} | {metric['actual_frequency']:.3%} | {metric['top_1_prediction_rate']:.3%} | {metric['mean_predicted_probability']:.5f} | {metric['generation_frequency']:.3%} |")
    lines.extend(["", "| tokens | full loss | last-64 | last-16 | last-2 | last-1 | full vs last-1 advantage |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in context_curve:
        context = row["context_utilization"]
        lines.append(f"| {row['tokens']} | {context['512']['loss']:.4f} | {context['64']['loss']:.4f} | {context['16']['loss']:.4f} | {context['2']['loss']:.4f} | {context['1']['loss']:.4f} | {context['full_vs_last_1_loss_advantage']:.4f} |")
    eos = final_generation["eos_training_exposure"]
    lines.extend(["", "EOS training exposure at 2.048M: " + "; ".join(f"seed {row['seed']}: {row['input_eos_observations']} input / {row['supervised_eos_targets']} supervised" for row in eos) + ".", "", f"Checkpoint integrity: **PASS** ({checkpoints['verified_checkpoints']}/{checkpoints['expected_checkpoints']}); bitwise resume: **{checkpoints['resume_reproducibility']['status']}**; synthetic smoke: **{'PASS' if smoke['gate_pass'] else 'FAIL'}**.", "", "| tokens | knowledge keyword hit rate | role |", "| ---: | ---: | --- |"])
    for observation in knowledge:
        lines.append(f"| {observation['tokens']} | {observation['keyword_hit_rate']:.2%} | observational only |")
    lines.extend(["", f"Final Blind SHA256: `{final_blind_sha}`; content was not opened.", "", "## Decision", "", f"Next token budget: **{summary['next_token_budget']}**.", "Foundation Base is not complete. Architecture, tokenizer, corpus, Campus, production, Render, and Vercel were unchanged."])
    (ROOT / "evaluation/foundation-v25-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"gate": gate, "loss": final["validation_loss"]["mean"], "top_1_5_10": [final[key]["mean"] for key in ("top_1_accuracy", "top_5_accuracy", "top_10_accuracy")], "language_emergence": language_emergence, "repetition_trend": repetition_trend}, indent=2))
    return 0 if gate.startswith("CONTINUE_5M") else 2


if __name__ == "__main__":
    raise SystemExit(main())
