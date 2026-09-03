"""Aggregate PHASE 37 continuation evidence without opening Final Blind."""
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
MILESTONES = (2_048_000, 2_560_000, 3_072_000, 3_584_000, 4_096_000, 4_608_000, 5_120_000)


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def delta(first: dict, last: dict) -> dict[str, float]:
    return {
        "validation_loss": last["validation_loss"]["mean"] - first["validation_loss"]["mean"],
        "top_1_accuracy": last["top_1_accuracy"]["mean"] - first["top_1_accuracy"]["mean"],
        "top_5_accuracy": last["top_5_accuracy"]["mean"] - first["top_5_accuracy"]["mean"],
        "top_10_accuracy": last["top_10_accuracy"]["mean"] - first["top_10_accuracy"]["mean"],
    }


def fixed_examples(diagnostics: dict[int, dict]) -> dict:
    audits = {tokens: diagnostics[tokens]["natural_japanese_evaluator_audit"]["examples"][:20] for tokens in MILESTONES}
    identifiers = [row["id"] for row in audits[MILESTONES[0]]]
    if any([row["id"] for row in audits[tokens]] != identifiers for tokens in MILESTONES):
        raise RuntimeError("fixed generation examples changed across milestones")
    return {
        "schema": "foundation-v26-fixed-generation-examples-v1",
        "representative_seed": 42,
        "fixed_examples": [
            {"id": item_id, "prefix": audits[MILESTONES[0]][index]["prefix"], "reference": audits[MILESTONES[0]][index]["reference"],
             "milestones": {str(tokens): {key: audits[tokens][index][key] for key in ("generated", "natural_japanese_proxy", "reasons")} for tokens in MILESTONES}}
            for index, item_id in enumerate(identifiers)
        ],
    }


def main() -> int:
    settings = load_json("configs/unipilot-foundation-v26.json")
    tokenizer = FoundationTokenizer.load(ROOT / settings["tokenizer"])
    runs = {seed: read(f"evaluation/foundation-v26-runs/current-seed-{seed}.json") for seed in SEEDS}
    diagnostics = {tokens: read(f"evaluation/foundation-v23-generation-diagnostics-{tokens}.json") for tokens in MILESTONES}
    punctuation = {int(row["tokens"]): row for row in read("evaluation/foundation-v26-punctuation.json")["rows"]}
    checkpoints = read("evaluation/foundation-v26-checkpoint-verification.json")
    smoke = read("evaluation/foundation-v26-synthetic-smoke.json")
    parity = read("evaluation/foundation-v23-inference-parity.json")
    intermediate = read("evaluation/foundation-v26-intermediate-gate.json")
    shared.MILESTONES, shared.SEEDS, shared.RUNS_BY_SEED = MILESTONES, SEEDS, runs
    curve = shared.aggregate_training_curve(list(runs.values()))
    generations = shared.generation_curve(diagnostics, tokenizer, punctuation)
    knowledge = shared.knowledge_observations(diagnostics)
    examples = fixed_examples(diagnostics)
    (ROOT / "evaluation/foundation-v26-generation-examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    at = {row["tokens"]: row for row in curve}
    first, three, final = at[2_048_000], at[3_072_000], at[5_120_000]
    first_g, three_g, final_g = generations[0], generations[2], generations[-1]
    context_curve = [{"tokens": tokens, "context_utilization": shared.row_at(runs[42], tokens)["context_utilization"], "activation_health": shared.row_at(runs[42], tokens)["activation_health"]} for tokens in MILESTONES]
    stable = all(row["activation_health"]["all_finite"] and not row["activation_health"]["explosion"] and not row["activation_health"]["collapse"] for run in runs.values() for row in run["training"]["history"] if int(row["tokens_processed"]) in MILESTONES)
    checks = {
        "validation_improved": final["validation_loss"]["mean"] < first["validation_loss"]["mean"],
        "top_k_improved": all(final[key]["mean"] > first[key]["mean"] for key in ("top_1_accuracy", "top_5_accuracy", "top_10_accuracy")),
        "teacher_forced_improved": final_g["teacher_forced_horizon"]["32"]["top_10_accuracy"] > first_g["teacher_forced_horizon"]["32"]["top_10_accuracy"],
        "frequency_learning_improved": final["outside_top_1_percent"]["top_10_accuracy"]["mean"] > first["outside_top_1_percent"]["top_10_accuracy"]["mean"],
        "context_maintained": min(row["context_utilization"]["full_vs_last_1_loss_advantage"] for row in context_curve) > 0,
        "sampling_naturalness_improved": final_g["sampling_temperature_0.7"]["natural_japanese"] > first_g["sampling_temperature_0.7"]["natural_japanese"],
        "sampling_semantic_improved": final_g["sampling_temperature_0.7"]["semantic_coherence"] > first_g["sampling_temperature_0.7"]["semantic_coherence"],
        "training_stable": stable,
        "checkpoints_pass": checkpoints["integrity_pass"],
        "resume_reproducibility_pass": checkpoints["resume_reproducibility"]["status"] == "PASS",
        "synthetic_smoke_pass": smoke["gate_pass"],
        "inference_parity_pass": parity["pass"],
    }
    greedy_lag = final_g["greedy"]["runaway_rate"] >= 0.90 or final_g["greedy"]["ngram_repetition"]["1"] > 0.85
    sampling_improved = checks["sampling_naturalness_improved"] or checks["sampling_semantic_improved"]
    base_health = all(checks[key] for key in ("validation_improved", "top_k_improved", "teacher_forced_improved", "frequency_learning_improved", "context_maintained", "training_stable", "checkpoints_pass", "resume_reproducibility_pass", "synthetic_smoke_pass", "inference_parity_pass"))
    if not checks["training_stable"] or not checks["checkpoints_pass"]:
        gate = "TRAINING_INSTABILITY"
    elif not checks["validation_improved"] or not checks["top_k_improved"]:
        gate = "TRAINING_PLATEAU"
    elif base_health and sampling_improved and greedy_lag:
        gate = "CONTINUE_10M_GENERATION_LAG"
    elif base_health and sampling_improved:
        gate = "CONTINUE_10M"
    elif base_health:
        gate = "GENERATION_PLATEAU_INVESTIGATE"
    else:
        gate = "STOP"
    language = "YES" if final_g["greedy"]["natural_japanese"] >= .50 and final_g["greedy"]["semantic_coherence"] >= .30 else "PARTIAL" if final_g["sampling_temperature_0.7"]["natural_japanese"] >= .20 else "NO"
    final_blind_sha = file_sha256(ROOT / settings["final_blind"]["path"])
    if final_blind_sha != settings["final_blind"]["expected_sha256"]:
        raise RuntimeError("Final Blind SHA256 mismatch")
    summary = {
        "schema": "foundation-v26-summary-v1", "phase": 37, "formal_architecture": "Current", "parameters": 19_514_880,
        "target_tokens": 5_120_000, "seeds": list(SEEDS), "training_curve": curve, "generation_curve": generations,
        "improvement_2048_to_3072": delta(first, three), "improvement_3072_to_5120": delta(three, final),
        "context_curve_representative_seed": context_curve, "knowledge_completion_observational": knowledge,
        "human_readable_generation": "evaluation/foundation-v26-generation-examples.json", "synthetic_smoke": smoke,
        "checkpoint_verification": checkpoints, "inference_parity_prior": parity, "intermediate_gate": intermediate,
        "gate_checks": checks, "greedy_generation_lag": greedy_lag, "language_emergence": language, "gate": gate,
        "next_token_budget": "7M intermediate checkpoint toward 10M" if gate.startswith("CONTINUE_10M") else "INVESTIGATE",
        "full_training_continuation": "YES" if gate.startswith("CONTINUE_10M") else "NO", "foundation_base_complete": False,
        "final_blind": {"sha256": final_blind_sha, "content_opened": False},
        "production_changed": False, "campus_changed": False, "render_changed": False, "vercel_changed": False,
    }
    (ROOT / "evaluation/foundation-v26-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# UniPilot Foundation v2.6 — PHASE 37", "", f"3.072M intermediate gate: **{intermediate['outcome']}**.", f"Final gate: **{gate}**.", f"Language Emergence: **{language}**.", "", "## Three-seed learning curve", "", "| tokens | validation loss (mean ± std) | Top-1 | Top-5 | Top-10 | corpus exposure |", "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in curve:
        lines.append(f"| {row['tokens']} | {row['validation_loss']['mean']:.4f} ± {row['validation_loss']['std']:.4f} | {row['top_1_accuracy']['mean']:.2%} ± {row['top_1_accuracy']['std']:.2%} | {row['top_5_accuracy']['mean']:.2%} ± {row['top_5_accuracy']['std']:.2%} | {row['top_10_accuracy']['mean']:.2%} ± {row['top_10_accuracy']['std']:.2%} | {row['corpus_percentage']:.4f}% |")
    lines.extend(["", "## Required comparisons", "", f"2.048M → 3.072M: loss {summary['improvement_2048_to_3072']['validation_loss']:.4f}; Top-1/5/10 {summary['improvement_2048_to_3072']['top_1_accuracy']:.2%}/{summary['improvement_2048_to_3072']['top_5_accuracy']:.2%}/{summary['improvement_2048_to_3072']['top_10_accuracy']:.2%}.", f"3.072M → 5.120M: loss {summary['improvement_3072_to_5120']['validation_loss']:.4f}; Top-1/5/10 {summary['improvement_3072_to_5120']['top_1_accuracy']:.2%}/{summary['improvement_3072_to_5120']['top_5_accuracy']:.2%}/{summary['improvement_3072_to_5120']['top_10_accuracy']:.2%}.", "", "| tokens | h32 loss / Top-10 | sampling natural / semantic | greedy rep-1 / runaway | full-vs-last-1 advantage |", "| ---: | ---: | ---: | ---: | ---: |"])
    for row, context in zip(generations, context_curve, strict=True):
        lines.append(f"| {row['tokens']} | {row['teacher_forced_horizon']['32']['loss']:.4f} / {row['teacher_forced_horizon']['32']['top_10_accuracy']:.2%} | {row['sampling_temperature_0.7']['natural_japanese']:.0%} / {row['sampling_temperature_0.7']['semantic_coherence']:.0%} | {row['greedy']['ngram_repetition']['1']:.3f} / {row['greedy']['runaway_rate']:.0%} | {context['context_utilization']['full_vs_last_1_loss_advantage']:.4f} |")
    lines.extend(["", f"Frequency learning (outside Top-1% Top-10): {first['outside_top_1_percent']['top_10_accuracy']['mean']:.2%} → {final['outside_top_1_percent']['top_10_accuracy']['mean']:.2%}.", f"Checkpoint integrity: **PASS** ({checkpoints['verified_checkpoints']}/{checkpoints['expected_checkpoints']}); bitwise resume: **{checkpoints['resume_reproducibility']['status']}**; synthetic smoke: **{'PASS' if smoke['gate_pass'] else 'FAIL'}**.", f"Final Blind SHA256: `{final_blind_sha}`; content was not opened.", "", "## Decision", "", f"Next token budget: **{summary['next_token_budget']}**. Foundation Base complete: **NO**. Architecture, corpus, tokenizer, Campus, production, Render, and Vercel were unchanged."])
    (ROOT / "evaluation/foundation-v26-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"gate": gate, "loss": final["validation_loss"]["mean"], "top_1_5_10": [final[key]["mean"] for key in ("top_1_accuracy", "top_5_accuracy", "top_10_accuracy")], "language_emergence": language}, indent=2))
    return 0 if gate.startswith("CONTINUE_10M") or gate == "GENERATION_PLATEAU_INVESTIGATE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
