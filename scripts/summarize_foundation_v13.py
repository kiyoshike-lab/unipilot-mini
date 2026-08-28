from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FINAL_BLIND_SHA256 = "fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b"
PRIMARY_SAMPLING = "sampling_t07_topk40_topp09_no_penalty"
GREEDY = "greedy_no_penalty"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generation_metrics(generation: dict, step: int, mode: str) -> dict:
    row = next(item for item in generation["results"] if item["step"] == step)
    return row["modes"][mode]["metrics"]


def consecutive_increases(values: list[float]) -> int:
    maximum = current = 0
    for left, right in zip(values, values[1:]):
        current = current + 1 if right > left else 0
        maximum = max(maximum, current)
    return maximum


def generation_score(metrics: dict) -> float:
    levels = metrics["level_pass_rates"]
    return (
        levels["level_1_japanese_local_syntax"]
        + levels["level_2_semantic_sentence"]
        + levels["level_3_paragraph_coherence"]
        + metrics["completion_rate"]
        + metrics["prompt_alignment_rate"]
        - 0.25 * metrics["runaway_rate"]
        - 0.25 * metrics["mean_repetition_rate"]
    )


def table_training(rows: list[dict]) -> list[str]:
    selected = {0, 50, 100, 150, 200, 250}
    lines = [
        "| Step | Train loss | Validation | PPL | LR | Tokens | Corpus | tok/s | RAM MB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["step"] not in selected:
            continue
        train = "—" if row["train_loss"] is None else f"{row['train_loss']:.4f}"
        speed = "—" if row["tokens_per_second"] is None else f"{row['tokens_per_second']:.1f}"
        lines.append(
            f"| {row['step']} | {train} | {row['validation_loss']:.4f} | "
            f"{row['perplexity']:.1f} | {row['learning_rate']:.2e} | "
            f"{row['tokens_processed']:,} | {row['corpus_percentage']:.4f}% | "
            f"{speed} | {row['peak_ram_mb']:.1f} |"
        )
    return lines


def table_generation(generation: dict) -> list[str]:
    lines = [
        "| Step | Mode | Valid | Natural | Semantic | Completion | EOS | Runaway | Repetition | L0/L1/L2/L3/L4/L5 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in generation["results"]:
        for mode in (GREEDY, PRIMARY_SAMPLING):
            metrics = row["modes"][mode]["metrics"]
            levels = metrics["level_pass_rates"]
            level_values = [levels[key] for key in (
                "level_0_valid_characters", "level_1_japanese_local_syntax",
                "level_2_semantic_sentence", "level_3_paragraph_coherence",
                "level_4_prompt_aligned_completion", "level_5_instruction_following",
            )]
            lines.append(
                f"| {row['step']} | {mode} | {metrics['character_validity_rate']:.0%} | "
                f"{metrics['natural_japanese_rate']:.0%} | "
                f"{metrics['semantic_coherence_rate']:.0%} | "
                f"{metrics['completion_rate']:.0%} | {metrics['eos_rate']:.0%} | "
                f"{metrics['runaway_rate']:.0%} | {metrics['mean_repetition_rate']:.1%} | "
                f"{'/'.join(f'{value:.0%}' for value in level_values)} |"
            )
    return lines


def markdown(summary: dict, training: dict, generation: dict) -> str:
    lines = [
        "# UniPilot Foundation v1.3 — Full Clean 250-Step Report",
        "",
        "Clean Foundation v1.1 corpusとvocab 4096 tokenizerだけを使い、20M Baseをscratchから250step学習した。",
        "",
        "## Training curve",
        "",
        *table_training(training["history"]),
        "",
        "## Generation (50 fixed completion prompts)",
        "",
        *table_generation(generation),
        "",
        "Base評価はrepetition penaltyなし。samplingはtemperature 0.7 / top-k 40 / top-p 0.9。",
        "",
        "## Step 250 sampling observation",
        "",
        "| Mode | Valid | Natural | Semantic | Completion | EOS | Runaway | Repetition |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["generation"]["step_250_sampling_observation"].items():
        metrics = row["metrics"]
        lines.append(
            f"| {name} | {metrics['character_validity_rate']:.0%} | "
            f"{metrics['natural_japanese_rate']:.0%} | "
            f"{metrics['semantic_coherence_rate']:.0%} | "
            f"{metrics['completion_rate']:.0%} | {metrics['eos_rate']:.0%} | "
            f"{metrics['runaway_rate']:.0%} | {metrics['mean_repetition_rate']:.1%} |"
        )
    lines.extend([
        "",
        "## EOS / Base knowledge probes",
        "",
        "| Step | EOS probability after document | Knowledge keyword Greedy/Sampling | Natural Greedy/Sampling |",
        "|---:|---:|---:|---:|",
    ])
    for row in summary["generation"]["probe_table"]:
        lines.append(
            f"| {row['step']} | {row['eos_document_probability']:.6f} | "
            f"{row['knowledge_keyword_greedy']:.0%}/{row['knowledge_keyword_sampling']:.0%} | "
            f"{row['knowledge_natural_greedy']:.0%}/{row['knowledge_natural_sampling']:.0%} |"
        )
    lines.extend([
        "",
        "## Findings",
        "",
        f"- Best validation: step {summary['training']['best_validation_step']} / "
        f"{summary['training']['best_validation_loss']:.6f}",
        f"- Step 250 processed: {summary['training']['tokens_processed']:,} tokens / "
        f"{summary['training']['corpus_percentage']:.4f}% / "
        f"{summary['training']['epoch_equivalent']:.6f} epoch",
        f"- 日本語の立ち上がり: {summary['generation_findings']['japanese_started']}",
        f"- 意味文の増加: {summary['generation_findings']['semantic_increased']}",
        f"- Step 100から改善: {summary['generation_findings']['improved_from_step_100']}",
        f"- Validation gap at 250: {summary['training']['final_validation_train_gap']:.6f}",
        "",
        "## Gate",
        "",
        f"- 250step Gate: **{summary['gate']['status']}**",
        f"- 500stepへ進むべきか: **{summary['decisions']['continue_to_500']}**",
        f"- Corpus拡張が今必要か: **{summary['decisions']['corpus_extension_needed']}**",
        f"- Model変更が必要か: **{summary['decisions']['model_change_needed']}**",
        f"- 理由: {summary['gate']['reason']}",
        "",
        "## Integrity / protection",
        "",
        f"- Checkpoints: {summary['verification']['checkpoint_integrity']}",
        f"- Resume: {summary['verification']['resume_reproducibility']}",
        f"- Tokenizer roundtrip: {summary['verification']['tokenizer_roundtrip']}",
        f"- Final Blind SHA256: `{summary['protected']['final_blind_sha256']}`（内容未使用）",
        "- 500/1000step、46M、Campus pretraining、instruction tuning、DPOは未実行。",
        "- Production v0.4、Campus v2.3、Render、Vercel、Releaseは変更していない。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default="evaluation/foundation-v13-training-curve.json")
    parser.add_argument("--generation", default="evaluation/foundation-v13-generation.json")
    parser.add_argument("--checkpoints", default="evaluation/foundation-v13-checkpoint-integrity.json")
    parser.add_argument("--resume", default="evaluation/foundation-v13-resume-reproducibility.json")
    parser.add_argument("--output", default="evaluation/foundation-v13-summary.json")
    parser.add_argument("--report", default="evaluation/foundation-v13-report.md")
    args = parser.parse_args()
    training = load(args.training)
    generation = load(args.generation)
    checkpoints = load(args.checkpoints)
    resume = load(args.resume)
    phase23 = load("evaluation/foundation-v12-training-dynamics.json")
    tokenizer = load("evaluation/foundation-v11-tokenizer-benchmark.json")
    final_blind_path = ROOT / "data/foundation_v09/evaluation/final-blind-1000.json"
    final_blind_digest = sha256(final_blind_path)

    history = training["history"]
    validation = [row["validation_loss"] for row in history]
    best = min(history, key=lambda row: row["validation_loss"])
    final = next(row for row in history if row["step"] == 250)
    initial = next(row for row in history if row["step"] == 0)
    step100 = generation_metrics(generation, 100, PRIMARY_SAMPLING)
    step250 = generation_metrics(generation, 250, PRIMARY_SAMPLING)
    greedy250 = generation_metrics(generation, 250, GREEDY)
    level1_100 = step100["level_pass_rates"]["level_1_japanese_local_syntax"]
    level1_250 = step250["level_pass_rates"]["level_1_japanese_local_syntax"]
    level2_100 = step100["level_pass_rates"]["level_2_semantic_sentence"]
    level2_250 = step250["level_pass_rates"]["level_2_semantic_sentence"]
    loss_healthy = (
        not training["nan_or_inf"]
        and not training["diverged"]
        and final["validation_loss"] < initial["validation_loss"]
        and final["train_loss"] < 8.0
        and consecutive_increases(validation) <= 2
    )
    clear_level_growth = (
        level1_250 >= level1_100 + 0.10
        or (level1_250 >= 0.20 and level2_250 >= 0.05)
    )
    improved_from_100 = generation_score(step250) >= generation_score(step100) + 0.05
    corruption_free = (
        greedy250["character_validity_rate"] == 1.0
        and step250["character_validity_rate"] == 1.0
    )
    integrity = (
        checkpoints["all_checkpoints"] == "PASS"
        and resume["resume_integrity"] == "PASS"
        and final_blind_digest == EXPECTED_FINAL_BLIND_SHA256
    )
    if not integrity or not loss_healthy:
        gate = "STOP"
        reason = "loss/checkpoint/resume protectionのいずれかが正常条件を満たさない。"
    elif clear_level_growth and improved_from_100 and corruption_free:
        gate = "CONTINUE"
        reason = "lossが健全に下降し、Level 1以上とstep 100比の生成指標が明確に改善した。"
    else:
        gate = "INVESTIGATE"
        reason = (
            "lossは下降したが、Level 1〜2の増加、step 100比改善、または文字健全性の"
            "いずれかが250step継続条件に届かなかった。"
        )

    selected_steps = {0, 50, 100, 150, 200, 250}
    training_table = [row for row in history if row["step"] in selected_steps]
    generation_table = []
    for step in sorted(selected_steps):
        for mode in (GREEDY, PRIMARY_SAMPLING):
            generation_table.append({
                "step": step,
                "mode": mode,
                **generation_metrics(generation, step, mode),
            })
    roundtrip_pass = all(
        row["exact_roundtrip_rate"] == 1
        for row in tokenizer["results"]
        if row["actual_vocab"] == 4096
    )
    step_250_result = next(row for row in generation["results"] if row["step"] == 250)
    sampling_observation = {
        name: {"settings": row["settings"], "metrics": row["metrics"]}
        for name, row in step_250_result["sampling_observation"].items()
    }
    probe_table = []
    for row in generation["results"]:
        greedy_knowledge = row["knowledge_probe"][GREEDY]
        sampling_knowledge = row["knowledge_probe"][PRIMARY_SAMPLING]
        probe_table.append({
            "step": row["step"],
            "eos_document_probability": row["eos_document_probe"][
                "mean_eos_probability_after_complete_document"
            ],
            "eos_document_top1_rate": row["eos_document_probe"]["eos_top1_rate"],
            "knowledge_keyword_greedy": greedy_knowledge["keyword_hit_rate"],
            "knowledge_keyword_sampling": sampling_knowledge["keyword_hit_rate"],
            "knowledge_natural_greedy": greedy_knowledge["natural_japanese_rate"],
            "knowledge_natural_sampling": sampling_knowledge["natural_japanese_rate"],
        })
    summary = {
        "schema_version": "foundation-v13-summary-v1",
        "model": {
            "name": training["model_config"]["model_name"],
            "parameters": training["parameters"],
            "vocab": training["model_config"]["vocab_size"],
            "context": training["model_config"]["context_length"],
            "weight_tying": True,
            "scratch_start": training["scratch_start"],
            "resumed_from": training["resumed_from"],
        },
        "optimizer": training["optimizer"],
        "training": {
            "selected_step_table": training_table,
            "best_validation_step": best["step"],
            "best_validation_loss": best["validation_loss"],
            "final_validation_train_gap": final["validation_loss"] - final["train_loss"],
            "validation_improvement": initial["validation_loss"] - final["validation_loss"],
            "loss_healthy": loss_healthy,
            "maximum_consecutive_validation_increases": consecutive_increases(validation),
            "tokens_processed": final["tokens_processed"],
            "corpus_fraction": final["corpus_fraction"],
            "corpus_percentage": final["corpus_percentage"],
            "epoch_equivalent": final["epoch_equivalent"],
        },
        "generation": {
            "selected_step_table": generation_table,
            "step_250_sampling_observation": sampling_observation,
            "probe_table": probe_table,
            "base_repetition_penalty_used": False,
        },
        "generation_findings": {
            "japanese_started": "YES" if level1_250 > 0 and level1_250 > level1_100 else "NO",
            "semantic_increased": "YES" if level2_250 > level2_100 else "NO",
            "improved_from_step_100": "YES" if improved_from_100 else "NO",
            "sampling_level_1_step_100": level1_100,
            "sampling_level_1_step_250": level1_250,
            "sampling_level_2_step_100": level2_100,
            "sampling_level_2_step_250": level2_250,
            "generation_score_step_100": generation_score(step100),
            "generation_score_step_250": generation_score(step250),
            "corruption_free": corruption_free,
        },
        "gate": {
            "status": gate,
            "checks": {
                "loss_healthy": loss_healthy,
                "clear_level_1_plus_growth": clear_level_growth,
                "generation_improved_from_step_100": improved_from_100,
                "corruption_free": corruption_free,
                "integrity": integrity,
            },
            "reason": reason,
        },
        "decisions": {
            "continue_to_500": "YES" if gate == "CONTINUE" else "NO",
            "foundation_500_executed": False,
            "foundation_1000_executed": False,
            "corpus_extension_needed": "NO",
            "model_change_needed": "NO" if loss_healthy and integrity else "YES",
            "standard_46m_allowed": False,
        },
        "verification": {
            "checkpoint_integrity": checkpoints["all_checkpoints"],
            "resume_reproducibility": resume["resume_integrity"],
            "tokenizer_roundtrip": "PASS" if roundtrip_pass else "FAIL",
            "training_core_phase23": phase23["gates"]["training_core"],
        },
        "protected": {
            "final_blind_sha256": final_blind_digest,
            "final_blind_expected_sha256": EXPECTED_FINAL_BLIND_SHA256,
            "final_blind_content_opened": False,
            "production_v04_changed": False,
            "campus_v23_changed": False,
            "render_changed": False,
            "vercel_changed": False,
            "release_changed": False,
        },
        "external_ai_api": "OFF",
        "push_or_deploy_performed": False,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / args.report).write_text(markdown(summary, training, generation), encoding="utf-8")
    print(json.dumps({
        "gate": gate,
        "best_validation": {"step": best["step"], "loss": best["validation_loss"]},
        "tokens_processed": final["tokens_processed"],
        "corpus_percentage": final["corpus_percentage"],
        "generation_findings": summary["generation_findings"],
        "decisions": summary["decisions"],
    }, ensure_ascii=False, indent=2))
    return 0 if gate != "STOP" else 1


if __name__ == "__main__":
    raise SystemExit(main())
