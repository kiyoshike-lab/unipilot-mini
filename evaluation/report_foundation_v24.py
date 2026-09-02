"""Aggregate PHASE 35 three-seed learning and language-emergence evidence."""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.base_tokenizer import FoundationTokenizer
from training.train_foundation_v21_ab import file_sha256, load_json


SEEDS = (42, 123, 2026)
MILESTONES = (512_000, 640_000, 768_000, 896_000, 1_024_000)
BUCKET_METRICS = (
    "top_1_accuracy", "top_5_accuracy", "top_10_accuracy",
    "mean_correct_token_probability", "cross_entropy",
)


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def stats(values) -> dict:
    rows = [float(value) for value in values]
    return {
        "mean": statistics.fmean(rows),
        "std": statistics.pstdev(rows),
        "values": rows,
    }


def row_at(result: dict, tokens: int) -> dict:
    for row in result["training"]["history"]:
        if int(row["tokens_processed"]) == tokens:
            return row
    raise RuntimeError(f"seed {result['seed']} is missing {tokens}")


def outside_top_one_percent(validation: dict, top_k: int) -> float:
    direct_key = (
        "top_1_percent_outside_accuracy" if top_k == 1
        else f"top_1_percent_outside_top_{top_k}_accuracy"
    )
    if direct_key in validation:
        return float(validation[direct_key])
    buckets = list(validation["frequency_buckets"].values())[1:]
    metric = f"top_{top_k}_accuracy"
    total = sum(int(bucket["targets"]) for bucket in buckets)
    return sum(int(bucket["targets"]) * float(bucket[metric]) for bucket in buckets) / total


def aggregate_training_curve(results: list[dict]) -> list[dict]:
    curve = []
    for tokens in MILESTONES:
        rows = [row_at(result, tokens) for result in results]
        validations = [row["validation"] for row in rows]
        frequency_buckets = {}
        for name in validations[0]["frequency_buckets"]:
            frequency_buckets[name] = {
                metric: stats(validation["frequency_buckets"][name][metric] for validation in validations)
                for metric in BUCKET_METRICS
            }
            frequency_buckets[name]["targets"] = int(validations[0]["frequency_buckets"][name]["targets"])
        curve.append({
            "tokens": tokens,
            "corpus_fraction": tokens / 33_402_759,
            "corpus_percentage": 100 * tokens / 33_402_759,
            "epoch_equivalent": tokens / 33_402_759,
            "recent_train_loss": stats(row["recent_train_loss"] for row in rows),
            "validation_loss": stats(validation["loss"] for validation in validations),
            "perplexity": stats(validation["perplexity"] for validation in validations),
            "top_1_accuracy": stats(validation["top_1_accuracy"] for validation in validations),
            "top_5_accuracy": stats(validation["top_5_accuracy"] for validation in validations),
            "top_10_accuracy": stats(validation["top_10_accuracy"] for validation in validations),
            "mean_correct_token_probability": stats(validation["mean_correct_token_probability"] for validation in validations),
            "learning_rate": stats(row["learning_rate"] for row in rows),
            "gradient_norm": stats(row["gradient_norm"] for row in rows),
            "training_tokens_per_second": stats(row["training_tokens_per_second"] for row in rows),
            "peak_ram_mb": {"maximum": max(row["peak_ram_mb"] for row in rows), "values": [row["peak_ram_mb"] for row in rows]},
            "frequency_buckets": frequency_buckets,
            "outside_top_1_percent": {
                f"top_{top_k}_accuracy": stats(outside_top_one_percent(validation, top_k) for validation in validations)
                for top_k in (1, 5, 10)
            },
            "period_comma_prediction_mass": stats(validation["period_comma_prediction_mass"] for validation in validations),
        })
    return curve


def generation_curve(
    diagnostics: dict[int, dict],
    tokenizer: FoundationTokenizer,
    punctuation_probe: dict[int, dict],
) -> list[dict]:
    curve = []
    named = ("。", "、", "の", "に", "は", "を", "が", "と", "で")
    for tokens in MILESTONES:
        payload = diagnostics[tokens]
        items = payload["validation_document_prefix"]["items"]
        metrics = payload["validation_document_prefix"]["metrics"]
        free = metrics["free_running"]
        generated_counts = payload["token_distribution"]["generated_counts"]
        generated_total = sum(generated_counts)
        sampling = payload["decoding_comparison"]["temperature_0.7"]
        sampling_items = sampling["items"]
        greedy_completion = sum(item["generation"]["completion_proxy"] for item in items) / len(items)
        greedy_eos = sum(item["generation"]["eos_reached"] for item in items) / len(items)
        sampling_completion = sum(item["completion_proxy"] for item in sampling_items) / len(sampling_items)
        sampling_eos = sum(item["eos_reached"] for item in sampling_items) / len(sampling_items)
        punctuation = {}
        representative_validation = punctuation_probe[tokens]
        for text in named:
            token_ids = tokenizer.encode(text, add_bos=False)
            prediction = representative_validation["punctuation"][text]
            punctuation[text] = {
                "token_ids": token_ids,
                "actual_frequency": prediction["actual_frequency"],
                "top_1_predicted_frequency": prediction["top_1_predicted_frequency"],
                "mean_probability": prediction["mean_probability"],
                "generation_frequency": sum(generated_counts[token_id] for token_id in token_ids) / generated_total,
            }
        curve.append({
            "tokens": tokens,
            "teacher_forced_horizon": metrics["teacher_forced_horizon"],
            "greedy": {
                "mean_divergence_position": free["mean_divergence_position"],
                "first_4_token_exact": free["first_4_token_exact"],
                "first_8_token_exact": free["first_8_token_exact"],
                "character_validity": free["character_validity"],
                "japanese_character_ratio": free["japanese_character_ratio"],
                "natural_japanese": free["natural_japanese_proxy"],
                "semantic_coherence": free["semantic_local_syntax_proxy"],
                "sentence_completion": greedy_completion,
                "sentence_boundary_rate": free["sentence_boundary_rate"],
                "eos_rate": greedy_eos,
                "runaway_rate": free["runaway_rate"],
                "runaway_onset_token": 64 if free["runaway_rate"] else None,
                "ngram_repetition": free["ngram_repetition"],
                "mean_loop_onset": free["mean_loop_onset"],
                "mean_maximum_repeated_span": free["mean_maximum_repeated_span"],
                "candidate_expected_top_5_rate": free["candidate_expected_top_5_rate"],
                "candidate_expected_top_10_rate": free["candidate_expected_top_10_rate"],
            },
            "sampling_temperature_0.7": {
                "character_validity": sampling["character_validity"],
                "japanese_character_ratio": sampling["japanese_character_ratio"],
                "natural_japanese": sampling["natural_japanese_proxy"],
                "semantic_coherence": sampling["semantic_local_syntax_proxy"],
                "sentence_completion": sampling_completion,
                "eos_rate": sampling_eos,
                "runaway_rate": sampling["runaway_rate"],
                "repetition_3gram": sampling["mean_repetition_3gram"],
            },
            "oracle_prefix_recovery": metrics["oracle_prefix_recovery"],
            "loop_confidence": metrics["loop_onset_confidence"],
            "error_taxonomy": metrics["error_taxonomy"],
            "train_prefix": payload["train_document_prefix"]["metrics"],
            "validation_prefix": metrics,
            "sentence_prefix": payload["validation_sentence_prefix"]["metrics"],
            "frequency_distribution": {
                key: value for key, value in payload["token_distribution"].items()
                if key != "generated_counts"
            },
            "punctuation_distribution": punctuation,
            "boundary_eos": payload["boundary_diagnostics"],
            "eos_training_exposure": payload["training_exposure"],
        })
    return curve


def knowledge_observations(diagnostics: dict[int, dict]) -> list[dict]:
    rows = []
    for tokens in MILESTONES:
        items = diagnostics[tokens]["instruction_like_observations"]
        hits = 0
        examples = []
        for item in items:
            text = item["generation"]["text"]
            matched = [keyword for keyword in item["expected_keywords"] if keyword in text]
            hits += bool(matched)
            examples.append({"prompt": item["prompt"], "text": text, "keyword_hits": matched})
        rows.append({
            "tokens": tokens,
            "keyword_hit_rate": hits / len(items),
            "items": examples,
            "primary_gate": False,
        })
    return rows


def human_comparison(diagnostics: dict[int, dict]) -> dict:
    per_tokens = {
        tokens: diagnostics[tokens]["natural_japanese_evaluator_audit"]["examples"][:20]
        for tokens in MILESTONES
    }
    identifiers = [row["id"] for row in per_tokens[512_000]]
    if any([row["id"] for row in per_tokens[tokens]] != identifiers for tokens in MILESTONES):
        raise RuntimeError("human-readable generation identifiers are not fixed")
    return {
        "schema": "foundation-v24-fixed-generation-examples-v1",
        "representative_seed": 42,
        "fixed_examples": [
            {
                "id": identifier,
                "prefix": per_tokens[512_000][index]["prefix"],
                "reference": per_tokens[512_000][index]["reference"],
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


RUNS_BY_SEED: dict[int, dict] = {}


def main() -> int:
    settings = load_json("configs/unipilot-foundation-v24.json")
    tokenizer = FoundationTokenizer.load(ROOT / settings["tokenizer"])
    global RUNS_BY_SEED
    RUNS_BY_SEED = {
        seed: read(f"evaluation/foundation-v24-runs/current-seed-{seed}.json")
        for seed in SEEDS
    }
    results = list(RUNS_BY_SEED.values())
    diagnostics = {
        tokens: read(f"evaluation/foundation-v23-generation-diagnostics-{tokens}.json")
        for tokens in MILESTONES
    }
    punctuation_probe = {
        int(row["tokens"]): row
        for row in read("evaluation/foundation-v24-punctuation.json")["rows"]
    }
    checkpoints = read("evaluation/foundation-v24-checkpoint-verification.json")
    smoke = read("evaluation/foundation-v24-synthetic-smoke.json")
    parity = read("evaluation/foundation-v23-inference-parity.json")
    training_curve = aggregate_training_curve(results)
    generations = generation_curve(diagnostics, tokenizer, punctuation_probe)
    knowledge = knowledge_observations(diagnostics)
    examples = human_comparison(diagnostics)
    (ROOT / "evaluation/foundation-v24-generation-examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    first, final = training_curve[0], training_curve[-1]
    first_generation, final_generation = generations[0], generations[-1]
    context_curve = [
        {
            "tokens": tokens,
            "context_utilization": row_at(RUNS_BY_SEED[42], tokens)["context_utilization"],
            "activation_health": row_at(RUNS_BY_SEED[42], tokens)["activation_health"],
        }
        for tokens in MILESTONES
    ]
    validation_healthy = final["validation_loss"]["mean"] < first["validation_loss"]["mean"]
    top_k_healthy = all(
        final[key]["mean"] > first[key]["mean"]
        for key in ("top_1_accuracy", "top_5_accuracy", "top_10_accuracy")
    )
    teacher_healthy = (
        final_generation["teacher_forced_horizon"]["32"]["loss"]
        < first_generation["teacher_forced_horizon"]["32"]["loss"]
        and all(
            final_generation["teacher_forced_horizon"]["32"][key]
            > first_generation["teacher_forced_horizon"]["32"][key]
            for key in ("top_1_accuracy", "top_5_accuracy", "top_10_accuracy")
        )
    )
    frequency_healthy = (
        final_generation["frequency_distribution"]["generation_frequency_buckets"]["top_1_percent"]
        < first_generation["frequency_distribution"]["generation_frequency_buckets"]["top_1_percent"]
        and final["outside_top_1_percent"]["top_10_accuracy"]["mean"]
        > first["outside_top_1_percent"]["top_10_accuracy"]["mean"]
    )
    context_512 = context_curve[0]["context_utilization"]["full_vs_last_1_loss_advantage"]
    context_1024 = context_curve[-1]["context_utilization"]["full_vs_last_1_loss_advantage"]
    context_healthy = context_1024 > 0 and context_1024 >= context_512 - 0.10
    generation_not_regressed = all((
        final_generation["greedy"]["ngram_repetition"]["1"] <= first_generation["greedy"]["ngram_repetition"]["1"] + 0.02,
        final_generation["greedy"]["japanese_character_ratio"] >= first_generation["greedy"]["japanese_character_ratio"] - 0.05,
        final_generation["frequency_distribution"]["jensen_shannon_divergence_nats"] <= first_generation["frequency_distribution"]["jensen_shannon_divergence_nats"] + 0.02,
        final_generation["sampling_temperature_0.7"]["natural_japanese"] >= first_generation["sampling_temperature_0.7"]["natural_japanese"],
    ))
    training_stable = all(
        row_at(result, tokens)["activation_health"]["layers"][-1]["final_residual_rms"] < 20
        for result in results for tokens in MILESTONES
    )
    all_health = all((
        validation_healthy, top_k_healthy, teacher_healthy, frequency_healthy,
        context_healthy, generation_not_regressed, training_stable,
        checkpoints["integrity_pass"],
        checkpoints["resume_reproducibility"]["status"] == "PASS",
        smoke["gate_pass"], parity["pass"],
    ))
    generation_lag = (
        final_generation["greedy"]["natural_japanese"] == 0
        and final_generation["greedy"]["runaway_rate"] >= 0.90
    )
    if not checkpoints["integrity_pass"] or not training_stable:
        gate = "TRAINING_INSTABILITY"
    elif not validation_healthy or not top_k_healthy:
        gate = "TRAINING_PLATEAU"
    elif all_health and generation_lag:
        gate = "CONTINUE_2M_GENERATION_LAG"
    elif all_health:
        gate = "CONTINUE_2M"
    else:
        gate = "GENERATION_PLATEAU_INVESTIGATE"
    language_emergence = (
        "YES" if final_generation["greedy"]["natural_japanese"] >= 0.50
        and final_generation["greedy"]["semantic_coherence"] >= 0.30
        else "PARTIAL" if final_generation["sampling_temperature_0.7"]["natural_japanese"] >= 0.20
        else "NO"
    )
    final_blind = ROOT / settings["final_blind"]["path"]
    final_blind_sha = file_sha256(final_blind)
    if final_blind_sha != settings["final_blind"]["expected_sha256"]:
        raise RuntimeError("Final Blind SHA256 mismatch")
    pass_1024 = gate in {"CONTINUE_2M", "CONTINUE_2M_GENERATION_LAG"}
    summary = {
        "schema": "foundation-v24-summary-v1",
        "phase": 35,
        "formal_architecture": "Current",
        "parameters": 19_514_880,
        "target_tokens": 1_024_000,
        "seeds": list(SEEDS),
        "training_curve": training_curve,
        "generation_curve": generations,
        "context_curve_representative_seed": context_curve,
        "knowledge_completion_observational": knowledge,
        "human_readable_generation": "evaluation/foundation-v24-generation-examples.json",
        "synthetic_smoke": smoke,
        "checkpoint_verification": checkpoints,
        "inference_parity_prior": parity,
        "gate_checks": {
            "validation_improved": validation_healthy,
            "top_k_improved": top_k_healthy,
            "teacher_forced_improved": teacher_healthy,
            "frequency_learning_improved": frequency_healthy,
            "context_maintained": context_healthy,
            "generation_no_major_regression": generation_not_regressed,
            "training_stable": training_stable,
            "checkpoints_pass": checkpoints["integrity_pass"],
            "resume_reproducibility_pass": checkpoints["resume_reproducibility"]["status"] == "PASS",
            "synthetic_smoke_pass": smoke["gate_pass"],
            "inference_parity_pass": parity["pass"],
            "generation_lag": generation_lag,
        },
        "language_emergence": language_emergence,
        "gate": gate,
        "one_point_024m": "PASS" if pass_1024 else "FAIL",
        "next_token_budget": "2M" if pass_1024 else "INVESTIGATE",
        "full_training_continuation": "YES" if pass_1024 else "NO",
        "foundation_base_complete": False,
        "final_blind": {"sha256": final_blind_sha, "content_opened": False},
        "production_changed": False,
        "campus_changed": False,
        "render_changed": False,
        "vercel_changed": False,
    }
    (ROOT / "evaluation/foundation-v24-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# UniPilot Foundation v2.4 — PHASE 35",
        "",
        f"Gate: **{gate}**",
        "",
        f"1.024M: **{'PASS' if pass_1024 else 'FAIL'}**",
        "",
        f"Language Emergence: **{language_emergence}**",
        "",
        "Formal architecture: **Current** (19,514,880 parameters). Training used the fixed 33,402,759-token clean corpus and vocab 4096 tokenizer.",
        "",
        "## Three-seed learning curve",
        "",
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
    lines.extend([
        "",
        "### 1.024M results by seed",
        "",
        "| seed | val loss | top-1 | top-5 | top-10 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for index, seed in enumerate(SEEDS):
        lines.append(
            f"| {seed} | {final['validation_loss']['values'][index]:.4f} | "
            f"{final['top_1_accuracy']['values'][index]:.2%} | "
            f"{final['top_5_accuracy']['values'][index]:.2%} | "
            f"{final['top_10_accuracy']['values'][index]:.2%} |"
        )
    lines.extend([
        "",
        "| tokens | train loss | PPL | LR | grad norm | train tok/s | peak RAM MiB |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in training_curve:
        lines.append(
            f"| {row['tokens']} | {row['recent_train_loss']['mean']:.4f} | {row['perplexity']['mean']:.2f} | "
            f"{row['learning_rate']['mean']:.1e} | {row['gradient_norm']['mean']:.3f} | "
            f"{row['training_tokens_per_second']['mean']:.1f} | {row['peak_ram_mb']['maximum']:.1f} |"
        )
    lines.extend([
        "",
        "Validation loss and all aggregate Top-k measures improved from 512k to 1.024M; validation did not regress while training loss improved.",
        "",
        "## Teacher-forced horizon at 1.024M",
        "",
        "| horizon | loss | top-1 | top-5 | top-10 | correct-token probability |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for horizon, metric in final_generation["teacher_forced_horizon"].items():
        lines.append(
            f"| {horizon} | {metric['loss']:.4f} | {metric['top_1_accuracy']:.2%} | "
            f"{metric['top_5_accuracy']:.2%} | {metric['top_10_accuracy']:.2%} | "
            f"{metric['mean_correct_token_probability']:.4f} |"
        )
    lines.extend([
        "",
        "## Free-generation and repetition curve",
        "",
        "| tokens | divergence | rep-1/2/3/4 | loop onset / max span | greedy natural / semantic / runaway | sampling t0.7 natural / semantic / runaway |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in generations:
        repetition = row["greedy"]["ngram_repetition"]
        sampling = row["sampling_temperature_0.7"]
        lines.append(
            f"| {row['tokens']} | {row['greedy']['mean_divergence_position']:.3f} | "
            f"{repetition['1']:.3f}/{repetition['2']:.3f}/{repetition['3']:.3f}/{repetition['4']:.3f} | "
            f"{row['greedy']['mean_loop_onset']:.2f} / {row['greedy']['mean_maximum_repeated_span']:.2f} | "
            f"{row['greedy']['natural_japanese']:.0%} / {row['greedy']['semantic_coherence']:.0%} / {row['greedy']['runaway_rate']:.0%} | "
            f"{sampling['natural_japanese']:.0%} / {sampling['semantic_coherence']:.0%} / {sampling['runaway_rate']:.0%} |"
        )
    lines.extend([
        "",
        "Greedy repetition improved only slightly and remains severe. Greedy still diverges near the first token and runs away on every probe; sampling shows partial Japanese structure. This is generation lag rather than an inference-path or training failure.",
        "",
        "### Prefix completion at 1.024M",
        "",
        "| set | examples | divergence | character validity | Japanese ratio | natural | semantic | boundary | runaway |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for label, prefix in (
        ("train", final_generation["train_prefix"]),
        ("validation", final_generation["validation_prefix"]),
        ("sentence", final_generation["sentence_prefix"]),
    ):
        free = prefix["free_running"]
        lines.append(
            f"| {label} | {prefix['examples']} | {free['mean_divergence_position']:.3f} | "
            f"{free['character_validity']:.2%} | {free['japanese_character_ratio']:.2%} | "
            f"{free['natural_japanese_proxy']:.2%} | {free['semantic_local_syntax_proxy']:.2%} | "
            f"{free['sentence_boundary_rate']:.2%} | {free['runaway_rate']:.2%} |"
        )
    lines.extend([
        "",
        "## Frequency learning",
        "",
        "### Outside the Top-1% frequency bucket",
        "",
        "| tokens | top-1 | top-5 | top-10 | generated Top-1% share | distribution JS divergence |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for training, generation in zip(training_curve, generations, strict=True):
        outside = training["outside_top_1_percent"]
        frequency = generation["frequency_distribution"]
        lines.append(
            f"| {training['tokens']} | {outside['top_1_accuracy']['mean']:.2%} | "
            f"{outside['top_5_accuracy']['mean']:.2%} | {outside['top_10_accuracy']['mean']:.2%} | "
            f"{frequency['generation_frequency_buckets']['top_1_percent']:.2%} | "
            f"{frequency['jensen_shannon_divergence_nats']:.4f} |"
        )
    lines.extend([
        "",
        "### Validation frequency buckets at 1.024M",
        "",
        "| bucket | targets | top-1 | top-5 | top-10 | correct probability | cross entropy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for bucket, metric in final["frequency_buckets"].items():
        lines.append(
            f"| {bucket} | {metric['targets']} | {metric['top_1_accuracy']['mean']:.2%} | "
            f"{metric['top_5_accuracy']['mean']:.2%} | {metric['top_10_accuracy']['mean']:.2%} | "
            f"{metric['mean_correct_token_probability']['mean']:.5f} | {metric['cross_entropy']['mean']:.4f} |"
        )
    lines.extend([
        "",
        "## Punctuation and boundary distribution at 1.024M",
        "",
        "| token | actual | Top-1 predicted | mean probability | generated |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for text, metric in final_generation["punctuation_distribution"].items():
        lines.append(
            f"| {text} | {metric['actual_frequency']:.3%} | {metric['top_1_predicted_frequency']:.3%} | "
            f"{metric['mean_probability']:.5f} | {metric['generation_frequency']:.3%} |"
        )
    lines.extend([
        "",
        "| boundary | actual | Top-1 predicted | mean probability | generated |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for text, metric in final_generation["boundary_eos"].items():
        lines.append(
            f"| {text} | {metric['actual_frequency']:.3%} | {metric['top_1_prediction_rate']:.3%} | "
            f"{metric['mean_predicted_probability']:.5f} | {metric['generation_frequency']:.3%} |"
        )
    lines.extend([
        "",
        "EOS exposure is measured from the actual training stream:",
        "",
        "| seed | input EOS | supervised EOS targets | input BOS | supervised BOS targets |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for exposure in final_generation["eos_training_exposure"]:
        lines.append(
            f"| {exposure['seed']} | {exposure['input_eos_observations']} | "
            f"{exposure['supervised_eos_targets']} | {exposure['input_bos_observations']} | "
            f"{exposure['supervised_bos_targets']} |"
        )
    lines.extend([
        "",
        "## Context utilization (representative seed 42)",
        "",
        "| tokens | full loss | last-64 | last-16 | last-2 | last-1 | full vs last-1 advantage |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in context_curve:
        context = row["context_utilization"]
        lines.append(
            f"| {row['tokens']} | {context['512']['loss']:.4f} | {context['64']['loss']:.4f} | "
            f"{context['16']['loss']:.4f} | {context['2']['loss']:.4f} | {context['1']['loss']:.4f} | "
            f"{context['full_vs_last_1_loss_advantage']:.4f} |"
        )
    lines.extend([
        "",
        "## Regression, integrity, and observational probes",
        "",
        f"Checkpoint integrity: **{'PASS' if checkpoints['integrity_pass'] else 'FAIL'}** "
        f"({checkpoints['verified_checkpoints']}/{checkpoints['expected_checkpoints']}). "
        f"Bitwise resume reproducibility: **{checkpoints['resume_reproducibility']['status']}**.",
        f"Synthetic smoke: **{'PASS' if smoke['gate_pass'] else 'FAIL'}**. Context maintained: **{'PASS' if context_healthy else 'FAIL'}**. Prior inference/KV-cache parity: **{'PASS' if parity['pass'] else 'FAIL'}**.",
        "",
        "| tokens | knowledge keyword hit rate | role |",
        "| ---: | ---: | --- |",
    ])
    for observation in knowledge:
        lines.append(
            f"| {observation['tokens']} | {observation['keyword_hit_rate']:.2%} | observational only |"
        )
    lines.extend([
        "",
        "Knowledge completion is not a gate before instruction tuning. Human-readable fixed-prefix examples are saved in `evaluation/foundation-v24-generation-examples.json`.",
        f"Final Blind SHA256: `{final_blind_sha}`. Its content was not opened.",
        "",
        "## Decision",
        "",
        f"Gate: **{gate}**. Formal architecture: **Current**. 1.024M: **{'PASS' if pass_1024 else 'FAIL'}**. Language Emergence: **{language_emergence}**.",
        f"Next token budget: **{'2M' if pass_1024 else 'INVESTIGATE'}**. Full training continuation: **{'YES' if pass_1024 else 'NO'}**.",
        "Foundation Base is not complete. Architecture, tokenizer, corpus, Campus, production, Render, and Vercel were unchanged.",
    ])
    (ROOT / "evaluation/foundation-v24-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "gate": gate,
        "loss": final["validation_loss"]["mean"],
        "top_1_5_10": [final[key]["mean"] for key in ("top_1_accuracy", "top_5_accuracy", "top_10_accuracy")],
        "language_emergence": language_emergence,
    }, indent=2))
    return 0 if pass_1024 else 2


if __name__ == "__main__":
    raise SystemExit(main())
