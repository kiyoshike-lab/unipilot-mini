"""Aggregate PHASE 33 curves and issue the mandatory 512k research gate."""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_foundation_v21_ab import file_sha256, load_json


SEEDS = (42, 123, 2026)
MILESTONES = (256_000, 320_000, 384_000, 448_000, 512_000)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def avg(values: list[float]) -> float:
    return statistics.fmean(values)


def average_tree(values: list[dict]) -> dict:
    """Average the numeric leaves of a same-shaped diagnostic mapping."""
    return {
        key: average_tree([value[key] for value in values])
        if isinstance(values[0][key], dict) else avg([value[key] for value in values])
        for key in values[0]
    }


def milestone_row(result: dict, tokens: int) -> dict:
    for row in result["training"]["history"]:
        if int(row["tokens_processed"]) == tokens:
            return row
    raise RuntimeError(f"missing {tokens} evaluation for seed {result['seed']}")


def aggregate_milestone(rows: list[dict], tokens: int) -> dict:
    validation = [row["validation"] for row in rows]
    buckets = {}
    for name in validation[0]["frequency_buckets"]:
        buckets[name] = {
            metric: avg([item["frequency_buckets"][name][metric] for item in validation])
            for metric in ("top_1_accuracy", "top_5_accuracy", "top_10_accuracy", "mean_correct_token_probability", "cross_entropy")
        }
    layer9 = [row["activation_health"]["layers"][-1] for row in rows]
    contexts = [row["context_utilization"] for row in rows]
    return {
        "tokens": tokens,
        "seeds": len(rows),
        "validation_loss": avg([item["loss"] for item in validation]),
        "perplexity": avg([item["perplexity"] for item in validation]),
        "top_1_accuracy": avg([item["top_1_accuracy"] for item in validation]),
        "top_5_accuracy": avg([item["top_5_accuracy"] for item in validation]),
        "top_10_accuracy": avg([item["top_10_accuracy"] for item in validation]),
        "mean_correct_token_probability": avg([item["mean_correct_token_probability"] for item in validation]),
        "training_tokens_per_second": avg([row["training_tokens_per_second"] for row in rows]),
        "peak_ram_mb": max(row["peak_ram_mb"] for row in rows),
        "gradient_norm": avg([row["gradient_norm"] for row in rows]),
        "frequency_buckets": buckets,
        "top_1_percent_outside_accuracy": avg([item["top_1_percent_outside_accuracy"] for item in validation]),
        "period_comma_prediction_mass": avg([item["period_comma_prediction_mass"] for item in validation]),
        "sentence_boundaries": {
            name: {
                "top_1_accuracy": avg([
                    item["sentence_boundaries"][name]["top_1_accuracy"]
                    for item in validation if item["sentence_boundaries"][name]["top_1_accuracy"] is not None
                ]) if any(item["sentence_boundaries"][name]["top_1_accuracy"] is not None for item in validation) else None,
                "predicted_frequency": avg([item["sentence_boundaries"][name]["predicted_frequency"] for item in validation]),
            }
            for name in validation[0]["sentence_boundaries"]
        },
        "activation_health": {
            "embedding_rms": avg([row["activation_health"]["embedding_rms"] for row in rows]),
            "layer_9_final_residual_rms": avg([item["final_residual_rms"] for item in layer9]),
            "layer_9_mlp_output_rms": avg([item["mlp_output_rms"] for item in layer9]),
        },
        "context_utilization": average_tree(contexts),
    }


def generation_summary(seed42: dict) -> list[dict]:
    result = []
    for tokens in MILESTONES:
        row = milestone_row(seed42, tokens)
        generation = row.get("generation")
        if not generation:
            continue
        result.append({
            "tokens": tokens,
            "modes": {
                name: {key: values["metrics"][key] for key in (
                    "character_validity", "natural_japanese_proxy", "semantic_coherence_proxy",
                    "completion_proxy", "eos_rate", "runaway_rate", "mean_repetition_rate",
                )}
                for name, values in generation.items()
            },
        })
    return result


def gate(curve: list[dict], generation: list[dict], verification: dict) -> tuple[str, dict]:
    first, final = curve[0], curve[-1]
    stable = all(math.isfinite(row["validation_loss"]) and math.isfinite(row["gradient_norm"]) for row in curve)
    learning = final["validation_loss"] < first["validation_loss"] and final["top_1_accuracy"] >= first["top_1_accuracy"]
    context_healthy = final["context_utilization"]["full_vs_last_1_loss_advantage"] >= first["context_utilization"]["full_vs_last_1_loss_advantage"] - 0.10
    punctuation_healthy = final["period_comma_prediction_mass"] <= first["period_comma_prediction_mass"] + 0.05
    greedy = next((row["modes"].get("greedy_no_penalty") for row in generation if row["tokens"] == 512_000), None)
    generation_emerged = bool(greedy and greedy["character_validity"] >= 0.80 and greedy["natural_japanese_proxy"] >= 0.50 and greedy["mean_repetition_rate"] < 0.35)
    evidence = {
        "training_stable": stable,
        "validation_learning": learning,
        "context_healthy": context_healthy,
        "punctuation_healthy": punctuation_healthy,
        "generation_emerged": generation_emerged,
        "checkpoint_integrity": bool(verification["integrity_pass"]),
    }
    if not stable or not verification["integrity_pass"]:
        return "TRAINING_INSTABILITY", evidence
    if not learning:
        return "PLATEAU_INVESTIGATE", evidence
    if not generation_emerged:
        return "INVESTIGATE_GENERATION", evidence
    if context_healthy and punctuation_healthy:
        return "CONTINUE_1M", evidence
    return "PLATEAU_INVESTIGATE", evidence


def main() -> int:
    settings = load_json("configs/unipilot-foundation-v22.json")
    results = []
    for seed in SEEDS:
        v21 = read(ROOT / f"evaluation/foundation-v21-runs/current-seed-{seed}.json")
        v22 = read(ROOT / f"evaluation/foundation-v22-runs/current-seed-{seed}.json")
        results.append(v22)
        historical = dict(milestone_row(v21, 256_000))
        historical.pop("checkpoint", None)
        if historical != milestone_row(v22, 256_000):
            raise RuntimeError(f"seed {seed} altered its 256k historical evaluation")
    curve = [aggregate_milestone([milestone_row(result, tokens) for result in results], tokens) for tokens in MILESTONES]
    knowledge = read(ROOT / "evaluation/foundation-v22-knowledge-probes.json")
    verification = read(ROOT / "evaluation/foundation-v22-checkpoint-verification.json")
    smoke = read(ROOT / "evaluation/foundation-v22-synthetic-smoke.json")
    generations = generation_summary(next(result for result in results if result["seed"] == 42))
    final_blind_path = ROOT / settings["final_blind"]["path"]
    final_blind_sha256 = file_sha256(final_blind_path)
    if final_blind_sha256 != settings["final_blind"]["expected_sha256"]:
        raise RuntimeError("Final Blind SHA256 mismatch")
    decision, evidence = gate(curve, generations, verification)
    final_generation = next((item for item in generations if item["tokens"] == 512_000), {"modes": {}})
    greedy = final_generation["modes"].get("greedy_no_penalty", {})
    language_status = "YES" if evidence["generation_emerged"] else (
        "PARTIAL" if greedy.get("character_validity", 0) > 0 else "NO"
    )
    summary = {
        "schema": "foundation-v22-summary-v1",
        "phase": 33,
        "architecture": "Current / 10L / 384D / 6H / FFN1536 / Pre-LN GELU / learned absolute position / final LN / tied embeddings",
        "parameters": 19_514_880,
        "seeds": list(SEEDS),
        "curve": curve,
        "generation": generations,
        "knowledge_completion": knowledge,
        "synthetic_smoke": smoke,
        "checkpoint_verification": verification,
        "final_blind": {"sha256": final_blind_sha256, "content_opened": False},
        "language_emergence_status": language_status,
        "gate": decision,
        "gate_evidence": evidence,
        "foundation_base_complete": False,
        "production_changed": False,
        "campus_changed": False,
        "final_blind_used": False,
    }
    output = ROOT / "evaluation/foundation-v22-summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# UniPilot Foundation v2.2 — PHASE 33 report",
        "",
        f"Gate: **{decision}**",
        "",
        "| tokens | val loss | top-1 | top-5 | top-10 | tok/s | peak RAM MB |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in curve:
        lines.append(f"| {row['tokens']} | {row['validation_loss']:.6f} | {row['top_1_accuracy']:.4f} | {row['top_5_accuracy']:.4f} | {row['top_10_accuracy']:.4f} | {row['training_tokens_per_second']:.1f} | {row['peak_ram_mb']:.1f} |")
    lines.extend([
        "",
        f"Language emergence (corrected observable proxy): **{language_status}**.",
        f"Checkpoint integrity: **{'PASS' if verification['integrity_pass'] else 'FAIL'}** ({verification['verified_checkpoints']} new checkpoints).",
        "Final Blind content was not opened; only its SHA256 was checked.",
        "Foundation Base is not complete. No production, Campus, deploy, or external API change was made.",
        "",
        "## Gate evidence",
        "",
        *[f"- {key}: {value}" for key, value in evidence.items()],
    ])
    report = ROOT / "evaluation/foundation-v22-report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"gate": decision, "language_emergence": language_status}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
