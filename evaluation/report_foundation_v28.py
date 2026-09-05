"""Build the concise PHASE 39 Foundation v2.8 final report."""
from __future__ import annotations

import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2026)
MILESTONES = (10_240_000, 12_288_000, 15_360_000)


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def history_row(run: dict, tokens: int) -> dict:
    return next(row for row in run["training"]["history"] if row["tokens_processed"] == tokens)


def main() -> int:
    runs = [load(f"evaluation/foundation-v28-runs/current-seed-{seed}.json") for seed in SEEDS]
    checkpoint_integrity = load("evaluation/foundation-v28-checkpoint-verification.json")
    migration = load("evaluation/gpu-migration-report.json")
    curve = []
    for tokens in MILESTONES:
        rows = [history_row(run, tokens) for run in runs]
        validation = [row["validation"] for row in rows]
        curve.append({
            "tokens": tokens,
            "loss": statistics.fmean(row["loss"] for row in validation),
            "top_1": statistics.fmean(row["top_1_accuracy"] for row in validation),
            "top_5": statistics.fmean(row["top_5_accuracy"] for row in validation),
            "top_10": statistics.fmean(row["top_10_accuracy"] for row in validation),
            "corpus_exposure_percent": statistics.fmean(row["corpus_percentage"] for row in rows),
        })
    representative = runs[0]["final"]
    sampling = representative["generation"]["sampling_t07_topk40_topp09_no_penalty"]["metrics"]
    greedy = representative["generation"]["greedy_no_penalty"]["metrics"]
    if (
        sampling["natural_japanese_proxy"] >= .5
        and sampling["semantic_coherence_proxy"] >= .5
        and greedy["runaway_rate"] <= .1
    ):
        language_emergence = "YES"
    elif sampling["natural_japanese_proxy"] > 0 and sampling["semantic_coherence_proxy"] > 0:
        language_emergence = "PARTIAL"
    else:
        language_emergence = "NOT_YET"
    base_maturing = curve[-1]["loss"] < curve[0]["loss"]
    foundation_complete = language_emergence == "YES" and checkpoint_integrity["integrity_pass"]
    training_speeds = {
        str(seed): runs[index]["training"]["session_training_tokens_per_second"]
        for index, seed in enumerate(SEEDS)
    }
    training_telemetry = {}
    for index, seed in enumerate(SEEDS):
        telemetry_rows = [
            row["gpu_telemetry"]
            for row in runs[index]["training"]["history"]
            if row.get("gpu_telemetry") and row["gpu_telemetry"].get("samples", 0)
        ]
        final_telemetry = runs[index]["training"].get("gpu_telemetry_final")
        if final_telemetry and final_telemetry.get("samples", 0):
            telemetry_rows.append(final_telemetry)
        training_telemetry[str(seed)] = {
            "gpu_utilization_percent_max": max(
                row["gpu_utilization_percent_max"] for row in telemetry_rows
            ),
            "gpu_temperature_c_max": max(row["gpu_temperature_c_max"] for row in telemetry_rows),
            "gpu_power_w_max": max(row["gpu_power_w_max"] for row in telemetry_rows),
            "nvidia_smi_memory_used_mib_max": max(
                row["gpu_memory_used_mib_max"] for row in telemetry_rows
            ),
            "torch_peak_vram_mb": runs[index]["training"]["peak_vram_mb"],
        }
    final_gate = (
        "FOUNDATION_BASE_COMPLETE"
        if foundation_complete else "REVIEW_REQUIRED_GENERATION_LAG_AT_15M"
    )
    output = {
        "schema": "foundation-v28-summary-v1",
        "phase": 39,
        "phase38_final_tokens": 10_240_000,
        "phase38_gate": "CONTINUE_15M_GENERATION_LAG",
        "final_tokens": 15_360_000,
        "training_curve": curve,
        "sampling": {
            "naturalness": sampling["natural_japanese_proxy"],
            "semantic_coherence": sampling["semantic_coherence_proxy"],
        },
        "greedy": {
            "repetition": greedy["mean_repetition_rate"],
            "runaway": greedy["runaway_rate"],
        },
        "language_emergence": language_emergence,
        "base_maturing": base_maturing,
        "foundation_base_complete": foundation_complete,
        "final_gate": final_gate,
        "checkpoint_integrity": checkpoint_integrity,
        "gpu_migration": {
            "gate": migration["gpu_migration_gate"],
            "speed_gate": migration["speed_gate"],
            "cpu_tokens_per_second": migration["benchmark"]["cpu_fp32"]["tokens_per_second"],
            "gpu_tokens_per_second": migration["benchmark"]["gpu_fp32"]["tokens_per_second"],
            "speedup": migration["benchmark"]["speedup"],
            "peak_vram_mb": migration["peak_vram_mb"],
            "amp_tested": migration["amp"]["tested"],
            "amp_adopted": migration["amp"]["adopted"],
        },
        "training_tokens_per_second_by_seed": training_speeds,
        "gpu_training_telemetry_by_seed": training_telemetry,
        "gpu_training_telemetry_max": {
            key: max(row[key] for row in training_telemetry.values())
            for key in next(iter(training_telemetry.values()))
        },
        "precision_mode": "fp32",
        "final_blind": {
            "sha256": "fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b",
            "content_opened": False,
        },
    }
    (ROOT / "evaluation/foundation-v28-summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    final = curve[-1]
    markdown = (
        "# Foundation v2.8 — PHASE 39\n\n"
        f"Final Gate: **{final_gate}**\n\n"
        f"15.360M loss {final['loss']:.4f}; Top-1/5/10 "
        f"{final['top_1']:.2%}/{final['top_5']:.2%}/{final['top_10']:.2%}. "
        f"Sampling naturalness/semantic {sampling['natural_japanese_proxy']:.0%}/"
        f"{sampling['semantic_coherence_proxy']:.0%}; greedy repetition/runaway "
        f"{greedy['mean_repetition_rate']:.3f}/{greedy['runaway_rate']:.0%}.\n"
    )
    (ROOT / "evaluation/foundation-v28-report.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "gate": final_gate,
        "tokens": output["final_tokens"],
        "loss": final["loss"],
        "foundation_base_complete": foundation_complete,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
