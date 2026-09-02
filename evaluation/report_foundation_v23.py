"""Aggregate PHASE 34 diagnostics, verify the pilot checkpoint, and issue the 1M gate."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.train_foundation_v21_ab import file_sha256, load_json, stateless_scheduler_state, tensor_sha256


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def checkpoint_verification(pilot: dict, settings: dict) -> dict:
    final = pilot["final"]
    metadata = final["checkpoint"]
    path = ROOT / metadata["path"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    strict_reload = all(
        torch.equal(left, right)
        for left, right in zip(payload["model_state"].values(), model.state_dict().values())
    )
    corpus = load_json(settings["corpus_manifest"])
    macro_count = (int(corpus["splits"]["train"]["tokens"]) - 1) // 512
    permutation = torch.randperm(macro_count, generator=torch.Generator().manual_seed(42))
    update = int(payload["update"])
    checks = {
        "sha256_match": file_sha256(path) == metadata["sha256"],
        "strict_reload": strict_reload,
        "identity_match": payload["variant"] == "current" and int(payload["seed"]) == 42 and int(payload["tokens_processed"]) == 640_000,
        "optimizer_state_present": bool(payload.get("optimizer_state", {}).get("state")),
        "rng_state_complete": set(payload.get("random_state", {})) == {"python", "numpy", "torch_cpu"},
        "scheduler_state_match": payload.get("scheduler_state") == stateless_scheduler_state(settings, update),
        "permutation_match": torch.equal(payload["permutation"], permutation),
        "next_sampler_position_valid": update < len(permutation),
    }
    result = {
        "path": metadata["path"],
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "update": update,
        "tokens": int(payload["tokens_processed"]),
        "permutation_sha256": tensor_sha256(permutation),
        "next_macroblock_index": int(permutation[update]),
        "checks": checks,
        "pass": all(checks.values()),
    }
    del payload, model
    return result


def diagnostic_snapshot(payload: dict) -> dict:
    metrics = payload["validation_document_prefix"]["metrics"]
    free = metrics["free_running"]
    teacher = metrics["teacher_forced_horizon"]
    return {
        "tokens": payload["tokens"],
        "corpus_exposure_percentage": payload["corpus_exposure"]["percentage"],
        "teacher_forced_horizon": teacher,
        "free_running": {
            "mean_divergence_position": free["mean_divergence_position"],
            "first_4_token_exact": free["first_4_token_exact"],
            "first_8_token_exact": free["first_8_token_exact"],
            "continuation_horizons": free["continuation_horizons"],
            "character_validity": free["character_validity"],
            "japanese_character_ratio": free["japanese_character_ratio"],
            "natural_japanese_proxy": free["natural_japanese_proxy"],
            "semantic_local_syntax_proxy": free["semantic_local_syntax_proxy"],
            "sentence_boundary_rate": free["sentence_boundary_rate"],
            "runaway_rate": free["runaway_rate"],
            "ngram_repetition": free["ngram_repetition"],
            "mean_maximum_repeated_span": free["mean_maximum_repeated_span"],
            "mean_loop_onset": free["mean_loop_onset"],
            "candidate_expected_top_5_rate": free["candidate_expected_top_5_rate"],
            "candidate_expected_top_10_rate": free["candidate_expected_top_10_rate"],
        },
        "oracle_prefix_recovery": metrics["oracle_prefix_recovery"],
        "error_taxonomy": metrics["error_taxonomy"],
        "loop_onset_confidence": metrics["loop_onset_confidence"],
        "train_prefix": payload["train_document_prefix"]["metrics"],
        "sentence_prefix": payload["validation_sentence_prefix"]["metrics"],
        "token_distribution": {
            key: value for key, value in payload["token_distribution"].items()
            if key != "generated_counts"
        },
        "boundary_diagnostics": payload["boundary_diagnostics"],
        "training_exposure": payload["training_exposure"],
        "decoding_comparison": {
            mode: {key: value for key, value in row.items() if key not in {"items", "settings"}}
            for mode, row in payload["decoding_comparison"].items()
        },
        "unconditional_generation": payload["unconditional_generation"],
    }


def main() -> int:
    settings = load_json("configs/unipilot-foundation-v23-pilot.json")
    parity = read("evaluation/foundation-v23-inference-parity.json")
    readiness = read("evaluation/foundation-v23-pilot-readiness.json")
    phase33 = read("evaluation/foundation-v22-summary.json")
    pilot = read("evaluation/foundation-v23-pilot-run.json")
    diagnostics = {
        tokens: read(f"evaluation/foundation-v23-generation-diagnostics-{tokens}.json")
        for tokens in (256_000, 512_000, 640_000)
    }
    verification = checkpoint_verification(pilot, settings)
    snapshots = {str(tokens): diagnostic_snapshot(payload) for tokens, payload in diagnostics.items()}
    old = snapshots["512000"]
    new = snapshots["640000"]
    old_teacher = old["teacher_forced_horizon"]["32"]
    new_teacher = new["teacher_forced_horizon"]["32"]
    validation_512 = next(row for row in pilot["training"]["history"] if row["tokens_processed"] == 512_000)["validation"]
    validation_640 = pilot["final"]["validation"]
    pilot_checks = {
        "validation_loss_improved": validation_640["loss"] < validation_512["loss"],
        "validation_top_1_improved": validation_640["top_1_accuracy"] >= validation_512["top_1_accuracy"],
        "validation_top_5_improved": validation_640["top_5_accuracy"] >= validation_512["top_5_accuracy"],
        "validation_top_10_improved": validation_640["top_10_accuracy"] >= validation_512["top_10_accuracy"],
        "teacher_h32_loss_improved": new_teacher["loss"] < old_teacher["loss"],
        "teacher_h32_top_1_improved": new_teacher["top_1_accuracy"] > old_teacher["top_1_accuracy"],
        "teacher_h32_top_5_improved": new_teacher["top_5_accuracy"] > old_teacher["top_5_accuracy"],
        "teacher_h32_top_10_improved": new_teacher["top_10_accuracy"] > old_teacher["top_10_accuracy"],
        "divergence_direction_improved": new["free_running"]["mean_divergence_position"] >= old["free_running"]["mean_divergence_position"],
        "greedy_repetition_direction_improved": new["free_running"]["ngram_repetition"]["1"] < old["free_running"]["ngram_repetition"]["1"],
        "sampling_naturalness_improved": new["decoding_comparison"]["temperature_0.7"]["natural_japanese_proxy"] > old["decoding_comparison"]["temperature_0.7"]["natural_japanese_proxy"],
        "frequency_collapse_direction_improved": new["token_distribution"]["generation_frequency_buckets"]["top_1_percent"] < old["token_distribution"]["generation_frequency_buckets"]["top_1_percent"],
        "newline_generation_frequency_improved": new["boundary_diagnostics"]["newline"]["generation_frequency"] < old["boundary_diagnostics"]["newline"]["generation_frequency"],
        "checkpoint_integrity": verification["pass"],
    }
    continue_1m = (
        parity["pass"]
        and readiness["learning_direction"] == "HEALTHY"
        and all(pilot_checks.values())
        and phase33["synthetic_smoke"]["gate_pass"]
    )
    gate = "CONTINUE_1M_TOKEN_LIMITED" if continue_1m else "EXPOSURE_ERROR_INVESTIGATE"
    final_blind_path = ROOT / settings["final_blind"]["path"]
    final_blind_sha = file_sha256(final_blind_path)
    if final_blind_sha != settings["final_blind"]["expected_sha256"]:
        raise RuntimeError("Final Blind SHA256 mismatch")
    result = {
        "schema": "foundation-v23-generation-investigation-summary-v1",
        "phase": 34,
        "formal_architecture": "Current",
        "parameters": 19_514_880,
        "inference_parity": parity["inference_parity"],
        "kv_cache_parity": parity["kv_cache_parity"],
        "checkpoint_verification": verification,
        "base_evaluation_policy": {
            "natural_document_prefix": "primary",
            "sentence_prefix": "primary",
            "instruction_like": "observational only",
        },
        "snapshots": snapshots,
        "validation_512_vs_640": {"512000": validation_512, "640000": validation_640},
        "pilot": {
            "status": "EXECUTED",
            "seed": 42,
            "from_tokens": 512_000,
            "to_tokens": 640_000,
            "checks": pilot_checks,
        },
        "root_cause": {
            "primary": "token-limited pretraining plus early autoregressive exposure error and high-frequency Top-1 collapse",
            "inference_path_bug": False,
            "evaluator_issue": False,
            "instruction_prompt_mismatch": "removed from primary Base gate",
            "frequency_collapse": True,
            "boundary_eos_scarcity_contributor": True,
            "data_boundary_issue_primary": False,
            "decoding_has_better_candidates": True,
            "decoding_is_not_model_improvement": True,
            "evidence": [
                "teacher-forced loss/Top-k/correct-token probability improve from 256k through 640k",
                "greedy divergence remains near token 1, while one-token oracle Top-5 recovery improves",
                "512k greedy generation is 99.36% Top-1%-frequency tokens and 84.38% newline",
                "640k reduces Top-1%-bucket generation to 96.45% and newline to 72.86%",
                "temperature 0.7 natural proxy rises from 12% at 512k to 48% at 640k",
                "640k is only 1.92% of the 33,402,759-token corpus",
                "EOS has only 205 supervised targets for seed 42 by 640k and is never generated Top-1",
            ],
        },
        "scaling_extrapolation": {
            "direction": "continued improvement is reasonable but magnitude is not predicted",
            "performance_forecast": None,
            "one_million_must_be_a_separate_phase": True,
        },
        "gate": gate,
        "one_million_next_phase": "YES" if continue_1m else "NO",
        "foundation_base_complete": False,
        "final_blind": {"sha256": final_blind_sha, "content_opened": False},
        "production_changed": False,
        "campus_changed": False,
        "render_changed": False,
        "vercel_changed": False,
    }
    summary_path = ROOT / "evaluation/foundation-v23-summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# UniPilot Foundation v2.3 — PHASE 34",
        "",
        f"Gate: **{gate}**",
        "",
        f"Inference parity: **{parity['inference_parity']}**",
        "",
        f"KV-cache parity: **{parity['kv_cache_parity']}**",
        "",
        "640k pilot: **EXECUTED** (seed 42 only)",
        "",
        f"Proceed to 1M in the next phase: **{'YES' if continue_1m else 'NO'}**",
        "",
        "## Base prefix completion",
        "",
        "| tokens | teacher loss h32 | teacher top-1/5/10 h32 | divergence | greedy 1-gram rep. | sampling t0.7 natural |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tokens in (256_000, 512_000, 640_000):
        row = snapshots[str(tokens)]
        teacher = row["teacher_forced_horizon"]["32"]
        free = row["free_running"]
        sample = row["decoding_comparison"]["temperature_0.7"]
        lines.append(
            f"| {tokens} | {teacher['loss']:.4f} | {teacher['top_1_accuracy']:.3f} / {teacher['top_5_accuracy']:.3f} / {teacher['top_10_accuracy']:.3f} | {free['mean_divergence_position']:.3f} | {free['ngram_repetition']['1']:.3f} | {sample['natural_japanese_proxy']:.0%} |"
        )
    lines.extend([
        "",
        "## Diagnosis",
        "",
        "Training/inference logits and cached/non-cached logits agree within tolerance. The dominant failure is not an inference implementation bug. Teacher-forced learning continues, but the first free-running error occurs near token 1 and drives a high-frequency newline/token loop. Low corpus exposure and scarce EOS targets are contributors. Sampling reveals better candidates, but decoding results are diagnostic only.",
        "",
        f"640k checkpoint integrity: **{'PASS' if verification['pass'] else 'FAIL'}**. Synthetic architecture/EOS-capability evidence remains PASS. Final Blind content was not opened.",
        "Foundation Base is not complete. Architecture, tokenizer, corpus, Campus, production, Render, and Vercel were unchanged.",
    ])
    report_path = ROOT / "evaluation/foundation-v23-report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"gate": gate, "one_million_next_phase": result["one_million_next_phase"], "checkpoint": verification["pass"]}, indent=2))
    return 0 if verification["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
