"""Deterministic expanded held-out evaluator for PHASE 44 context gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.diagnose_foundation_v29_generation import (
    build_prefixes,
    document_ranges,
    generate_batch,
    summarize_generation,
    target_metrics,
)
from evaluation.evaluate_foundation_v32_maturity import teacher_forced_horizons
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import (
    DiagnosticConfigV17,
    DiagnosticTransformerV17,
)
from training.train_foundation_v21_ab import file_sha256, frequency_ranks, language_metrics
from training.run_foundation_v33_context_gate import (
    EXPECTED_FORMAL_SHA256,
    SEEDS,
    formal_checkpoint,
    gate_checkpoint,
)


CONTEXT_LENGTHS = (512, 256, 128, 64, 32, 16, 8, 2, 1)
CONTEXT_TARGETS = 256
CONTEXT_POSITION_SEED = 44_033
GREEDY = {
    "name": "greedy",
    "kind": "greedy",
    "temperature": 1.0,
    "top_k": None,
    "top_p": None,
    "repetition_penalty": 1.0,
    "no_repeat_ngram": None,
    "eos_threshold": None,
}
SAMPLE_T07 = {**GREEDY, "name": "temperature_0.7", "kind": "sampling", "temperature": 0.7}


def checkpoint_for(stage: str, seed: int) -> Path:
    if stage == "baseline":
        return formal_checkpoint(seed)
    if stage == "history-10240":
        return (
            ROOT
            / "checkpoints/foundation-v26-current/current"
            / f"seed-{seed}/checkpoint-tokens-10240000.pt"
        )
    if stage == "gate1":
        return gate_checkpoint(1, seed)
    if stage == "gate2":
        return gate_checkpoint(2, seed)
    raise ValueError(stage)


def context_positions(validation: np.memmap, tokenizer: FoundationTokenizer) -> np.ndarray:
    candidates = np.arange(512, len(validation) - 1, dtype=np.int64)
    regular = ~np.isin(validation[candidates], list(tokenizer.special_to_id.values()))
    candidates = candidates[regular]
    rng = np.random.default_rng(CONTEXT_POSITION_SEED)
    return np.sort(rng.choice(candidates, size=CONTEXT_TARGETS, replace=False))


@torch.inference_mode()
def context_profile(
    model: DiagnosticTransformerV17,
    validation: np.memmap,
    positions: np.ndarray,
) -> dict:
    model.eval()
    device = next(model.parameters()).device
    targets = torch.from_numpy(np.asarray(validation[positions], dtype=np.int64).copy()).to(device)
    profile = {}
    for context in CONTEXT_LENGTHS:
        inputs = np.stack(
            [np.asarray(validation[position - context : position], dtype=np.int64) for position in positions]
        )
        per_target_losses = []
        assigned_probabilities = []
        correct = 0
        batch_size = 8 if context == 512 else 16 if context == 256 else 32
        for start in range(0, len(positions), batch_size):
            values = torch.from_numpy(inputs[start : start + batch_size].copy()).to(device)
            truth = targets[start : start + batch_size]
            logits, _ = model(values)
            scores = logits[:, -1].float()
            losses = F.cross_entropy(scores, truth, reduction="none")
            assigned = torch.softmax(scores, -1).gather(1, truth[:, None]).squeeze(1)
            per_target_losses.extend(losses.cpu().tolist())
            assigned_probabilities.extend(assigned.cpu().tolist())
            correct += int((scores.argmax(-1) == truth).sum())
        row = {
            "context_tokens": context,
            "targets": len(positions),
            "loss": float(np.mean(per_target_losses)),
            "loss_std": float(np.std(per_target_losses)),
            "loss_sem": float(np.std(per_target_losses, ddof=1) / math.sqrt(len(per_target_losses))),
            "perplexity": math.exp(min(float(np.mean(per_target_losses)), 50)),
            "top1_accuracy": correct / len(positions),
            "mean_correct_token_probability": float(np.mean(assigned_probabilities)),
        }
        if context == 512:
            row["per_target_losses"] = per_target_losses
        profile[str(context)] = row
    profile["full_context_advantage_vs_1"] = profile["1"]["loss"] - profile["512"]["loss"]
    profile["full_context_advantage_vs_2"] = profile["2"]["loss"] - profile["512"]["loss"]
    profile["full_context_advantage_vs_16"] = profile["16"]["loss"] - profile["512"]["loss"]
    profile["full_context_advantage_vs_64"] = profile["64"]["loss"] - profile["512"]["loss"]
    return profile


def generation_metrics(
    model: DiagnosticTransformerV17,
    tokenizer: FoundationTokenizer,
    prefixes: list[dict],
) -> dict:
    prompts = [row["prefix_ids"] for row in prefixes]
    greedy = generate_batch(
        model, tokenizer, prompts, GREEDY, list(range(100)), 128, trace=True
    )
    sampling = generate_batch(
        model,
        tokenizer,
        prompts,
        SAMPLE_T07,
        [44_000 + index for index in range(100)],
        64,
        trace=False,
    )
    onsets = [row["loop"]["loop_onset"] for row in greedy if row["loop"]["loop_onset"]]
    onset_steps = [
        step
        for row in greedy
        for step in row["trace"]
        if row["loop"]["loop_onset"] and step["step"] == row["loop"]["loop_onset"]
    ]
    greedy_summary = summarize_generation(greedy)
    greedy_summary.update(
        {
            "median_loop_onset": float(np.median(onsets)) if onsets else None,
            "mean_loop_onset": float(np.mean(onsets)) if onsets else None,
            "loop_onset_distribution": {
                "entropy": float(np.mean([row["entropy"] for row in onset_steps])),
                "top1_probability": float(
                    np.mean([row["top5"][0]["probability"] for row in onset_steps])
                ),
                "top1_top2_margin": float(
                    np.mean([row["top1_top2_margin"] for row in onset_steps])
                ),
                "eos_probability": float(
                    np.mean([row["eos_probability"] for row in onset_steps])
                ),
            },
        }
    )
    return {"greedy": greedy_summary, "temperature_0.7": summarize_generation(sampling)}


def sanity_checks(
    model: DiagnosticTransformerV17,
    payload: dict,
    positions: np.ndarray,
    validation: np.memmap,
    tokenizer: FoundationTokenizer,
) -> dict:
    dropout_off = not model.training and all(
        not module.training for module in model.modules() if isinstance(module, torch.nn.Dropout)
    )
    checks = {
        "held_out_validation_only": True,
        "validation_path": "data/foundation_v11/packed/vocab-4096/validation.bin",
        "tokenizer_path": "tokenizer/foundation-v11-base-4096.json",
        "tokenizer_vocab": tokenizer.vocab_size,
        "same_target_positions_all_contexts": True,
        "target_positions_count": len(positions) == CONTEXT_TARGETS,
        "target_positions_in_bounds": int(positions.min()) >= 512
        and int(positions.max()) < len(validation),
        "model_eval": not model.training,
        "dropout_off": dropout_off,
        "inference_mode": True,
        "device": str(next(model.parameters()).device),
        "context_truncation": list(CONTEXT_LENGTHS),
        "token_weighting": "unweighted_cross_entropy_for_every_context",
        "special_target_handling": "same_regular-token target sample for every context",
        "batching": "fixed_by_context_and_identical_across_checkpoints",
        "config_unchanged": payload["config"] == model.config.to_dict(),
    }
    boolean_checks = [value for value in checks.values() if isinstance(value, bool)]
    checks["pass"] = all(boolean_checks)
    return checks


def evaluate(stage: str, seed: int, context_only: bool) -> dict:
    started = time.perf_counter()
    path = checkpoint_for(stage, seed)
    before = file_sha256(path)
    if stage == "baseline" and before != EXPECTED_FORMAL_SHA256[seed]:
        raise RuntimeError(f"formal checkpoint SHA mismatch for seed {seed}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    tokenizer = FoundationTokenizer.load(ROOT / "tokenizer/foundation-v11-base-4096.json")
    train = np.memmap(
        ROOT / "data/foundation_v11/packed/vocab-4096/train.bin", dtype=np.uint16, mode="r"
    )
    validation = np.memmap(
        ROOT / "data/foundation_v11/packed/vocab-4096/validation.bin",
        dtype=np.uint16,
        mode="r",
    )
    positions = context_positions(validation, tokenizer)
    context = context_profile(model, validation, positions)
    result = {
        "schema": "foundation-v33-context-gate-evaluation-v1",
        "phase": 44,
        "stage": stage,
        "seed": seed,
        "tokens_processed": int(payload["tokens_processed"]),
        "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": before,
        "checkpoint_unchanged": file_sha256(path) == before,
        "context_target_positions_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
        "context": context,
        "sanity": sanity_checks(model, payload, positions, validation, tokenizer),
        "context_only": context_only,
    }
    if not context_only:
        ranks = frequency_ranks(train, tokenizer.vocab_size)
        ranges = document_ranges(validation, tokenizer.bos_id, tokenizer.eos_id)
        prefixes = build_prefixes(validation, ranges, tokenizer)
        terminal_unique = np.asarray(
            [int(position) for position in np.flatnonzero(validation == tokenizer.eos_id) if position >= 128]
        )
        terminal = np.resize(terminal_unique, 500)
        nonterminal = np.linspace(128, len(validation) - 2, 500, dtype=int)
        result.update(
            {
                "validation": language_metrics(model, tokenizer, validation, ranks, 8192),
                "terminal_eos": target_metrics(model, validation, terminal, tokenizer.eos_id),
                "nonterminal_eos": target_metrics(
                    model, validation, nonterminal, tokenizer.eos_id
                ),
                "generation": generation_metrics(model, tokenizer, prefixes),
                "teacher_forced_horizons": teacher_forced_horizons(
                    model, validation, prefixes
                ),
            }
        )
    result["wall_seconds"] = time.perf_counter() - started
    destination = ROOT / f"evaluation/phase44/{stage}/seed-{seed}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("baseline", "history-10240", "gate1", "gate2"), required=True
    )
    parser.add_argument("--context-only", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    for seed in args.seeds:
        result = evaluate(args.stage, seed, args.context_only)
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "seed": seed,
                    "tokens": result["tokens_processed"],
                    "full_context_loss": result["context"]["512"]["loss"],
                    "full_advantage": result["context"]["full_context_advantage_vs_1"],
                    "validation_loss": result.get("validation", {}).get("loss"),
                    "wall_seconds": result["wall_seconds"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
