"""PHASE 46 fixed, sequential CPU evaluator for 15.872M short gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.diagnose_foundation_v29_generation import (
    build_prefixes, document_ranges, generate_batch, ngram_repetition, summarize_generation, target_metrics,
)
from evaluation.evaluate_foundation_v32_maturity import teacher_forced_horizons
from evaluation.evaluate_foundation_v33_context_gate import (
    CONTEXT_LENGTHS, context_positions, context_profile, sanity_checks, GREEDY, SAMPLE_T07,
)
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.train_foundation_v21_ab import file_sha256, frequency_ranks


SEEDS = (42, 123, 2026)
PROBE_TOKENS = 8192


def checkpoint(stage: str, seed: int) -> Path:
    if stage == "baseline":
        return ROOT / f"checkpoints/foundation-v33-context-gate/gate-2/seed-{seed}/checkpoint-tokens-15872000.pt"
    gate = 1 if stage == "gate1" else 2
    tokens = 16_128_000 if gate == 1 else 16_384_000
    return ROOT / f"checkpoints/foundation-v35-thermal-short-gate/gate-{gate}/seed-{seed}/checkpoint-tokens-{tokens}.pt"


def distribution(values: torch.Tensor) -> dict:
    data = values.double().numpy()
    logs = np.log(np.clip(data, 1e-30, None))
    return {
        "mean_correct_token_probability": float(np.mean(data)),
        "median_correct_token_probability": float(np.median(data)),
        "geometric_mean_correct_token_probability": float(np.exp(np.mean(logs))),
        "cross_entropy": float(-np.mean(logs)),
    }


@torch.inference_mode()
def language_metrics_detailed(model: DiagnosticTransformerV17, validation: np.memmap, ranks: np.ndarray) -> dict:
    model.eval()
    target_rows, top_rows, assigned_rows = [], [], []
    weighted_loss = 0.0
    context = model.config.context_length
    for start in range(0, PROBE_TOKENS, context):
        size = min(context, PROBE_TOKENS - start)
        values = np.asarray(validation[start:start + size + 1], dtype=np.int64).copy()
        inputs = torch.from_numpy(values[:-1]).unsqueeze(0)
        targets = torch.from_numpy(values[1:])
        logits, loss = model(inputs, targets.unsqueeze(0))
        scores = logits[0].float()
        target_rows.append(targets)
        top_rows.append(scores.topk(10, -1).indices)
        assigned_rows.append(torch.softmax(scores, -1).gather(1, targets[:, None]).squeeze(1))
        weighted_loss += float(loss) * size
    targets, top, assigned = torch.cat(target_rows), torch.cat(top_rows), torch.cat(assigned_rows)
    target_ranks = ranks[targets.numpy()]
    vocab = model.config.vocab_size
    boundaries = [math.ceil(vocab * value) for value in (.01, .05, .20, .80)]
    definitions = (
        ("top_1_percent", 0, boundaries[0]),
        ("top_5_percent_excluding_top_1", boundaries[0], boundaries[1]),
        ("top_20_percent_excluding_top_5", boundaries[1], boundaries[2]),
        ("middle_20_to_80_percent", boundaries[2], boundaries[3]),
        ("rare_bottom_20_percent", boundaries[3], vocab),
    )
    buckets = {}
    for name, low, high in definitions:
        mask = torch.from_numpy((target_ranks >= low) & (target_ranks < high))
        truth = targets[mask]
        buckets[name] = {
            "rank_range": [low, high - 1], "targets": int(mask.sum()),
            "top_1_accuracy": float((top[mask, 0] == truth).float().mean()),
            "top_5_accuracy": float((top[mask, :5] == truth[:, None]).any(-1).float().mean()),
            "top_10_accuracy": float((top[mask] == truth[:, None]).any(-1).float().mean()),
            **distribution(assigned[mask]),
        }
    loss = weighted_loss / PROBE_TOKENS
    return {
        "tokens": PROBE_TOKENS, "loss": loss, "perplexity": math.exp(loss),
        "top_1_accuracy": float((top[:, 0] == targets).float().mean()),
        "top_5_accuracy": float((top[:, :5] == targets[:, None]).any(-1).float().mean()),
        "top_10_accuracy": float((top == targets[:, None]).any(-1).float().mean()),
        "frequency_buckets": buckets,
    }


def generation_metrics(model: DiagnosticTransformerV17, tokenizer: FoundationTokenizer, prefixes: list[dict]) -> dict:
    prompts = [row["prefix_ids"] for row in prefixes]
    greedy_rows = generate_batch(model, tokenizer, prompts, GREEDY, list(range(100)), 128, trace=True)
    sampling_rows = generate_batch(model, tokenizer, prompts, SAMPLE_T07, [44_000 + i for i in range(100)], 64, trace=False)
    onsets = [row["loop"]["loop_onset"] for row in greedy_rows if row["loop"]["loop_onset"]]
    onset_steps = [
        step for row in greedy_rows for step in row["trace"]
        if row["loop"]["loop_onset"] and step["step"] == row["loop"]["loop_onset"]
    ]
    greedy = summarize_generation(greedy_rows)
    greedy.update({
        "first_break": any(not row["runaway"] for row in greedy_rows),
        "median_loop_onset": float(np.median(onsets)) if onsets else None,
        "mean_loop_onset": float(np.mean(onsets)) if onsets else None,
        "repetition": {str(n): float(np.mean([ngram_repetition(row["ids"], n) for row in greedy_rows])) for n in (1, 2, 3, 4)},
        "loop_onset_distribution": {
            "entropy": float(np.mean([row["entropy"] for row in onset_steps])),
            "top1_probability": float(np.mean([row["top5"][0]["probability"] for row in onset_steps])),
            "top1_top2_margin": float(np.mean([row["top1_top2_margin"] for row in onset_steps])),
            "eos_probability": float(np.mean([row["eos_probability"] for row in onset_steps])),
        },
    })
    sampling = summarize_generation(sampling_rows)
    sampling["topic_retention_proxy"] = float(np.mean([
        len(set(row["ids"]) & set(prompt)) / max(1, len(set(row["ids"])))
        for row, prompt in zip(sampling_rows, prompts)
    ]))
    return {"greedy": greedy, "temperature_0.7": sampling}


def evaluate(stage: str, seed: int) -> dict:
    started = time.perf_counter()
    path = checkpoint(stage, seed)
    before = file_sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    tokenizer = FoundationTokenizer.load(ROOT / "tokenizer/foundation-v11-base-4096.json")
    train = np.memmap(ROOT / "data/foundation_v11/packed/vocab-4096/train.bin", dtype=np.uint16, mode="r")
    validation = np.memmap(ROOT / "data/foundation_v11/packed/vocab-4096/validation.bin", dtype=np.uint16, mode="r")
    positions = context_positions(validation, tokenizer)
    ranks = frequency_ranks(train, tokenizer.vocab_size)
    ranges = document_ranges(validation, tokenizer.bos_id, tokenizer.eos_id)
    prefixes = build_prefixes(validation, ranges, tokenizer)
    terminal = np.resize(np.asarray([int(p) for p in np.flatnonzero(validation == tokenizer.eos_id) if p >= 128]), 500)
    nonterminal = np.linspace(128, len(validation) - 2, 500, dtype=int)
    result = {
        "schema": "foundation-v35-short-gate-evaluation-v1", "phase": 46, "stage": stage, "seed": seed,
        "tokens_processed": int(payload["tokens_processed"]), "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": before, "checkpoint_unchanged": file_sha256(path) == before,
        "context_target_positions_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
        "context": context_profile(model, validation, positions),
        "sanity": sanity_checks(model, payload, positions, validation, tokenizer),
        "validation": language_metrics_detailed(model, validation, ranks),
        "terminal_eos": target_metrics(model, validation, terminal, tokenizer.eos_id),
        "nonterminal_eos": target_metrics(model, validation, nonterminal, tokenizer.eos_id),
        "generation": generation_metrics(model, tokenizer, prefixes),
        "teacher_forced_horizons": teacher_forced_horizons(model, validation, prefixes),
        "evaluation_execution": {"device": "cpu", "parallel_cpu_evaluation": "DISABLED", "torch_threads": torch.get_num_threads()},
    }
    result["wall_seconds"] = time.perf_counter() - started
    target = ROOT / f"evaluation/phase46/{stage}/seed-{seed}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("baseline", "gate1", "gate2"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    for seed in SEEDS:
        result = evaluate(args.stage, seed)
        print(json.dumps({"stage": args.stage, "seed": seed, "loss": result["validation"]["loss"], "full_context": result["context"]["512"]["loss"], "seconds": result["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
