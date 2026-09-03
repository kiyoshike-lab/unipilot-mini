"""PHASE 34 Base-LM generation dynamics diagnostics at fixed token milestones."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate_foundation_v13 import KNOWLEDGE_PROBES
from evaluation.investigate_foundation_v14 import language_proxy, sample_token
from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import DiagnosticConfigV17, DiagnosticTransformerV17
from training.train_foundation_v21_ab import file_sha256, load_json


PREFIX_LENGTHS = (16, 32, 64, 128)
CONTINUATION_HORIZONS = (8, 16, 32, 64)
TEACHER_HORIZONS = (1, 2, 4, 8, 16, 32)
BOUNDARIES = ("。", "！", "？", "newline", "<EOS>")
PARTICLES = ("の", "に", "は", "を", "が", "と", "で")
GREEDY = {
    "name": "greedy",
    "kind": "greedy",
    "temperature": 1.0,
    "top_k": None,
    "top_p": None,
    "repetition_penalty": 1.0,
}
DECODING_MODES = (
    GREEDY,
    {**GREEDY, "name": "temperature_0.7", "kind": "sampling", "temperature": 0.7},
    {**GREEDY, "name": "temperature_1.0", "kind": "sampling"},
    {**GREEDY, "name": "top_k_20", "kind": "sampling", "top_k": 20},
    {**GREEDY, "name": "top_k_50", "kind": "sampling", "top_k": 50},
    {**GREEDY, "name": "top_p_0.9", "kind": "sampling", "top_p": 0.9},
)


def checkpoint_path(tokens: int) -> Path:
    if tokens == 256_000:
        return ROOT / "checkpoints/foundation-v21-ab/current/seed-42/checkpoint-tokens-256000.pt"
    if tokens == 512_000:
        return ROOT / "checkpoints/foundation-v22-current/current/seed-42/checkpoint-tokens-512000.pt"
    if tokens == 640_000:
        return ROOT / "checkpoints/foundation-v23-pilot/current/seed-42/checkpoint-tokens-640000.pt"
    if tokens in {768_000, 896_000, 1_024_000}:
        return ROOT / f"checkpoints/foundation-v24-current/current/seed-42/checkpoint-tokens-{tokens}.pt"
    if tokens in {1_280_000, 1_536_000, 1_792_000, 2_048_000}:
        return ROOT / f"checkpoints/foundation-v25-current/current/seed-42/checkpoint-tokens-{tokens}.pt"
    raise ValueError(f"unsupported diagnostic milestone: {tokens}")


def load_model(tokens: int) -> tuple[DiagnosticTransformerV17, dict]:
    path = checkpoint_path(tokens)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = DiagnosticTransformerV17(DiagnosticConfigV17(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload


def document_ranges(data: np.memmap, bos_id: int, eos_id: int) -> list[tuple[int, int]]:
    bos = np.flatnonzero(data == bos_id)
    eos = np.flatnonzero(data == eos_id)
    ranges = []
    eos_cursor = 0
    for start in bos:
        while eos_cursor < len(eos) and eos[eos_cursor] <= start:
            eos_cursor += 1
        if eos_cursor >= len(eos):
            break
        ranges.append((int(start) + 1, int(eos[eos_cursor])))
        eos_cursor += 1
    return ranges


def sample_document_prefixes(
    data: np.memmap,
    ranges: list[tuple[int, int]],
    split: str,
    per_length: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    shuffled = list(ranges)
    rng.shuffle(shuffled)
    rows = []
    for prefix_length in PREFIX_LENGTHS:
        candidates = [(start, end) for start, end in shuffled if end - start >= prefix_length + 64]
        if len(candidates) < per_length:
            raise RuntimeError(f"insufficient {split} documents for prefix length {prefix_length}")
        for index, (start, end) in enumerate(candidates[:per_length]):
            prefix = np.asarray(data[start:start + prefix_length], dtype=np.int64).tolist()
            truth = np.asarray(data[start + prefix_length:start + prefix_length + 64], dtype=np.int64).tolist()
            rows.append({
                "id": f"{split}-document-p{prefix_length}-{index:03d}",
                "split": split,
                "kind": "natural_document_prefix",
                "prefix_length": prefix_length,
                "source_start": start,
                "document_end": end,
                "prefix_ids": prefix,
                "truth_ids": truth,
            })
    return rows


def sample_sentence_prefixes(
    data: np.memmap,
    ranges: list[tuple[int, int]],
    boundary_id: int,
    count: int,
    seed: int,
) -> list[dict]:
    candidates = []
    for start, end in ranges:
        positions = np.flatnonzero(data[start:end] == boundary_id)
        for position in positions:
            sentence_start = start + int(position) + 1
            if sentence_start + 32 + 64 <= end:
                candidates.append((sentence_start, end))
    random.Random(seed).shuffle(candidates)
    if len(candidates) < count:
        raise RuntimeError("insufficient held-out sentence prefixes")
    rows = []
    for index, (start, end) in enumerate(candidates[:count]):
        rows.append({
            "id": f"validation-sentence-p32-{index:03d}",
            "split": "validation",
            "kind": "sentence_prefix",
            "prefix_length": 32,
            "source_start": start,
            "document_end": end,
            "prefix_ids": np.asarray(data[start:start + 32], dtype=np.int64).tolist(),
            "truth_ids": np.asarray(data[start + 32:start + 96], dtype=np.int64).tolist(),
        })
    return rows


def ngram_repetition(ids: list[int], n: int) -> float:
    grams = [tuple(ids[index:index + n]) for index in range(max(0, len(ids) - n + 1))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def loop_details(ids: list[int]) -> dict:
    best_span = 0
    best_loop_length = None
    best_onset = None
    for onset in range(len(ids)):
        for loop_length in range(1, min(16, (len(ids) - onset) // 2) + 1):
            pattern = ids[onset:onset + loop_length]
            cursor = onset + loop_length
            while cursor + loop_length <= len(ids) and ids[cursor:cursor + loop_length] == pattern:
                cursor += loop_length
            span = cursor - onset
            if span >= loop_length * 2 and span > best_span:
                best_span = span
                best_loop_length = loop_length
                best_onset = onset + 1
    return {
        "maximum_repeated_span": best_span,
        "loop_length": best_loop_length,
        "loop_onset_token": best_onset,
    }


def evaluator_reasons(text: str, proxy: dict) -> list[str]:
    visible = re.sub(r"\s+", "", text.strip())
    reasons = []
    if not proxy["character_valid"]:
        reasons.append("invalid_character_sequence")
    if len(visible) < 20:
        reasons.append("visible_length_below_20")
    if proxy["japanese_character_ratio"] < 0.35:
        reasons.append("japanese_character_ratio_below_0.35")
    if proxy["punctuation_ratio"] > 0.25:
        reasons.append("punctuation_ratio_above_0.25")
    if proxy["newline_ratio"] > 0.30:
        reasons.append("newline_ratio_above_0.30")
    if proxy["repetition_rate"] >= 0.35:
        reasons.append("repetition_rate_at_least_0.35")
    if proxy["content_run_count"] < 3:
        reasons.append("content_run_count_below_3")
    return reasons or ["all_natural_japanese_proxy_conditions_passed"]


@torch.inference_mode()
def teacher_forced_metrics(
    model: DiagnosticTransformerV17,
    prefix: list[int],
    truth: list[int],
) -> dict:
    values = prefix + truth[:-1]
    logits, _ = model(torch.tensor([values], dtype=torch.long))
    continuation_logits = logits[0, len(prefix) - 1:len(prefix) - 1 + len(truth)].float()
    targets = torch.tensor(truth, dtype=torch.long)
    log_probabilities = torch.log_softmax(continuation_logits, dim=-1)
    losses = -log_probabilities.gather(1, targets[:, None]).squeeze(1)
    top10 = torch.topk(continuation_logits, 10, dim=-1).indices
    rows = {}
    for horizon in TEACHER_HORIZONS:
        target = targets[:horizon, None]
        rows[str(horizon)] = {
            "loss": float(losses[:horizon].mean()),
            "top_1_accuracy": float((top10[:horizon, :1] == target).any(dim=1).float().mean()),
            "top_5_accuracy": float((top10[:horizon, :5] == target).any(dim=1).float().mean()),
            "top_10_accuracy": float((top10[:horizon] == target).any(dim=1).float().mean()),
            "mean_correct_token_probability": float(log_probabilities[:horizon].exp().gather(1, target).mean()),
        }
    return rows


@torch.inference_mode()
def traced_generate(
    model: DiagnosticTransformerV17,
    tokenizer: FoundationTokenizer,
    prefix: list[int],
    truth: list[int],
    mode: dict,
    seed: int,
    max_new_tokens: int = 64,
) -> dict:
    all_ids = list(prefix)
    generated = []
    trace = []
    generator = torch.Generator().manual_seed(seed)
    forbidden = [value for token, value in tokenizer.special_to_id.items() if token != "<EOS>"]
    cache = None
    for step in range(max_new_tokens):
        current = all_ids if cache is None else [all_ids[-1]]
        logits, _, cache = model(
            torch.tensor([current], dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
        )
        scores = logits[0, -1].float()
        probabilities = torch.softmax(scores, dim=-1)
        top_probabilities, top_ids = torch.topk(probabilities, 10)
        filtered = scores.clone()
        filtered[forbidden] = -torch.inf
        next_id = sample_token(filtered, mode, generator)
        expected = truth[step] if step < len(truth) else None
        trace.append({
            "step": step + 1,
            "selected_id": next_id,
            "selected_piece": tokenizer.decode([next_id], skip_special=False),
            "expected_id": expected,
            "expected_in_top_5": expected in top_ids[:5].tolist() if expected is not None else None,
            "expected_in_top_10": expected in top_ids.tolist() if expected is not None else None,
            "top_1_probability": float(top_probabilities[0]),
            "top_2_probability": float(top_probabilities[1]),
            "entropy": float(-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum()),
            "top_1_top_2_margin": float(top_probabilities[0] - top_probabilities[1]),
            "top_10": [
                {
                    "id": int(token_id),
                    "piece": tokenizer.decode([int(token_id)], skip_special=False),
                    "probability": float(probability),
                }
                for token_id, probability in zip(top_ids.tolist(), top_probabilities.tolist())
            ],
        })
        generated.append(next_id)
        all_ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break
    text = tokenizer.decode(generated, skip_special=True)
    proxy = language_proxy(text, eos_reached=bool(generated and generated[-1] == tokenizer.eos_id))
    divergence = next((index + 1 for index, (left, right) in enumerate(zip(generated, truth)) if left != right), min(len(generated), len(truth)) + 1)
    loops = loop_details(generated)
    punctuation_ids = {
        tokenizer.eos_id,
        *tokenizer.encode("。", add_bos=False),
        *tokenizer.encode("、", add_bos=False),
        *tokenizer.encode("！", add_bos=False),
        *tokenizer.encode("？", add_bos=False),
        *tokenizer.encode("\n", add_bos=False),
    }
    loop_index = (loops["loop_onset_token"] - 1) if loops["loop_onset_token"] else None
    loop_pattern = (
        generated[loop_index:loop_index + loops["loop_length"]]
        if loop_index is not None else []
    )
    punctuation_loop_onset = (
        loops["loop_onset_token"] if loop_pattern and all(token in punctuation_ids for token in loop_pattern)
        else None
    )
    invalid_sequence_onset = next((
        index + 1 for index, token in enumerate(generated)
        if "�" in tokenizer.decode([token], skip_special=False)
    ), None)
    runaway = len(generated) == max_new_tokens and generated[-1] != tokenizer.eos_id
    return {
        "ids": generated,
        "text": text,
        "tokens": len(generated),
        "trace": trace,
        "divergence_position": divergence,
        "eos_reached": bool(generated and generated[-1] == tokenizer.eos_id),
        "runaway": runaway,
        "runaway_onset_token": max_new_tokens if runaway else None,
        "ngram_repetition": {str(n): ngram_repetition(generated, n) for n in range(1, 5)},
        **loops,
        "repetition_onset_token": loops["loop_onset_token"],
        "punctuation_loop_onset_token": punctuation_loop_onset,
        "invalid_sequence_onset_token": invalid_sequence_onset,
        **proxy,
        "evaluator_reasons": evaluator_reasons(text, proxy),
    }


@torch.inference_mode()
def oracle_recovery(
    model: DiagnosticTransformerV17,
    prefix: list[int],
    truth: list[int],
    generated: list[int],
    divergence_position: int,
) -> dict:
    index = divergence_position - 1
    if index < 0 or index + 1 >= len(truth):
        return {"applicable": False, "top_1_recovered": None, "top_5_recovered": None}
    corrected_context = prefix + generated[:index] + [truth[index]]
    logits, _ = model(torch.tensor([corrected_context], dtype=torch.long))
    top5 = torch.topk(logits[0, -1], 5).indices.tolist()
    return {
        "applicable": True,
        "corrected_token_position": index + 1,
        "next_truth_id": truth[index + 1],
        "top_1_recovered": top5[0] == truth[index + 1],
        "top_5_recovered": truth[index + 1] in top5,
    }


def classify_error(item: dict, tokenizer: FoundationTokenizer) -> str:
    generated = item["generation"]
    ids = generated["ids"]
    if generated["eos_reached"] and len(ids) < 8:
        return "premature boundary"
    if "�" in generated["text"]:
        return "byte/token fragment"
    if any(ids[index:index + 4] == [ids[index]] * 4 for index in range(max(0, len(ids) - 3))):
        return "same-token repetition"
    boundary_ids = {tokenizer.eos_id, *tokenizer.encode("。", add_bos=False), *tokenizer.encode("、", add_bos=False)}
    if generated["loop_length"] == 1 and generated["loop_onset_token"] and ids[generated["loop_onset_token"] - 1] in boundary_ids:
        return "punctuation loop"
    particle_ids = {ids_[0] for text in PARTICLES if len(ids_ := tokenizer.encode(text, add_bos=False)) == 1}
    if generated["loop_length"] == 1 and generated["loop_onset_token"] and ids[generated["loop_onset_token"] - 1] in particle_ids:
        return "particle loop"
    if generated["loop_length"] is not None:
        return "short phrase repetition"
    overlap = sum(left == right for left, right in zip(ids, item["truth_ids"])) / max(1, min(len(ids), len(item["truth_ids"])))
    if overlap < 0.02:
        return "topic drift"
    if not generated["semantic_coherence_proxy"]:
        return "grammatical failure"
    if generated["runaway"]:
        return "runaway"
    if not generated["completion_proxy"]:
        return "semantic failure"
    return "other"


def aggregate_items(items: list[dict]) -> dict:
    count = len(items)
    taxonomy = Counter(item["error_class"] for item in items)
    horizons = {}
    for horizon in CONTINUATION_HORIZONS:
        overlaps = []
        exact = []
        for item in items:
            generated = item["generation"]["ids"][:horizon]
            truth = item["truth_ids"][:horizon]
            overlaps.append(sum(left == right for left, right in zip(generated, truth)) / horizon)
            exact.append(generated == truth)
        horizons[str(horizon)] = {
            "continuation_token_overlap": sum(overlaps) / count,
            "exact_continuation_rate": sum(exact) / count,
        }
    teacher = {}
    for horizon in TEACHER_HORIZONS:
        teacher[str(horizon)] = {
            metric: sum(item["teacher_forced"][str(horizon)][metric] for item in items) / count
            for metric in ("loss", "top_1_accuracy", "top_5_accuracy", "top_10_accuracy", "mean_correct_token_probability")
        }
    applicable_oracle = [item["oracle_recovery"] for item in items if item["oracle_recovery"]["applicable"]]
    loop_traces = [
        item["generation"]["trace"][item["generation"]["loop_onset_token"] - 1]
        for item in items
        if item["generation"]["loop_onset_token"] is not None
    ]
    return {
        "examples": count,
        "teacher_forced_horizon": teacher,
        "free_running": {
            "mean_divergence_position": sum(item["generation"]["divergence_position"] for item in items) / count,
            "first_4_token_exact": sum(item["generation"]["ids"][:4] == item["truth_ids"][:4] for item in items) / count,
            "first_8_token_exact": sum(item["generation"]["ids"][:8] == item["truth_ids"][:8] for item in items) / count,
            "character_validity": sum(item["generation"]["character_valid"] for item in items) / count,
            "japanese_character_ratio": sum(item["generation"]["japanese_character_ratio"] for item in items) / count,
            "natural_japanese_proxy": sum(item["generation"]["natural_japanese_proxy"] for item in items) / count,
            "semantic_local_syntax_proxy": sum(item["generation"]["semantic_coherence_proxy"] for item in items) / count,
            "sentence_boundary_rate": sum(item["generation"]["sentence_boundaries"] > 0 for item in items) / count,
            "runaway_rate": sum(item["generation"]["runaway"] for item in items) / count,
            "ngram_repetition": {
                str(n): sum(item["generation"]["ngram_repetition"][str(n)] for item in items) / count
                for n in range(1, 5)
            },
            "mean_maximum_repeated_span": sum(item["generation"]["maximum_repeated_span"] for item in items) / count,
            "mean_loop_onset": sum(item["generation"]["loop_onset_token"] for item in items if item["generation"]["loop_onset_token"] is not None) / max(1, sum(item["generation"]["loop_onset_token"] is not None for item in items)),
            "candidate_expected_top_5_rate": sum(step["expected_in_top_5"] for item in items for step in item["generation"]["trace"] if step["expected_in_top_5"] is not None) / max(1, sum(step["expected_in_top_5"] is not None for item in items for step in item["generation"]["trace"])),
            "candidate_expected_top_10_rate": sum(step["expected_in_top_10"] for item in items for step in item["generation"]["trace"] if step["expected_in_top_10"] is not None) / max(1, sum(step["expected_in_top_10"] is not None for item in items for step in item["generation"]["trace"])),
            "continuation_horizons": horizons,
        },
        "oracle_prefix_recovery": {
            "applicable": len(applicable_oracle),
            "top_1_recovery_rate": sum(row["top_1_recovered"] for row in applicable_oracle) / max(1, len(applicable_oracle)),
            "top_5_recovery_rate": sum(row["top_5_recovered"] for row in applicable_oracle) / max(1, len(applicable_oracle)),
        },
        "error_taxonomy": {
            key: {"count": taxonomy[key], "rate": taxonomy[key] / count}
            for key in (
                "punctuation loop", "same-token repetition", "short phrase repetition",
                "particle loop", "byte/token fragment", "topic drift", "grammatical failure",
                "semantic failure", "premature boundary", "runaway", "other",
            )
        },
        "loop_onset_confidence": {
            "examples": len(loop_traces),
            "mean_top_1_probability": sum(row["top_1_probability"] for row in loop_traces) / max(1, len(loop_traces)),
            "mean_top_2_probability": sum(row["top_2_probability"] for row in loop_traces) / max(1, len(loop_traces)),
            "mean_entropy": sum(row["entropy"] for row in loop_traces) / max(1, len(loop_traces)),
            "mean_margin": sum(row["top_1_top_2_margin"] for row in loop_traces) / max(1, len(loop_traces)),
        },
    }


def evaluate_examples(
    model: DiagnosticTransformerV17,
    tokenizer: FoundationTokenizer,
    examples: list[dict],
    seed: int,
) -> tuple[list[dict], dict]:
    rows = []
    started = time.perf_counter()
    for index, example in enumerate(examples):
        teacher = teacher_forced_metrics(model, example["prefix_ids"], example["truth_ids"])
        generation = traced_generate(
            model, tokenizer, example["prefix_ids"], example["truth_ids"],
            GREEDY, seed + index, 64,
        )
        row = {
            **example,
            "prefix_text": tokenizer.decode(example["prefix_ids"], skip_special=True),
            "truth_text": tokenizer.decode(example["truth_ids"], skip_special=True),
            "teacher_forced": teacher,
            "generation": generation,
        }
        row["oracle_recovery"] = oracle_recovery(
            model, example["prefix_ids"], example["truth_ids"],
            generation["ids"], generation["divergence_position"],
        )
        row["error_class"] = classify_error(row, tokenizer)
        rows.append(row)
        if (index + 1) % 25 == 0:
            print(json.dumps({"evaluated": index + 1, "total": len(examples)}), flush=True)
    metrics = aggregate_items(rows)
    metrics["wall_seconds"] = time.perf_counter() - started
    return rows, metrics


def token_counts(data: np.memmap, vocab_size: int) -> np.ndarray:
    result = np.zeros(vocab_size, dtype=np.int64)
    for start in range(0, len(data), 1_000_000):
        result += np.bincount(np.asarray(data[start:start + 1_000_000], dtype=np.int64), minlength=vocab_size)
    return result


def distribution_diagnostics(
    tokenizer: FoundationTokenizer,
    train_counts: np.ndarray,
    validation_counts: np.ndarray,
    items: list[dict],
) -> dict:
    generated_ids = [token for item in items for token in item["generation"]["ids"]]
    generated_counts = np.bincount(generated_ids, minlength=tokenizer.vocab_size).astype(np.float64)
    actual = validation_counts.astype(np.float64)
    generated = (generated_counts + 1e-9) / (generated_counts.sum() + 1e-9 * len(generated_counts))
    actual = (actual + 1e-9) / (actual.sum() + 1e-9 * len(actual))
    middle = 0.5 * (generated + actual)
    js = 0.5 * np.sum(generated * np.log(generated / middle)) + 0.5 * np.sum(actual * np.log(actual / middle))
    generated_rank = np.empty(len(generated), dtype=np.int64)
    actual_rank = np.empty(len(actual), dtype=np.int64)
    generated_rank[np.argsort(-generated)] = np.arange(len(generated))
    actual_rank[np.argsort(-actual)] = np.arange(len(actual))
    rank_correlation = float(np.corrcoef(generated_rank, actual_rank)[0, 1])
    train_rank = np.empty(len(train_counts), dtype=np.int64)
    train_rank[np.argsort(-train_counts)] = np.arange(len(train_counts))
    bucket_names = (
        ("top_1_percent", 0, 41),
        ("top_5_percent_excluding_top_1", 41, 205),
        ("top_20_percent_excluding_top_5", 205, 820),
        ("middle_20_to_80_percent", 820, 3277),
        ("rare_bottom_20_percent", 3277, len(train_counts)),
    )
    buckets = {
        name: sum(start <= train_rank[token] < end for token in generated_ids) / max(1, len(generated_ids))
        for name, start, end in bucket_names
    }
    return {
        "generated_tokens": len(generated_ids),
        "jensen_shannon_divergence_nats": float(js),
        "frequency_rank_correlation": rank_correlation,
        "generation_frequency_buckets": buckets,
        "generated_counts": generated_counts.astype(int).tolist(),
    }


@torch.inference_mode()
def boundary_diagnostics(
    model: DiagnosticTransformerV17,
    tokenizer: FoundationTokenizer,
    validation: np.memmap,
    validation_counts: np.ndarray,
    generated_counts: list[int],
) -> dict:
    ids_by_name = {
        "。": tokenizer.encode("。", add_bos=False),
        "！": tokenizer.encode("！", add_bos=False),
        "？": tokenizer.encode("？", add_bos=False),
        "newline": tokenizer.encode("\n", add_bos=False),
        "<EOS>": [tokenizer.eos_id],
    }
    probe = np.asarray(validation[:8193], dtype=np.int64).copy()
    logits, _ = model(torch.from_numpy(probe[:-1]).view(-1, 512))
    targets = torch.from_numpy(probe[1:]).view(-1, 512)
    probabilities = torch.softmax(logits.float(), dim=-1)
    top1 = probabilities.argmax(dim=-1)
    generated_total = sum(generated_counts)
    rows = {}
    for name, token_ids in ids_by_name.items():
        token_tensor = torch.tensor(token_ids)
        target_mask = (targets[..., None] == token_tensor).any(dim=-1)
        predicted_mask = (top1[..., None] == token_tensor).any(dim=-1)
        rows[name] = {
            "token_ids": token_ids,
            "actual_frequency": int(validation_counts[token_ids].sum()) / int(validation_counts.sum()),
            "mean_predicted_probability": float(probabilities[..., token_ids].sum(dim=-1).mean()),
            "top_1_prediction_rate": float(predicted_mask.float().mean()),
            "top_1_accuracy_when_actual": float((predicted_mask & target_mask).sum() / max(1, int(target_mask.sum()))),
            "generation_frequency": sum(generated_counts[token_id] for token_id in token_ids) / max(1, generated_total),
        }
    return rows


def decoding_comparison(
    model: DiagnosticTransformerV17,
    tokenizer: FoundationTokenizer,
    examples: list[dict],
    seed: int,
) -> dict:
    output = {}
    for mode_index, mode in enumerate(DECODING_MODES):
        rows = []
        for index, example in enumerate(examples[:50]):
            generation = traced_generate(
                model, tokenizer, example["prefix_ids"], example["truth_ids"],
                mode, seed + mode_index * 10_000 + index, 32,
            )
            rows.append(generation)
        output[mode["name"]] = {
            "settings": mode,
            "examples": len(rows),
            "character_validity": sum(row["character_valid"] for row in rows) / len(rows),
            "japanese_character_ratio": sum(row["japanese_character_ratio"] for row in rows) / len(rows),
            "natural_japanese_proxy": sum(row["natural_japanese_proxy"] for row in rows) / len(rows),
            "semantic_local_syntax_proxy": sum(row["semantic_coherence_proxy"] for row in rows) / len(rows),
            "sentence_boundary_rate": sum(row["sentence_boundaries"] > 0 for row in rows) / len(rows),
            "runaway_rate": sum(row["runaway"] for row in rows) / len(rows),
            "mean_repetition_3gram": sum(row["ngram_repetition"]["3"] for row in rows) / len(rows),
            "items": rows,
        }
    return output


def unconditional_generation(
    model: DiagnosticTransformerV17,
    tokenizer: FoundationTokenizer,
    seed: int,
) -> dict:
    result = {}
    for index, mode in enumerate(DECODING_MODES):
        row = traced_generate(model, tokenizer, [tokenizer.bos_id], [], mode, seed + index, 64)
        result[mode["name"]] = row
    return result


def exposure_counts(
    train: np.memmap,
    tokenizer: FoundationTokenizer,
    seeds: tuple[int, ...],
    tokens: int,
) -> list[dict]:
    updates = tokens // 512
    macro_count = (len(train) - 1) // 512
    rows = []
    for seed in seeds:
        permutation = torch.randperm(macro_count, generator=torch.Generator().manual_seed(seed))
        input_eos = target_eos = input_bos = target_bos = 0
        for index in permutation[:updates].tolist():
            start = index * 512
            values = np.asarray(train[start:start + 513], dtype=np.int64)
            input_eos += int(np.count_nonzero(values[:-1] == tokenizer.eos_id))
            target_eos += int(np.count_nonzero(values[1:] == tokenizer.eos_id))
            input_bos += int(np.count_nonzero(values[:-1] == tokenizer.bos_id))
            target_bos += int(np.count_nonzero(values[1:] == tokenizer.bos_id))
        rows.append({
            "seed": seed,
            "tokens_processed": tokens,
            "input_eos_observations": input_eos,
            "supervised_eos_targets": target_eos,
            "input_bos_observations": input_bos,
            "supervised_bos_targets": target_bos,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens", type=int,
        choices=(
            256_000, 512_000, 640_000, 768_000, 896_000, 1_024_000,
            1_280_000, 1_536_000, 1_792_000, 2_048_000,
        ),
        required=True,
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    settings = load_json("configs/unipilot-foundation-v22.json")
    tokenizer = FoundationTokenizer.load(ROOT / settings["tokenizer"])
    corpus = load_json(settings["corpus_manifest"])
    train_meta = corpus["splits"]["train"]
    validation_meta = corpus["splits"]["validation"]
    train = np.memmap(ROOT / train_meta["path"], dtype=np.uint16, mode="r")
    validation = np.memmap(ROOT / validation_meta["path"], dtype=np.uint16, mode="r")
    train_ranges = document_ranges(train, tokenizer.bos_id, tokenizer.eos_id)
    validation_ranges = document_ranges(validation, tokenizer.bos_id, tokenizer.eos_id)
    train_examples = sample_document_prefixes(train, train_ranges, "train", 13, 34_101)
    validation_examples = sample_document_prefixes(validation, validation_ranges, "validation", 50, 34_102)
    period_ids = tokenizer.encode("。", add_bos=False)
    if len(period_ids) != 1:
        raise RuntimeError("sentence-prefix boundary is not a single token")
    sentence_examples = sample_sentence_prefixes(validation, validation_ranges, period_ids[0], 50, 34_103)
    model, payload = load_model(args.tokens)
    val_rows, val_metrics = evaluate_examples(model, tokenizer, validation_examples, args.tokens + 1_000)
    train_rows, train_metrics = evaluate_examples(model, tokenizer, train_examples, args.tokens + 2_000)
    sentence_rows, sentence_metrics = evaluate_examples(model, tokenizer, sentence_examples, args.tokens + 3_000)
    train_counts = token_counts(train, tokenizer.vocab_size)
    validation_counts = token_counts(validation, tokenizer.vocab_size)
    distribution = distribution_diagnostics(tokenizer, train_counts, validation_counts, val_rows)
    boundaries = boundary_diagnostics(
        model, tokenizer, validation, validation_counts, distribution["generated_counts"]
    )
    decoding = decoding_comparison(model, tokenizer, [row for row in val_rows if row["prefix_length"] == 32], args.tokens + 4_000)
    unconditional = unconditional_generation(model, tokenizer, args.tokens + 5_000)
    instruction = []
    for index, (prompt, keywords) in enumerate(KNOWLEDGE_PROBES):
        prefix = tokenizer.encode(prompt, add_bos=True)
        generated = traced_generate(model, tokenizer, prefix, [], GREEDY, args.tokens + 6_000 + index, 64)
        instruction.append({"prompt": prompt, "expected_keywords": keywords, "generation": generated})
    human_readable = [
        {
            "id": row["id"],
            "prefix": row["prefix_text"],
            "reference": row["truth_text"],
            "generated": row["generation"]["text"],
            "natural_japanese_proxy": row["generation"]["natural_japanese_proxy"],
            "reasons": row["generation"]["evaluator_reasons"],
        }
        for row in val_rows[:50]
    ]
    exposure_seeds = (42,) if args.tokens == 640_000 else (42, 123, 2026)
    exposure = exposure_counts(train, tokenizer, exposure_seeds, args.tokens)
    corpus_boundaries = {
        split: {
            "tokens": int(meta["tokens"]),
            "documents_manifest": int(meta["documents"]),
            "bos_count": int((train_counts if split == "train" else validation_counts)[tokenizer.bos_id]),
            "eos_count": int((train_counts if split == "train" else validation_counts)[tokenizer.eos_id]),
            "tokens_per_document": int(meta["tokens"]) / int(meta["documents"]),
        }
        for split, meta in (("train", train_meta), ("validation", validation_meta))
    }
    result = {
        "schema": "foundation-v23-generation-dynamics-v1",
        "phase": 34,
        "tokens": args.tokens,
        "checkpoint": checkpoint_path(args.tokens).relative_to(ROOT).as_posix(),
        "checkpoint_sha256": file_sha256(checkpoint_path(args.tokens)),
        "seed": int(payload["seed"]),
        "evaluation_types": {
            "natural_document_prefix": "primary Base gate",
            "sentence_prefix": "primary Base gate",
            "instruction_like": "observational only",
        },
        "validation_document_prefix": {"metrics": val_metrics, "items": val_rows},
        "train_document_prefix": {"metrics": train_metrics, "items": train_rows},
        "validation_sentence_prefix": {"metrics": sentence_metrics, "items": sentence_rows},
        "instruction_like_observations": instruction,
        "decoding_comparison": decoding,
        "unconditional_generation": unconditional,
        "token_distribution": distribution,
        "boundary_diagnostics": boundaries,
        "corpus_boundary_counts": corpus_boundaries,
        "training_exposure": exposure,
        "corpus_exposure": {
            "processed_tokens": args.tokens,
            "train_tokens": int(train_meta["tokens"]),
            "fraction": args.tokens / int(train_meta["tokens"]),
            "percentage": 100 * args.tokens / int(train_meta["tokens"]),
            "epoch_equivalent": args.tokens / int(train_meta["tokens"]),
        },
        "natural_japanese_evaluator_audit": {
            "thresholds_changed": False,
            "examples": human_readable,
        },
        "repetition_penalty_used": False,
        "presence_penalty_used": False,
        "frequency_penalty_used": False,
        "knowledge_is_primary_gate": False,
        "architecture_changed": False,
        "corpus_changed": False,
        "tokenizer_changed": False,
        "final_blind_used": False,
    }
    output = ROOT / f"evaluation/foundation-v23-generation-diagnostics-{args.tokens}.json"
    # Per-step Top-10 traces are intentionally complete; compact encoding keeps
    # the research artifact practical without dropping any diagnostic field.
    output.write_text(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "tokens": args.tokens,
        "validation_examples": len(val_rows),
        "divergence": val_metrics["free_running"]["mean_divergence_position"],
        "teacher_top_10_h32": val_metrics["teacher_forced_horizon"]["32"]["top_10_accuracy"],
        "output": output.relative_to(ROOT).as_posix(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
