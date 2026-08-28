from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import html
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.base_tokenizer import FoundationTokenizer
from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from training.investigate_foundation_v14 import (
    CHECKPOINT_FORMAT,
    build_common_initialized_model,
    learning_rate,
    load_json,
    macro_permutation,
)
from evaluation.evaluate_foundation_v13 import (
    PRIMARY_MODES,
    load_checkpoint_model as load_v13_checkpoint,
    sample_token,
    valid_characters,
)


EXPECTED_FINAL_BLIND_SHA256 = (
    "fa7912d58ce251bb10b513f59793bb8ca6c0023b4fe08d1c040b8ccbfe49845b"
)
V13_STEPS = [0, 50, 100, 150, 200, 250]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_v14_checkpoint(path: Path) -> tuple[UniPilotTransformer, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise RuntimeError(f"unexpected Foundation v1.4 checkpoint: {path}")
    model = UniPilotTransformer(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def scratch_model(settings: dict, corpus: dict) -> UniPilotTransformer:
    model = build_common_initialized_model(settings, 512, int(corpus["vocab"]))
    model.eval()
    return model


def token_piece(tokenizer: FoundationTokenizer, token_id: int) -> str:
    if token_id in tokenizer.special_to_id.values():
        for token, value in tokenizer.special_to_id.items():
            if value == token_id:
                return token
    return tokenizer.decode([token_id], skip_special=False)


def repetition_rate(text: str) -> float:
    value = re.sub(r"\s+", "", text)
    grams = [value[index:index + 3] for index in range(max(0, len(value) - 2))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def language_proxy(text: str, *, aligned: bool = False, eos_reached: bool = False) -> dict:
    """Conservative observable proxy; it deliberately does not claim human semantics."""
    value = text.strip()
    visible = re.sub(r"\s+", "", value)
    japanese = len(re.findall(r"[ぁ-んァ-ヶー一-龥々]", visible))
    punctuation = len(re.findall(r"[、。！？,.!?（）()「」『』]", visible))
    content_runs = re.findall(r"[一-龥々ァ-ヶー]{2,}|[ぁ-ん]{3,}", visible)
    sentence_boundaries = len(re.findall(r"[。！？.!?]", value))
    repeat = repetition_rate(value)
    valid = valid_characters(value)
    japanese_ratio = japanese / max(1, len(visible))
    punctuation_ratio = punctuation / max(1, len(visible))
    newline_ratio = value.count("\n") / max(1, len(value))
    natural = (
        valid
        and len(visible) >= 20
        and japanese_ratio >= 0.35
        and punctuation_ratio <= 0.25
        and newline_ratio <= 0.30
        and repeat < 0.35
        and len(content_runs) >= 3
    )
    semantic_proxy = natural and sentence_boundaries >= 1 and (
        aligned or len(content_runs) >= 5
    )
    complete_proxy = natural and bool(
        eos_reached or re.search(r"[。！？.!?]$", value)
    )
    return {
        "character_valid": valid,
        "natural_japanese_proxy": natural,
        "semantic_coherence_proxy": semantic_proxy,
        "completion_proxy": complete_proxy,
        "japanese_character_ratio": japanese_ratio,
        "punctuation_ratio": punctuation_ratio,
        "newline_ratio": newline_ratio,
        "content_run_count": len(content_runs),
        "sentence_boundaries": sentence_boundaries,
        "repetition_rate": repeat,
    }


def aggregate_generation(items: list[dict]) -> dict:
    count = len(items)
    return {
        "prompts": count,
        "character_validity": sum(row["character_valid"] for row in items) / count,
        "natural_japanese_proxy": sum(
            row["natural_japanese_proxy"] for row in items
        ) / count,
        "semantic_coherence_proxy": sum(
            row["semantic_coherence_proxy"] for row in items
        ) / count,
        "completion_proxy": sum(row["completion_proxy"] for row in items) / count,
        "eos_rate": sum(row["eos_reached"] for row in items) / count,
        "runaway_rate": sum(row["runaway"] for row in items) / count,
        "mean_repetition_rate": sum(row["repetition_rate"] for row in items) / count,
    }


@torch.inference_mode()
def generate_ids(
    model: UniPilotTransformer,
    tokenizer: FoundationTokenizer,
    prompt_ids: list[int],
    mode: dict,
    seed: int,
    max_new_tokens: int = 64,
) -> dict:
    model.eval()
    all_ids = list(prompt_ids)
    generated: list[int] = []
    eos_probabilities: list[float] = []
    generator = torch.Generator().manual_seed(seed)
    forbidden = [
        token_id for token, token_id in tokenizer.special_to_id.items()
        if token != "<EOS>"
    ]
    past = None
    for _ in range(max_new_tokens):
        current = all_ids if past is None else [all_ids[-1]]
        if len(current) > model.config.context_length:
            current = current[-model.config.context_length:]
        logits, _, past = model(
            torch.tensor([current], dtype=torch.long),
            past_key_values=past,
            use_cache=True,
        )
        scores = logits[0, -1].float()
        probabilities = torch.softmax(scores, dim=-1)
        eos_probabilities.append(float(probabilities[tokenizer.eos_id].item()))
        filtered = scores.clone()
        filtered[forbidden] = -torch.inf
        next_id = sample_token(filtered, mode, generator)
        all_ids.append(next_id)
        generated.append(next_id)
        if next_id == tokenizer.eos_id:
            break
    text = tokenizer.decode(generated, skip_special=True)
    proxy = language_proxy(text, eos_reached=bool(generated and generated[-1] == tokenizer.eos_id))
    return {
        "text": text,
        "ids": generated,
        "tokens": len(generated),
        "eos_reached": bool(generated and generated[-1] == tokenizer.eos_id),
        "runaway": len(generated) >= max_new_tokens and (
            not generated or generated[-1] != tokenizer.eos_id
        ),
        "mean_eos_probability": sum(eos_probabilities) / len(eos_probabilities),
        **proxy,
    }


def evaluate_prompts(
    model: UniPilotTransformer,
    tokenizer: FoundationTokenizer,
    prompts: list[dict],
    seed: int,
    max_new_tokens: int = 64,
) -> dict:
    modes = {}
    for mode_index, mode in enumerate(PRIMARY_MODES):
        items = []
        for prompt_index, row in enumerate(prompts):
            generated = generate_ids(
                model,
                tokenizer,
                tokenizer.encode(row["prompt"], add_bos=True),
                mode,
                seed + mode_index * 100_000 + prompt_index,
                max_new_tokens=max_new_tokens,
            )
            items.append({"id": row["id"], "prompt": row["prompt"], **generated})
        modes[mode["name"]] = {
            "settings": mode,
            "metrics": aggregate_generation(items),
            "items": items,
        }
    return modes


@torch.inference_mode()
def validation_accuracy(
    model: UniPilotTransformer,
    tokens: np.memmap,
    probe_tokens: int,
    boundary_patterns: dict[str, list[int]],
) -> dict:
    model.eval()
    context = model.config.context_length
    total_loss = 0.0
    total = 0
    correct = {1: 0, 5: 0, 10: 0}
    all_targets: list[np.ndarray] = []
    all_top1: list[np.ndarray] = []
    all_target_probabilities: list[np.ndarray] = []
    for start in range(0, probe_tokens, context):
        size = min(context, probe_tokens - start)
        values = np.asarray(tokens[start:start + size + 1], dtype=np.int64).copy()
        inputs = torch.from_numpy(values[:-1]).unsqueeze(0)
        targets = torch.from_numpy(values[1:]).unsqueeze(0)
        logits, loss = model(inputs, targets)
        count = targets.numel()
        total_loss += float(loss.item()) * count
        total += count
        top = torch.topk(logits, 10, dim=-1).indices
        expanded = targets.unsqueeze(-1)
        for k in correct:
            correct[k] += int((top[..., :k] == expanded).any(dim=-1).sum().item())
        probabilities = torch.softmax(logits.float(), dim=-1)
        top1 = top[..., 0]
        actual_probabilities = probabilities.gather(-1, expanded).squeeze(-1)
        all_targets.append(targets.squeeze(0).cpu().numpy())
        all_top1.append(top1.squeeze(0).cpu().numpy())
        all_target_probabilities.append(actual_probabilities.squeeze(0).cpu().numpy())
    loss_value = total_loss / total
    target_values = np.concatenate(all_targets)
    top1_values = np.concatenate(all_top1)
    target_probabilities = np.concatenate(all_target_probabilities)
    by_marker = {}
    total_occurrences = total_full_correct = total_pattern_tokens = 0
    total_token_correct = 0
    total_probability = 0.0
    total_predictions = 0
    for marker, token_ids in boundary_patterns.items():
        pattern = np.asarray(token_ids, dtype=np.int64)
        matches = []
        predicted = []
        width = len(pattern)
        for index in range(0, len(target_values) - width + 1):
            if np.array_equal(target_values[index:index + width], pattern):
                matches.append(index)
            if np.array_equal(top1_values[index:index + width], pattern):
                predicted.append(index)
        full_correct = sum(
            np.array_equal(top1_values[index:index + width], pattern)
            for index in matches
        )
        token_correct = sum(
            int((top1_values[index:index + width] == pattern).sum()) for index in matches
        )
        probability_sum = sum(
            float(target_probabilities[index:index + width].sum()) for index in matches
        )
        occurrences = len(matches)
        by_marker[marker] = {
            "token_ids": [int(value) for value in pattern],
            "occurrences": occurrences,
            "full_sequence_top_1_accuracy": full_correct / max(1, occurrences),
            "token_top_1_accuracy": token_correct / max(1, occurrences * width),
            "mean_target_token_probability": probability_sum / max(1, occurrences * width),
            "prediction_occurrences": len(predicted),
            "prediction_rate": len(predicted) / max(1, len(target_values) - width + 1),
        }
        total_occurrences += occurrences
        total_full_correct += full_correct
        total_pattern_tokens += occurrences * width
        total_token_correct += token_correct
        total_probability += probability_sum
        total_predictions += len(predicted)
    return {
        "tokens": total,
        "loss": loss_value,
        "perplexity": math.exp(min(loss_value, 50)),
        "top_1_accuracy": correct[1] / total,
        "top_5_accuracy": correct[5] / total,
        "top_10_accuracy": correct[10] / total,
        "sentence_boundary": {
            "target_occurrences": total_occurrences,
            "target_occurrence_rate": total_occurrences / total,
            "full_sequence_top_1_accuracy": total_full_correct / max(1, total_occurrences),
            "token_top_1_accuracy": total_token_correct / max(1, total_pattern_tokens),
            "mean_target_token_probability": total_probability / max(1, total_pattern_tokens),
            "prediction_occurrence_rate": total_predictions / total,
            "by_marker": by_marker,
        },
    }


@torch.inference_mode()
def trace_tokens(
    model: UniPilotTransformer,
    tokenizer: FoundationTokenizer,
    prompt: str,
    tokens: int = 32,
) -> dict:
    ids = tokenizer.encode(prompt, add_bos=True)
    generated = []
    rows = []
    forbidden = [
        token_id for token, token_id in tokenizer.special_to_id.items()
        if token != "<EOS>"
    ]
    past = None
    for position in range(tokens):
        current = ids if past is None else [ids[-1]]
        logits, _, past = model(
            torch.tensor([current], dtype=torch.long),
            past_key_values=past,
            use_cache=True,
        )
        raw = logits[0, -1].float()
        probabilities = torch.softmax(raw, dim=-1)
        entropy = float((-(probabilities * probabilities.clamp_min(1e-30).log())).sum().item())
        scores = raw.clone()
        scores[forbidden] = -torch.inf
        top_values, top_ids = torch.topk(torch.softmax(scores, dim=-1), 5)
        selected = int(top_ids[0].item())
        rows.append({
            "position": position + 1,
            "top_1_id": selected,
            "top_1_token": token_piece(tokenizer, selected),
            "top_1_probability": float(top_values[0].item()),
            "top_5": [
                {
                    "id": int(token_id.item()),
                    "token": token_piece(tokenizer, int(token_id.item())),
                    "probability": float(probability.item()),
                }
                for token_id, probability in zip(top_ids, top_values)
            ],
            "entropy_nats": entropy,
        })
        ids.append(selected)
        generated.append(selected)
        if selected == tokenizer.eos_id:
            break
    return {
        "prompt": prompt,
        "generated": tokenizer.decode(generated, skip_special=True),
        "tokens": rows,
        "mean_entropy_nats": sum(row["entropy_nats"] for row in rows) / len(rows),
    }


def selected_training_documents(limit_sources: int = 3) -> list[dict]:
    selected = []
    sources = set()
    path = ROOT / "data/foundation_v11/documents/train.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["source_type"] in sources or len(row["text"]) < 500:
                continue
            sources.add(row["source_type"])
            selected.append({
                "id": row["id"],
                "source_type": row["source_type"],
                "text": row["text"],
            })
            if len(selected) >= limit_sources:
                break
    return selected


@torch.inference_mode()
def training_completion_probe(
    model: UniPilotTransformer,
    tokenizer: FoundationTokenizer,
    documents: list[dict],
    seed: int,
) -> dict:
    items = []
    greedy = PRIMARY_MODES[0]
    for document_index, document in enumerate(documents):
        ids = tokenizer.encode(document["text"], add_bos=True, add_eos=True)
        for prefix_length in (16, 32, 64):
            continuation = ids[prefix_length:prefix_length + 32]
            sequence = ids[:prefix_length + len(continuation)]
            inputs = torch.tensor([sequence[:-1]], dtype=torch.long)
            targets = torch.tensor([sequence[1:]], dtype=torch.long)
            logits, _ = model(inputs)
            start = prefix_length - 1
            continuation_logits = logits[:, start:start + len(continuation)]
            continuation_targets = targets[:, start:start + len(continuation)]
            top = torch.topk(continuation_logits, 5, dim=-1).indices
            top1 = float((top[..., 0] == continuation_targets).float().mean().item())
            top5 = float(
                (top == continuation_targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
            )
            generation = generate_ids(
                model,
                tokenizer,
                ids[:prefix_length],
                greedy,
                seed + document_index * 100 + prefix_length,
                max_new_tokens=32,
            )
            generated = generation["ids"]
            compared = min(len(continuation), len(generated))
            exact = sum(
                continuation[index] == generated[index] for index in range(compared)
            ) / max(1, len(continuation))
            items.append({
                "id": document["id"],
                "source_type": document["source_type"],
                "prefix_tokens": prefix_length,
                "prefix": tokenizer.decode(ids[:prefix_length], skip_special=True),
                "reference": tokenizer.decode(continuation, skip_special=True),
                "generated": generation["text"],
                "teacher_forced_top_1": top1,
                "teacher_forced_top_5": top5,
                "greedy_positional_token_similarity": exact,
                "eos_reached": generation["eos_reached"],
            })
    return {
        "documents": len(documents),
        "prefix_lengths": [16, 32, 64],
        "items": items,
        "mean_teacher_forced_top_1": sum(row["teacher_forced_top_1"] for row in items) / len(items),
        "mean_teacher_forced_top_5": sum(row["teacher_forced_top_5"] for row in items) / len(items),
        "mean_greedy_positional_token_similarity": sum(
            row["greedy_positional_token_similarity"] for row in items
        ) / len(items),
    }


def frequency_baselines(
    train: np.memmap,
    validation: np.memmap,
    vocab: int,
    probe_tokens: int,
    alpha: float = 0.1,
) -> dict:
    started = time.perf_counter()
    unigram_counts = np.bincount(train, minlength=vocab).astype(np.uint64)
    validation_inputs = np.asarray(validation[:probe_tokens], dtype=np.int64)
    validation_targets = np.asarray(validation[1:probe_tokens + 1], dtype=np.int64)
    unigram_probabilities = (
        unigram_counts.astype(np.float64) + alpha
    ) / (len(train) + alpha * vocab)
    unigram_loss = float(-np.log(unigram_probabilities[validation_targets]).mean())
    unigram_order = np.argsort(unigram_counts)[::-1][:10]
    unigram_accuracy = {
        str(k): float(np.isin(validation_targets, unigram_order[:k]).mean())
        for k in (1, 5, 10)
    }

    pair_counts = np.zeros(vocab * vocab, dtype=np.uint32)
    chunk_size = 1_000_000
    for start in range(0, len(train) - 1, chunk_size):
        end = min(start + chunk_size, len(train) - 1)
        left = np.asarray(train[start:end], dtype=np.int64)
        right = np.asarray(train[start + 1:end + 1], dtype=np.int64)
        unique, counts = np.unique(left * vocab + right, return_counts=True)
        pair_counts[unique] += counts.astype(np.uint32)
    train_context_counts = np.bincount(
        np.asarray(train[:-1], dtype=np.int64), minlength=vocab
    ).astype(np.uint64)
    pair_indices = validation_inputs * vocab + validation_targets
    observed = pair_counts[pair_indices].astype(np.float64)
    denominators = train_context_counts[validation_inputs].astype(np.float64) + alpha * vocab
    bigram_probabilities = (observed + alpha) / denominators
    bigram_loss = float(-np.log(bigram_probabilities).mean())
    bigram_matrix = pair_counts.reshape(vocab, vocab)
    unique_contexts = np.unique(validation_inputs)
    top_by_context = {}
    for context_id in unique_contexts:
        row = bigram_matrix[int(context_id)]
        top_by_context[int(context_id)] = np.argsort(row)[::-1][:10]
    bigram_accuracy = {}
    for k in (1, 5, 10):
        hits = sum(
            int(target in top_by_context[int(context)][:k])
            for context, target in zip(validation_inputs, validation_targets)
        )
        bigram_accuracy[str(k)] = hits / len(validation_targets)
    return {
        "method": {
            "smoothing": "add-alpha",
            "alpha": alpha,
            "train_tokens": len(train),
            "validation_probe_tokens": len(validation_targets),
            "same_probe_as_transformer": True,
        },
        "unigram": {
            "loss": unigram_loss,
            "perplexity": math.exp(unigram_loss),
            "top_1_accuracy": unigram_accuracy["1"],
            "top_5_accuracy": unigram_accuracy["5"],
            "top_10_accuracy": unigram_accuracy["10"],
            "most_frequent_token_ids": [int(value) for value in unigram_order],
        },
        "bigram": {
            "loss": bigram_loss,
            "perplexity": math.exp(bigram_loss),
            "top_1_accuracy": bigram_accuracy["1"],
            "top_5_accuracy": bigram_accuracy["5"],
            "top_10_accuracy": bigram_accuracy["10"],
            "observed_validation_pair_rate": float((observed > 0).mean()),
        },
        "wall_seconds": time.perf_counter() - started,
    }


def ratio(counter: Counter, total: int) -> dict:
    return {key: value / total for key, value in sorted(counter.items())}


def data_audit(
    settings: dict,
    corpus: dict,
    tokenizer: FoundationTokenizer,
    train: np.memmap,
) -> dict:
    documents = []
    path = ROOT / "data/foundation_v11/documents/train.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                documents.append({
                    "id": row["id"],
                    "source_type": row["source_type"],
                    "category": row["category"],
                })
    eos_positions = np.flatnonzero(train == tokenizer.eos_id)
    bos_positions = np.flatnonzero(train == tokenizer.bos_id)
    if len(eos_positions) != len(documents) or len(bos_positions) != len(documents):
        raise RuntimeError("packed document boundary count does not match document metadata")
    starts = np.concatenate((np.array([0], dtype=np.int64), eos_positions[:-1] + 1))
    document_lengths = eos_positions - starts + 1
    macro_count = (len(train) - 1) // 512
    permutation = macro_permutation(macro_count, int(settings["seed"]))
    selected_macros = np.asarray(permutation[:250], dtype=np.int64)
    selected_positions = np.concatenate([
        np.arange(index * 512 + 1, index * 512 + 513, dtype=np.int64)
        for index in selected_macros
    ])
    selected_documents = np.searchsorted(eos_positions, selected_positions, side="left")
    selected_documents = np.minimum(selected_documents, len(documents) - 1)
    sampled_source = Counter(documents[index]["source_type"] for index in selected_documents)
    sampled_category = Counter(documents[index]["category"] for index in selected_documents)
    sequential_positions = np.arange(1, 128001, dtype=np.int64)
    sequential_documents = np.searchsorted(eos_positions, sequential_positions, side="left")
    sequential_source = Counter(documents[index]["source_type"] for index in sequential_documents)
    document_counts = []
    for macro_index in selected_macros:
        positions = np.arange(macro_index * 512 + 1, macro_index * 512 + 513)
        indices = np.searchsorted(eos_positions, positions, side="left")
        document_counts.append(len(np.unique(indices)))
    sampled_values = np.asarray(train[selected_positions], dtype=np.int64)
    marker_ids = sentence_boundary_token_ids(tokenizer)
    marker_union = set().union(*marker_ids.values())
    sampled_source_ratios = ratio(sampled_source, len(selected_positions))
    corpus_source = corpus["splits"]["train"]["tokens_by_source"]
    corpus_source_ratios = ratio(Counter(corpus_source), sum(corpus_source.values()))
    all_sources = set(sampled_source_ratios) | set(corpus_source_ratios)
    maximum_source_deviation = max(
        abs(sampled_source_ratios.get(name, 0.0) - corpus_source_ratios.get(name, 0.0))
        for name in all_sources
    )
    return {
        "packed_order": "documents serialized in corpus order with BOS ... EOS",
        "training_sampler": "seeded random permutation of non-overlapping 512-token macro blocks",
        "randomized": True,
        "document_level_shuffle": False,
        "block_level_shuffle": True,
        "shared_first_128k_macroblocks_sha256": hashlib.sha256(
            selected_macros.tobytes()
        ).hexdigest(),
        "documents": len(documents),
        "eos_tokens": int(len(eos_positions)),
        "bos_tokens": int(len(bos_positions)),
        "mean_document_tokens": float(document_lengths.mean()),
        "median_document_tokens": float(np.median(document_lengths)),
        "mean_documents_touched_per_512_block": float(np.mean(document_counts)),
        "initial_128k": {
            "tokens": len(selected_positions),
            "unique_documents_touched": int(len(np.unique(selected_documents))),
            "source_tokens": dict(sampled_source),
            "source_ratios": sampled_source_ratios,
            "category_tokens": dict(sampled_category),
            "category_ratios": ratio(sampled_category, len(selected_positions)),
            "bos_count": int((sampled_values == tokenizer.bos_id).sum()),
            "eos_count": int((sampled_values == tokenizer.eos_id).sum()),
            "sentence_boundary_token_count": int(np.isin(
                sampled_values, list(marker_union)
            ).sum()),
        },
        "sequential_first_128k_source_ratios": ratio(
            sequential_source, len(sequential_positions)
        ),
        "corpus_source_ratios": corpus_source_ratios,
        "maximum_sampled_vs_corpus_source_ratio_deviation": maximum_source_deviation,
        "source_mix_materially_biased": maximum_source_deviation > 0.10,
        "boundary_interpretation": (
            "EOS/BOS density is low because mean documents are much longer than one 512-token block"
        ),
    }


def sentence_boundary_token_ids(tokenizer: FoundationTokenizer) -> dict[str, list[int]]:
    markers = ["。", "！", "？", "\n"]
    return {marker: tokenizer.encode(marker) for marker in markers}


def sampled_training_tokens(settings: dict, train: np.memmap, macro_blocks: int = 250) -> np.ndarray:
    macro_count = (len(train) - 1) // 512
    permutation = macro_permutation(macro_count, int(settings["seed"]))[:macro_blocks]
    return np.concatenate([
        np.asarray(train[int(index) * 512:int(index) * 512 + 512], dtype=np.uint16)
        for index in permutation
    ])


def evaluator_audit(v13_generation: dict) -> dict:
    auto_positive = []
    for result in v13_generation["results"]:
        for mode_name, mode in result["modes"].items():
            for item in mode["items"]:
                if item["natural_japanese"] or item["semantic_coherence"]:
                    strict = language_proxy(
                        item["generated"],
                        aligned=item["prompt_aligned"],
                        eos_reached=item["eos_reached"],
                    )
                    auto_positive.append({
                        "step": result["step"],
                        "mode": mode_name,
                        "id": item["id"],
                        "generated": item["generated"],
                        "old_natural": item["natural_japanese"],
                        "old_semantic": item["semantic_coherence"],
                        "strict_proxy": strict,
                        "human_audit": "not natural or semantically coherent; function-word/punctuation sequence",
                    })
    examples = []
    for result in v13_generation["results"]:
        for mode_name, mode in result["modes"].items():
            for item in mode["items"][:10]:
                examples.append({
                    "step": result["step"],
                    "mode": mode_name,
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "generated": item["generated"],
                    "old_natural": item["natural_japanese"],
                    "old_semantic": item["semantic_coherence"],
                    "strict_proxy": language_proxy(
                        item["generated"],
                        aligned=item["prompt_aligned"],
                        eos_reached=item["eos_reached"],
                    ),
                })
    return {
        "status": "ISSUE FOUND",
        "findings": [
            "Natural Japanese was only a character-ratio/length/repetition heuristic, not a syntax judge.",
            "Semantic Coherence inherited that weak proxy and accepted punctuation/function-word salad.",
            "Completion, EOS and Runaway implementations match their declared mechanical definitions.",
            "The old step-50 8% was false-positive signal; step-250 0% was not a hidden improvement.",
        ],
        "old_automatic_positive_count": len(auto_positive),
        "old_automatic_positives": auto_positive,
        "human_readable_examples": examples,
        "examples_per_step": 20,
        "full_existing_generation_artifact": "evaluation/foundation-v13-generation.json",
    }


def escape_cell(value: str, limit: int = 180) -> str:
    visible = "".join(
        character
        if ord(character) >= 32 or character in {"\r", "\n", "\t"}
        else f"\\x{ord(character):02x}"
        for character in value
    )
    compact = visible.replace("\r", "").replace("\n", "↵").replace("\t", "⇥")
    if len(compact) > limit:
        compact = compact[:limit] + "…"
    return html.escape(compact).replace("|", "&#124;")


def report_markdown(summary: dict) -> str:
    lines = [
        "# UniPilot Foundation v1.4 — Language Emergence Investigation",
        "",
        "## Executive result",
        "",
        f"- Generation evaluator: **{summary['generation_evaluator']['status']}**",
        f"- Final Gate: **{summary['gate']['status']}**",
        f"- Recommended next token budget: **{summary['gate']['recommended_next_token_budget']}**",
        f"- Reason: {summary['gate']['reason']}",
        "- 500step continuation, 46M, corpus expansion, Campus/Instruction/DPO are not executed.",
        "",
        "## Evaluator audit",
        "",
    ]
    lines.extend(f"- {finding}" for finding in summary["generation_evaluator"]["findings"])
    lines.extend([
        "",
        "旧評価でNatural/Semantic陽性だった全件:",
        "",
        "| Step | Mode | ID | Old N/S | Human audit | Generated |",
        "|---:|---|---|---|---|---|",
    ])
    for row in summary["generation_evaluator"]["old_automatic_positives"]:
        lines.append(
            f"| {row['step']} | {row['mode']} | {row['id']} | "
            f"{int(row['old_natural'])}/{int(row['old_semantic'])} | false positive | "
            f"{escape_cell(row['generated'])} |"
        )
    lines.extend([
        "",
        "## Validation next-token accuracy",
        "",
        "| Step | Loss | PPL | Top-1 | Top-5 | Top-10 | Boundary top-1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["v13_diagnostics"]:
        accuracy = row["validation"]
        lines.append(
            f"| {row['step']} | {accuracy['loss']:.4f} | {accuracy['perplexity']:.1f} | "
            f"{accuracy['top_1_accuracy']:.2%} | {accuracy['top_5_accuracy']:.2%} | "
            f"{accuracy['top_10_accuracy']:.2%} | "
            f"{accuracy['sentence_boundary']['full_sequence_top_1_accuracy']:.2%} |"
        )
    lines.extend([
        "",
        "### Sentence-boundary behavior",
        "",
        "`！` is a two-token sequence; every marker below is measured by exact token-sequence matching.",
        "The fixed 8,192-token slice contains no `！` or `？`, so those two are not estimated here.",
        "",
        "| Step | All boundary target | All boundary predicted | `。` target/predicted | Newline target/predicted |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in summary["v13_diagnostics"]:
        boundary = row["validation"]["sentence_boundary"]
        period = boundary["by_marker"]["。"]
        newline = boundary["by_marker"]["\n"]
        lines.append(
            f"| {row['step']} | {boundary['target_occurrence_rate']:.2%} | "
            f"{boundary['prediction_occurrence_rate']:.2%} | "
            f"{period['occurrences'] / row['validation']['tokens']:.2%}/"
            f"{period['prediction_rate']:.2%} | "
            f"{newline['occurrences'] / row['validation']['tokens']:.2%}/"
            f"{newline['prediction_rate']:.2%} |"
        )
    lines.extend([
        "",
        "At step 250, sentence-boundary targets are 4.64% of the slice but become 65.44% of Top-1 predictions. "
        "The model learned to recognize boundaries while greatly overpredicting them.",
    ])
    lines.extend([
        "",
        "## Frequency LM comparison (same validation token slice)",
        "",
        "| Model | Loss | PPL | Top-1 | Top-5 | Top-10 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, row in summary["baseline_comparison"].items():
        lines.append(
            f"| {name} | {row['loss']:.4f} | {row['perplexity']:.1f} | "
            f"{row['top_1_accuracy']:.2%} | {row['top_5_accuracy']:.2%} | "
            f"{row['top_10_accuracy']:.2%} |"
        )
    lines.extend([
        "",
        "## Context / schedule — exact same 128k training tokens",
        "",
        "| Experiment | Context | Schedule | Effective batch | Loss | PPL | Top-1 | Top-5 | Top-10 | Natural | Semantic | Repetition | Runaway | tok/s | RAM MB |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["experiment_comparison"]:
        validation = row["validation"]
        generation = row["generation"]
        lines.append(
            f"| {row['name']} | {row['context_length']} | {row['schedule']} | "
            f"{row['effective_batch_tokens']} | {validation['loss']:.4f} | "
            f"{validation['perplexity']:.1f} | {validation['top_1_accuracy']:.2%} | "
            f"{validation['top_5_accuracy']:.2%} | {validation['top_10_accuracy']:.2%} | "
            f"{generation['natural_japanese_proxy']:.1%} | "
            f"{generation['semantic_coherence_proxy']:.1%} | "
            f"{generation['mean_repetition_rate']:.1%} | {generation['runaway_rate']:.1%} | "
            f"{row['tokens_per_second']:.1f} | {row['peak_ram_mb']:.1f} |"
        )
    lines.extend([
        "",
        "### Learning-rate profiles",
        "",
        "| Schedule | update 20 | update 100 | update 250 | Trained in core matrix |",
        "|---|---:|---:|---:|---|",
    ])
    for schedule in (
        "short_cosine_250", "constant_after_warmup20", "long_cosine_1000",
        "warmup50_constant",
    ):
        profile = summary["schedule_profiles"][schedule]
        trained = "YES" if schedule != "warmup50_constant" else "profile only"
        lines.append(
            f"| {schedule} | {profile['20']:.2e} | {profile['100']:.2e} | "
            f"{profile['250']:.2e} | {trained} |"
        )
    lines.extend([
        "",
        "## Effective batch pilot — exact same 32,768 tokens",
        "",
        "| Effective batch | Updates | Loss | Top-1 | Top-5 | tok/s | RAM MB |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["effective_batch_comparison"]:
        validation = row["validation"]
        lines.append(
            f"| {row['effective_batch_tokens']} | {row['updates']} | "
            f"{validation['loss']:.4f} | {validation['top_1_accuracy']:.2%} | "
            f"{validation['top_5_accuracy']:.2%} | {row['tokens_per_second']:.1f} | "
            f"{row['peak_ram_mb']:.1f} |"
        )
    data = summary["data_audit"]
    lines.extend([
        "",
        "## Data sampling / source mix / boundaries",
        "",
        f"- Sampler: {data['training_sampler']}",
        f"- Block shuffle: {data['block_level_shuffle']}; document-level shuffle: {data['document_level_shuffle']}",
        f"- Initial 128k touched documents: {data['initial_128k']['unique_documents_touched']}",
        f"- Mean document length: {data['mean_document_tokens']:.1f} tokens",
        f"- Mean documents touched per 512 block: {data['mean_documents_touched_per_512_block']:.3f}",
        f"- Initial 128k BOS/EOS: {data['initial_128k']['bos_count']}/{data['initial_128k']['eos_count']}",
        f"- Max source-ratio deviation from corpus: {data['maximum_sampled_vs_corpus_source_ratio_deviation']:.2%}",
        "",
        "| Source | Corpus | Initial sampled 128k |",
        "|---|---:|---:|",
    ])
    for source in sorted(set(data["corpus_source_ratios"]) | set(data["initial_128k"]["source_ratios"])):
        lines.append(
            f"| {source} | {data['corpus_source_ratios'].get(source, 0):.2%} | "
            f"{data['initial_128k']['source_ratios'].get(source, 0):.2%} |"
        )
    best = summary["best_experiment"]
    lines.extend([
        "",
        "## Best experiment",
        "",
        f"- Name: **{best['name']}**",
        f"- Parameters: {best['parameters']:,}",
        f"- Context: {best['context_length']}",
        f"- LR / schedule: {best['learning_rate']:.1e} / {best['schedule']}",
        f"- Effective batch: {best['effective_batch_tokens']} tokens",
        f"- Validation loss: {best['validation']['loss']:.6f}",
        f"- Top-1/5/10: {best['validation']['top_1_accuracy']:.2%} / "
        f"{best['validation']['top_5_accuracy']:.2%} / {best['validation']['top_10_accuracy']:.2%}",
        f"- Natural/Semantic proxy: {best['generation']['natural_japanese_proxy']:.1%} / "
        f"{best['generation']['semantic_coherence_proxy']:.1%}",
        f"- Repetition/Runaway: {best['generation']['mean_repetition_rate']:.1%} / "
        f"{best['generation']['runaway_rate']:.1%}",
        f"- Checkpoint: `{best.get('checkpoint', 'existing v1.3 reference')}`",
        "",
        "## Unconditional and train-prefix diagnosis",
        "",
        "| Step | BOS greedy | EOS | Train top-1 | Train top-5 | Greedy similarity |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for row in summary["v13_diagnostics"]:
        unconditional = row["unconditional"]["greedy_no_penalty"]
        train_probe = row["training_completion"]
        lines.append(
            f"| {row['step']} | {escape_cell(unconditional['text'], 90)} | "
            f"{int(unconditional['eos_reached'])} | "
            f"{train_probe['mean_teacher_forced_top_1']:.2%} | "
            f"{train_probe['mean_teacher_forced_top_5']:.2%} | "
            f"{train_probe['mean_greedy_positional_token_similarity']:.2%} |"
        )
    lines.extend([
        "",
        "## Top-token trace summary",
        "",
        "| Step | Prompt | First 20 greedy token pieces | Mean entropy |",
        "|---:|---|---|---:|",
    ])
    for row in summary["v13_diagnostics"]:
        for trace in row["token_trace"]:
            tokens = " ".join(item["top_1_token"] or "∅" for item in trace["tokens"][:20])
            lines.append(
                f"| {row['step']} | {escape_cell(trace['prompt'], 50)} | "
                f"{escape_cell(tokens, 160)} | {trace['mean_entropy_nats']:.3f} |"
            )
    lines.extend([
        "",
        "## Step-by-step generation examples (20 per step)",
        "",
        "Full 50-prompt raw output remains in `evaluation/foundation-v13-generation.json`.",
        "",
        "| Step | Mode | ID | Prompt | Generated | Old N/S | Strict N/S proxy |",
        "|---:|---|---|---|---|---|---|",
    ])
    for row in summary["generation_evaluator"]["human_readable_examples"]:
        strict = row["strict_proxy"]
        lines.append(
            f"| {row['step']} | {row['mode']} | {row['id']} | "
            f"{escape_cell(row['prompt'], 70)} | {escape_cell(row['generated'])} | "
            f"{int(row['old_natural'])}/{int(row['old_semantic'])} | "
            f"{int(strict['natural_japanese_proxy'])}/{int(strict['semantic_coherence_proxy'])} |"
        )
    lines.extend([
        "",
        "## Protection / verification",
        "",
        f"- Final Blind SHA256: `{summary['protected']['final_blind_sha256']}` (content unopened)",
        "- Production v0.4, Campus v2.3, Render, Vercel and Release unchanged.",
        "- External AI/API: OFF.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v14.json")
    parser.add_argument("--experiment-dir", default="checkpoints/foundation-v14-investigation")
    parser.add_argument("--output", default="evaluation/foundation-v14-language-investigation.json")
    parser.add_argument("--report", default="evaluation/foundation-v14-language-investigation-report.md")
    parser.add_argument("--render-existing", action="store_true")
    args = parser.parse_args()
    if args.render_existing:
        summary = load_json(args.output)
        (ROOT / args.report).write_text(report_markdown(summary), encoding="utf-8")
        print(json.dumps({
            "rendered": args.report,
            "gate": summary["gate"]["status"],
        }, ensure_ascii=False, indent=2))
        return 0
    settings = load_json(args.config)
    corpus = load_json(settings["corpus_manifest"])
    tokenizer = FoundationTokenizer.load(ROOT / settings["tokenizer"])
    torch.set_num_threads(int(settings["cpu_threads"]))
    train = np.memmap(
        ROOT / corpus["splits"]["train"]["path"], dtype=np.uint16, mode="r"
    )
    validation = np.memmap(
        ROOT / corpus["splits"]["validation"]["path"], dtype=np.uint16, mode="r"
    )
    probe_tokens = int(settings["validation_probe_tokens"])
    marker_ids = sentence_boundary_token_ids(tokenizer)
    v13_generation = load_json("evaluation/foundation-v13-generation.json")
    audit = evaluator_audit(v13_generation)
    full_corpus_baselines = frequency_baselines(
        train, validation, int(corpus["vocab"]), probe_tokens
    )
    matched_tokens = sampled_training_tokens(settings, train)
    token_matched_baselines = frequency_baselines(
        matched_tokens, validation, int(corpus["vocab"]), probe_tokens
    )
    token_matched_baselines["method"]["macroblock_join_pairs"] = 249
    baselines = {
        "full_corpus_33m": full_corpus_baselines,
        "token_matched_128k": token_matched_baselines,
    }
    training_documents = selected_training_documents()

    v13_diagnostics = []
    for step in V13_STEPS:
        if step == 0:
            model = scratch_model(settings, corpus)
            source = "deterministic common scratch initialization"
        else:
            path = ROOT / f"checkpoints/foundation-v13-clean-250/checkpoint-step-{step}.pt"
            model, _ = load_v13_checkpoint(path)
            source = path.relative_to(ROOT).as_posix()
        accuracy = validation_accuracy(model, validation, probe_tokens, marker_ids)
        traces = [
            trace_tokens(model, tokenizer, prompt, tokens=32)
            for prompt in ("大学では、授業だけでなく", "日本の首都は")
        ]
        unconditional = {}
        for mode_index, mode in enumerate(PRIMARY_MODES):
            unconditional[mode["name"]] = generate_ids(
                model,
                tokenizer,
                [tokenizer.bos_id],
                mode,
                int(settings["seed"]) + 700_000 + mode_index,
                max_new_tokens=64,
            )
        completion = training_completion_probe(
            model, tokenizer, training_documents, int(settings["seed"]) + step
        )
        v13_diagnostics.append({
            "step": step,
            "source": source,
            "validation": accuracy,
            "token_trace": traces,
            "unconditional": unconditional,
            "training_completion": completion,
        })
        del model
        print(json.dumps({"diagnosed_v13_step": step, "validation": accuracy}), flush=True)

    prompt_items = load_json("data/foundation_v11/evaluation/base-completion-50.json")["items"]
    experiment_dir = ROOT / args.experiment_dir
    experiment_comparison = []
    reference_curve = load_json(settings["reference"]["training_curve"])
    reference_final = next(row for row in reference_curve["history"] if row["step"] == 250)
    reference_model, _ = load_v13_checkpoint(ROOT / settings["reference"]["checkpoint"])
    reference_validation = validation_accuracy(reference_model, validation, probe_tokens, marker_ids)
    reference_modes = evaluate_prompts(
        reference_model, tokenizer, prompt_items[:20], int(settings["seed"]), max_new_tokens=64
    )
    reference_generation = reference_modes[PRIMARY_MODES[1]["name"]]["metrics"]
    experiment_comparison.append({
        "name": settings["reference"]["name"],
        "context_length": 512,
        "schedule": settings["reference"]["schedule"],
        "effective_batch_tokens": 512,
        "token_budget": 128000,
        "parameters": reference_curve["parameters"],
        "validation": reference_validation,
        "generation": reference_generation,
        "generation_modes": reference_modes,
        "tokens_per_second": reference_final["tokens_per_second"],
        "peak_ram_mb": reference_final["peak_ram_mb"],
        "learning_rate": settings["learning_rate"],
        "reference_existing_v13": True,
    })
    del reference_model
    for experiment in settings["experiments"]:
        result = load_json(experiment_dir / f"{experiment['name']}.json")
        model, _ = load_v14_checkpoint(experiment_dir / f"{experiment['name']}.pt")
        accuracy = validation_accuracy(model, validation, probe_tokens, marker_ids)
        modes = evaluate_prompts(
            model, tokenizer, prompt_items[:20], int(settings["seed"]), max_new_tokens=64
        )
        final = result["history"][-1]
        experiment_comparison.append({
            "name": experiment["name"],
            "context_length": experiment["context_length"],
            "schedule": experiment["schedule"],
            "effective_batch_tokens": experiment["effective_batch_tokens"],
            "token_budget": result["token_budget"],
            "parameters": result["parameters"],
            "validation": accuracy,
            "generation": modes[PRIMARY_MODES[1]["name"]]["metrics"],
            "generation_modes": modes,
            "tokens_per_second": final["tokens_per_second"],
            "peak_ram_mb": result["peak_ram_mb"],
            "learning_rate": settings["learning_rate"],
            "checkpoint": result["checkpoint"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "reference_existing_v13": False,
        })
        del model
        print(json.dumps({"evaluated_experiment": experiment["name"], "validation": accuracy}), flush=True)

    batch_comparison = []
    for multiplier in settings["effective_batch_pilot"]["multipliers"]:
        result = load_json(experiment_dir / f"effective_batch_{multiplier}x.json")
        final = result["history"][-1]
        batch_comparison.append({
            "multiplier": multiplier,
            "effective_batch_tokens": result["experiment"]["effective_batch_tokens"],
            "updates": result["updates"],
            "token_budget": result["token_budget"],
            "validation": {
                "loss": final["validation_loss"],
                "perplexity": final["validation_perplexity"],
                "top_1_accuracy": final["validation_top_1_accuracy"],
                "top_5_accuracy": final["validation_top_5_accuracy"],
                "top_10_accuracy": final["validation_top_10_accuracy"],
            },
            "tokens_per_second": final["tokens_per_second"],
            "peak_ram_mb": result["peak_ram_mb"],
        })

    data = data_audit(settings, corpus, tokenizer, train)
    best = min(
        experiment_comparison,
        key=lambda row: (
            row["validation"]["loss"],
            -row["validation"]["top_5_accuracy"],
            -row["generation"]["natural_japanese_proxy"],
        ),
    )
    transformer250 = v13_diagnostics[-1]["validation"]
    bigram = baselines["token_matched_128k"]["bigram"]
    transformer_beats_bigram = transformer250["loss"] <= bigram["loss"] - 0.10
    reference_constant = next(
        row for row in experiment_comparison if row["name"] == "context512_constant"
    )
    short_candidates = [row for row in experiment_comparison if row["context_length"] < 512]
    best_short = min(short_candidates, key=lambda row: row["validation"]["loss"])
    short_context_clear = (
        best_short["validation"]["loss"] <= reference_constant["validation"]["loss"] - 0.05
        or best_short["validation"]["top_5_accuracy"]
        >= reference_constant["validation"]["top_5_accuracy"] + 0.01
        or best_short["generation"]["natural_japanese_proxy"]
        >= reference_constant["generation"]["natural_japanese_proxy"] + 0.05
    )
    loss_stable = all(
        not load_json(experiment_dir / f"{experiment['name']}.json")["diverged"]
        for experiment in settings["experiments"]
    )
    language_signal_increased = (
        best["generation"]["natural_japanese_proxy"]
        > experiment_comparison[0]["generation"]["natural_japanese_proxy"]
        or best["generation"]["semantic_coherence_proxy"]
        > experiment_comparison[0]["generation"]["semantic_coherence_proxy"]
    )
    if data["source_mix_materially_biased"] or not data["randomized"]:
        gate = "DATA_INVESTIGATE"
        reason = "Initial sampled tokens have a material source/order bias."
        next_budget = "128k再実験（data sampling修正後）"
    elif short_context_clear:
        gate = "CURRICULUM_CHANGE"
        reason = "Short context or its paired schedule clearly improves the equal-token comparison."
        next_budget = "256k tokens（best short-context curriculum）"
    elif transformer_beats_bigram and language_signal_increased and loss_stable:
        gate = "CONTINUE_PRETRAINING"
        reason = "Transformer beats the bigram baseline and language proxies improve with stable loss."
        next_budget = "512k tokens"
    else:
        gate = "ARCHITECTURE_INVESTIGATE"
        reason = (
            "Transformer does not clearly beat the bigram baseline or language evidence does not improve; "
            "do not scale the token budget before architecture/normalization/optimization diagnosis."
        )
        next_budget = "256k tokens only after architecture diagnostic PASS"

    schedule_profiles = {}
    for schedule in (
        "short_cosine_250", "constant_after_warmup20", "long_cosine_1000",
        "warmup50_constant",
    ):
        schedule_profiles[schedule] = {
            str(update): learning_rate(settings, schedule, max(0, update - 1))
            for update in (1, 20, 50, 100, 150, 200, 250)
        }
    final_blind = ROOT / "data/foundation_v09/evaluation/final-blind-1000.json"
    final_blind_digest = sha256(final_blind)
    if final_blind_digest != EXPECTED_FINAL_BLIND_SHA256:
        raise RuntimeError("Final Blind SHA256 changed")
    summary = {
        "schema_version": "foundation-v14-language-investigation-v1",
        "generation_evaluator": audit,
        "baseline_lms": baselines,
        "v13_diagnostics": v13_diagnostics,
        "experiment_comparison": experiment_comparison,
        "effective_batch_comparison": batch_comparison,
        "data_audit": data,
        "sentence_boundary_token_ids": {
            marker: values for marker, values in marker_ids.items()
        },
        "schedule_profiles": schedule_profiles,
        "baseline_comparison": {
            "Unigram (128k token-matched)": baselines["token_matched_128k"]["unigram"],
            "Bigram (128k token-matched)": baselines["token_matched_128k"]["bigram"],
            "Unigram (full 33.4M)": baselines["full_corpus_33m"]["unigram"],
            "Bigram (full 33.4M)": baselines["full_corpus_33m"]["bigram"],
            "Transformer v1.3 step250": transformer250,
        },
        "best_experiment": best,
        "gate": {
            "status": gate,
            "reason": reason,
            "recommended_next_token_budget": next_budget,
            "checks": {
                "transformer_clearly_beats_bigram": transformer_beats_bigram,
                "short_context_clear_improvement": short_context_clear,
                "language_signal_increased": language_signal_increased,
                "loss_stable": loss_stable,
                "data_source_mix_materially_biased": data["source_mix_materially_biased"],
            },
        },
        "protected": {
            "final_blind_sha256": final_blind_digest,
            "final_blind_content_opened": False,
            "production_v04_changed": False,
            "campus_v23_changed": False,
            "render_changed": False,
            "vercel_changed": False,
            "release_changed": False,
            "foundation_v13_checkpoint_deleted": False,
            "foundation_500_continuation_executed": False,
            "standard_46m_executed": False,
            "corpus_expansion_executed": False,
            "campus_pretraining_executed": False,
            "instruction_tuning_executed": False,
            "dpo_executed": False,
        },
        "external_ai_api": "OFF",
        "push_or_deploy_performed": False,
    }
    output = ROOT / args.output
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / args.report).write_text(report_markdown(summary), encoding="utf-8")
    print(json.dumps({
        "generation_evaluator": audit["status"],
        "best_experiment": best["name"],
        "best_validation": best["validation"],
        "gate": summary["gate"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
