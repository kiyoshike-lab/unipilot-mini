from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v15 import DiagnosticTransformer
from evaluation.audit_foundation_v15_architecture import context_sensitivity


def stats(values: torch.Tensor) -> dict:
    values = values.detach().float()
    finite = values[torch.isfinite(values)]
    return {
        "finite": finite.numel() == values.numel(),
        "mean": float(finite.mean()),
        "std": float(finite.std(unbiased=False)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "rms": float(finite.square().mean().sqrt()),
    }


@torch.inference_mode()
def detailed_probe(model: DiagnosticTransformer, token_ids: torch.Tensor) -> dict:
    model.eval()
    length = token_ids.size(1)
    positions = torch.arange(length)
    token = model.embeddings.token(token_ids)
    scaled_token = token * model.embeddings.token_scale
    position = model.embeddings.position(positions)[None, :, :]
    hidden = scaled_token + position
    layers = []
    for index, block in enumerate(model.blocks):
        layer_input = hidden
        attention_output, _ = block.attention(block.norm1(layer_input))
        residual = layer_input + attention_output
        mlp_output = block.feed_forward(block.norm2(residual))
        hidden = residual + mlp_output
        layers.append({
            "layer": index,
            "input": stats(layer_input),
            "attention": stats(attention_output),
            "residual": stats(residual),
            "mlp": stats(mlp_output),
            "output": stats(hidden),
        })
    final_hidden = model.final_norm(hidden)
    logits = model.output(final_hidden)
    probabilities = torch.softmax(logits, dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    top5 = probabilities.topk(5, dim=-1).values
    direct_logits, _ = model(token_ids)
    return {
        "embedding": {
            "formula": (
                "token_embedding * sqrt(d_model) + learned_position_embedding"
                if model.config.scale_token_embedding
                else "token_embedding + learned_position_embedding"
            ),
            "d_model": model.config.embedding_dim,
            "sqrt_d_model": math.sqrt(model.config.embedding_dim),
            "configured_token_scale": model.embeddings.token_scale,
            "raw_token": stats(token),
            "scaled_token": stats(scaled_token),
            "position": stats(position),
            "combined": stats(scaled_token + position),
            "scaled_to_position_rms_ratio": (
                stats(scaled_token)["rms"] / max(stats(position)["rms"], 1e-12)
            ),
        },
        "layers": layers,
        "final_hidden": stats(final_hidden),
        "logits": {
            **stats(logits),
            "mean_softmax_entropy": float(entropy.mean()),
            "normalized_softmax_entropy": float(entropy.mean() / math.log(model.config.vocab_size)),
            "mean_top_1_probability": float(top5[..., 0].mean()),
            "mean_top_5_probability_mass": float(top5.sum(-1).mean()),
        },
        "manual_forward_max_absolute_error": float((direct_logits - logits).abs().max()),
        "all_finite": all(
            row[component]["finite"]
            for row in layers
            for component in ("input", "attention", "residual", "mlp", "output")
        ) and bool(torch.isfinite(logits).all()),
    }


@torch.inference_mode()
def validation_metrics(
    model: DiagnosticTransformer, validation: np.memmap, probe_tokens: int = 8192
) -> dict:
    model.eval()
    context = model.config.context_length
    total = 0
    loss_sum = 0.0
    correct = {1: 0, 5: 0, 10: 0}
    for start in range(0, probe_tokens, context):
        size = min(context, probe_tokens - start)
        values = np.asarray(validation[start:start + size + 1], dtype=np.int64).copy()
        x = torch.from_numpy(values[:-1]).unsqueeze(0)
        y = torch.from_numpy(values[1:]).unsqueeze(0)
        logits, loss = model(x, y)
        loss_sum += float(loss) * size
        total += size
        top = logits.topk(10, dim=-1).indices
        for k in correct:
            correct[k] += int((top[..., :k] == y[..., None]).any(-1).sum())
    loss = loss_sum / total
    return {
        "tokens": total,
        "loss": loss,
        "perplexity": math.exp(min(loss, 50)),
        "top_1_accuracy": correct[1] / total,
        "top_5_accuracy": correct[5] / total,
        "top_10_accuracy": correct[10] / total,
    }


@torch.inference_mode()
def frequency_metrics(
    model: DiagnosticTransformer,
    tokenizer: FoundationTokenizer,
    train: np.memmap,
    validation: np.memmap,
    probe_tokens: int = 8192,
) -> dict:
    model.eval()
    vocab = model.config.vocab_size
    train_counts = np.bincount(train, minlength=vocab)
    order = np.argsort(train_counts)[::-1]
    rank = np.empty(vocab, dtype=np.int64)
    rank[order] = np.arange(vocab)
    target_chunks = []
    top1_chunks = []
    top5_chunks = []
    target_probability_chunks = []
    named_ids = {}
    for text in ("。", "、", "の", "に", "は", "を", "が", "<EOS>"):
        ids = [tokenizer.eos_id] if text == "<EOS>" else tokenizer.encode(text)
        if len(ids) != 1:
            raise RuntimeError(f"expected single token for calibration: {text} -> {ids}")
        named_ids[text] = ids[0]
    named_probability_sums = {text: 0.0 for text in named_ids}
    context = model.config.context_length
    total = 0
    for start in range(0, probe_tokens, context):
        size = min(context, probe_tokens - start)
        values = np.asarray(validation[start:start + size + 1], dtype=np.int64).copy()
        x = torch.from_numpy(values[:-1]).unsqueeze(0)
        targets = torch.from_numpy(values[1:])
        logits, _ = model(x)
        logits = logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        top5 = logits.topk(5, dim=-1).indices
        target_chunks.append(targets)
        top1_chunks.append(top5[:, 0])
        top5_chunks.append(top5)
        target_probability_chunks.append(probabilities.gather(1, targets[:, None]).squeeze(1))
        for text, token_id in named_ids.items():
            named_probability_sums[text] += float(probabilities[:, token_id].sum())
        total += size
    targets = torch.cat(target_chunks)
    top1 = torch.cat(top1_chunks)
    top5 = torch.cat(top5_chunks)
    assigned = torch.cat(target_probability_chunks)
    target_ranks = rank[targets.numpy()]
    boundaries = [math.ceil(vocab * value) for value in (.01, .05, .20, .80)]
    definitions = [
        ("top_1_percent", 0, boundaries[0]),
        ("top_5_percent_excluding_top_1", boundaries[0], boundaries[1]),
        ("top_20_percent_excluding_top_5", boundaries[1], boundaries[2]),
        ("middle_20_to_80_percent", boundaries[2], boundaries[3]),
        ("rare_bottom_20_percent", boundaries[3], vocab),
    ]
    buckets = {}
    for name, low, high in definitions:
        mask = torch.from_numpy((target_ranks >= low) & (target_ranks < high))
        count = int(mask.sum())
        target = targets[mask]
        buckets[name] = {
            "rank_range": [low, high - 1],
            "targets": count,
            "top_1_accuracy": float((top1[mask] == target).float().mean()) if count else None,
            "top_5_accuracy": float((top5[mask] == target[:, None]).any(-1).float().mean()) if count else None,
            "mean_target_probability": float(assigned[mask].mean()) if count else None,
            "cross_entropy": float(-assigned[mask].clamp_min(1e-12).log().mean()) if count else None,
        }
    tokens = {}
    for text, token_id in named_ids.items():
        target_mask = targets == token_id
        occurrences = int(target_mask.sum())
        tokens[text] = {
            "token_id": token_id,
            "actual_frequency": occurrences / total,
            "top_1_predicted_frequency": float((top1 == token_id).float().mean()),
            "average_probability": named_probability_sums[text] / total,
            "accuracy_when_target": float((top1[target_mask] == token_id).float().mean()) if occurrences else None,
            "top_5_accuracy_when_target": float(
                (top5[target_mask] == token_id).any(-1).float().mean()
            ) if occurrences else None,
        }
    non_top1 = [
        buckets[name]["top_1_accuracy"]
        for name in buckets
        if name != "top_1_percent" and buckets[name]["top_1_accuracy"] is not None
    ]
    return {
        "probe_tokens": total,
        "bucket_definition": "disjoint train-frequency rank buckets",
        "buckets": buckets,
        "tokens": tokens,
        "non_top_1_percent_any_top_1_accuracy": any(value > 0 for value in non_top1),
        "period_comma_top1_mass": (
            tokens["。"]["top_1_predicted_frequency"]
            + tokens["、"]["top_1_predicted_frequency"]
        ),
    }


def complete_metrics(
    model: DiagnosticTransformer,
    tokenizer: FoundationTokenizer,
    train: np.memmap,
    validation: np.memmap,
    audit_tokens: torch.Tensor,
    validation_tokens: int,
    include_frequency: bool,
) -> dict:
    measured = {
        "validation": validation_metrics(model, validation, validation_tokens),
        "probe": detailed_probe(model, audit_tokens),
    }
    if include_frequency:
        measured["context_sensitivity"] = context_sensitivity(model, validation)
        measured["frequency"] = frequency_metrics(
            model, tokenizer, train, validation, validation_tokens
        )
    return measured
