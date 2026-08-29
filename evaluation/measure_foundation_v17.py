from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from foundation.base_tokenizer import FoundationTokenizer
from foundation.diagnostic_transformer_v17 import DiagnosticTransformerV17


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
        "norm": float(torch.linalg.vector_norm(finite)),
    }


def norm_audit(norm, values: torch.Tensor) -> dict:
    normalized = norm(values)
    gamma = stats(norm.weight)
    beta = stats(norm.bias) if getattr(norm, "bias", None) is not None else None
    return {
        "input": stats(values),
        "normalized": stats(normalized),
        "gamma": gamma,
        "beta": beta,
    }


@torch.inference_mode()
def architecture_probe(model: DiagnosticTransformerV17, token_ids: torch.Tensor) -> dict:
    model.eval()
    length = token_ids.size(1)
    positions = torch.arange(length, device=token_ids.device)
    raw_token = model.embeddings.token(token_ids)
    raw_position = model.embeddings.position(positions)[None, :, :]
    token_only = raw_token * model.embeddings.token_scale
    position_only = raw_position * model.embeddings.position_scale
    hidden = token_only + position_only
    layers = []
    norm_rows = []
    for index, block in enumerate(model.blocks):
        layer_input = hidden
        norm1 = norm_audit(block.norm1, layer_input)
        attention_output, _ = block.attention(block.norm1(layer_input))
        post_attention = layer_input + attention_output
        norm2 = norm_audit(block.norm2, post_attention)
        mlp_output = block.feed_forward(block.norm2(post_attention))
        hidden = post_attention + mlp_output
        input_stats = stats(layer_input)
        attention_stats = stats(attention_output)
        post_attention_stats = stats(post_attention)
        mlp_stats = stats(mlp_output)
        output_stats = stats(hidden)
        layers.append({
            "layer": index,
            "pre_attention": input_stats,
            "attention_output": attention_stats,
            "post_attention_residual": post_attention_stats,
            "mlp_output": mlp_stats,
            "post_mlp_residual": output_stats,
            "attention_to_residual_rms_ratio": (
                attention_stats["rms"] / max(input_stats["rms"], 1e-12)
            ),
            "attention_to_post_residual_rms_ratio": (
                attention_stats["rms"] / max(post_attention_stats["rms"], 1e-12)
            ),
            "mlp_to_residual_rms_ratio": (
                mlp_stats["rms"] / max(post_attention_stats["rms"], 1e-12)
            ),
            "mlp_to_post_residual_rms_ratio": (
                mlp_stats["rms"] / max(output_stats["rms"], 1e-12)
            ),
        })
        norm_rows.append({"layer": index, "pre_attention": norm1, "pre_mlp": norm2})
    final_norm = norm_audit(model.final_norm, hidden)
    final_hidden = model.final_norm(hidden)
    logits = model.output(final_hidden)
    probabilities = torch.softmax(logits, dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    top5 = probabilities.topk(5, dim=-1).values
    direct_logits, _ = model(token_ids)
    return {
        "embedding": {
            "raw_token": stats(raw_token),
            "effective_token": stats(token_only),
            "raw_position": stats(raw_position),
            "effective_position": stats(position_only),
            "combined": stats(token_only + position_only),
            "effective_token_to_position_rms_ratio": (
                stats(token_only)["rms"] / max(stats(position_only)["rms"], 1e-12)
            ),
            "token_scale": model.embeddings.token_scale,
            "position_scale": model.embeddings.position_scale,
        },
        "representation_contribution": {
            "token_only": stats(token_only),
            "position_only": stats(position_only),
            "token_plus_position": stats(token_only + position_only),
        },
        "layers": layers,
        "norms": norm_rows,
        "final_norm": {
            "present": hasattr(model, "final_norm"),
            "order": "Embedding -> Blocks -> Final Norm -> LM Head",
            **final_norm,
        },
        "logits": {
            **stats(logits),
            "mean_softmax_entropy": float(entropy.mean()),
            "normalized_softmax_entropy": float(
                entropy.mean() / math.log(model.config.vocab_size)
            ),
            "mean_top_1_probability": float(top5[..., 0].mean()),
            "mean_top_5_probability_mass": float(top5.sum(-1).mean()),
        },
        "manual_forward_max_absolute_error": float((direct_logits - logits).abs().max()),
        "all_finite": bool(torch.isfinite(logits).all()) and all(
            component["finite"]
            for layer in layers
            for component in (
                layer["pre_attention"], layer["attention_output"],
                layer["post_attention_residual"], layer["mlp_output"],
                layer["post_mlp_residual"],
            )
        ),
    }


@torch.inference_mode()
def validation_metrics(
    model: DiagnosticTransformerV17, validation: np.memmap, probe_tokens: int = 8192
) -> dict:
    model.eval()
    context = model.config.context_length
    total = 0
    loss_sum = 0.0
    correct = {1: 0, 5: 0, 10: 0}
    started = time.perf_counter()
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
    elapsed = time.perf_counter() - started
    loss = loss_sum / total
    return {
        "tokens": total,
        "loss": loss,
        "perplexity": math.exp(min(loss, 50)),
        "top_1_accuracy": correct[1] / total,
        "top_5_accuracy": correct[5] / total,
        "top_10_accuracy": correct[10] / total,
        "wall_seconds": elapsed,
        "tokens_per_second": total / elapsed,
    }


def _named_token_ids(tokenizer: FoundationTokenizer) -> dict[str, int]:
    result = {}
    for text in ("。", "、", "の", "に", "は", "を", "が", "<EOS>"):
        ids = [tokenizer.eos_id] if text == "<EOS>" else tokenizer.encode(text)
        if len(ids) != 1:
            raise RuntimeError(f"expected one token: {text} -> {ids}")
        result[text] = ids[0]
    return result


@torch.inference_mode()
def hidden_token_similarity(
    model: DiagnosticTransformerV17,
    tokenizer: FoundationTokenizer,
    validation: np.memmap,
    probe_tokens: int = 2048,
) -> dict:
    model.eval()
    context = model.config.context_length
    embeddings = F.normalize(model.embeddings.token.weight.float(), dim=-1)
    named_ids = _named_token_ids(tokenizer)
    hidden_norms = []
    correct_cosines = []
    correct_logits = []
    named_cosines = {name: [] for name in named_ids}
    named_logits = {name: [] for name in named_ids}
    for start in range(0, probe_tokens, context):
        size = min(context, probe_tokens - start)
        values = np.asarray(validation[start:start + size + 1], dtype=np.int64).copy()
        token_ids = torch.from_numpy(values[:-1]).unsqueeze(0)
        targets = torch.from_numpy(values[1:])
        hidden = model.embeddings(token_ids)
        for block in model.blocks:
            hidden, _ = block(hidden)
        hidden = model.final_norm(hidden)[0].float()
        normalized_hidden = F.normalize(hidden, dim=-1)
        logits = hidden @ model.embeddings.token.weight.float().T
        hidden_norms.extend(hidden.norm(dim=-1).tolist())
        correct_cosines.extend(
            (normalized_hidden * embeddings[targets]).sum(-1).tolist()
        )
        correct_logits.extend(logits.gather(1, targets[:, None]).squeeze(1).tolist())
        for name, token_id in named_ids.items():
            named_cosines[name].extend((normalized_hidden @ embeddings[token_id]).tolist())
            named_logits[name].extend(logits[:, token_id].tolist())
    return {
        "probe_tokens": probe_tokens,
        "hidden_norm": stats(torch.tensor(hidden_norms)),
        "correct_token_cosine": stats(torch.tensor(correct_cosines)),
        "correct_token_logit": stats(torch.tensor(correct_logits)),
        "named_token_cosine": {
            name: stats(torch.tensor(values)) for name, values in named_cosines.items()
        },
        "named_token_logit": {
            name: stats(torch.tensor(values)) for name, values in named_logits.items()
        },
        "logit_identity": "tied LM logit = ||hidden|| * ||token_embedding|| * cosine",
    }
