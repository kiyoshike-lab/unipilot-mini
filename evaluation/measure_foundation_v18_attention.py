from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _rms(values: torch.Tensor) -> float:
    return float(values.detach().float().square().mean().sqrt())


def _custom_attention(block, normalized: torch.Tensor):
    batch, length, channels = normalized.shape
    q, k, v = block.attention.qkv(normalized).chunk(3, dim=-1)
    heads = block.attention.n_heads
    dimension = block.attention.head_dim
    q = q.view(batch, length, heads, dimension).transpose(1, 2)
    k = k.view(batch, length, heads, dimension).transpose(1, 2)
    v = v.view(batch, length, heads, dimension).transpose(1, 2)
    raw = q @ k.transpose(-2, -1)
    scaled = raw / math.sqrt(dimension)
    mask = block.attention.causal_mask[:, :, :length, :length]
    probabilities = torch.softmax(scaled.masked_fill(~mask, float("-inf")), dim=-1)
    attended = probabilities @ v
    attended = attended.transpose(1, 2).contiguous().view(batch, length, channels)
    output = block.attention.projection(attended)
    return q, k, raw, scaled, probabilities, output


def _reference_attention(block, normalized: torch.Tensor):
    batch, length, channels = normalized.shape
    weights = block.attention.in_proj_weight.chunk(3, dim=0)
    if block.attention.in_proj_bias is None:
        biases = (None, None, None)
    else:
        biases = block.attention.in_proj_bias.chunk(3, dim=0)
    q, k, v = [
        F.linear(normalized, weight, bias)
        for weight, bias in zip(weights, biases)
    ]
    heads = block.attention.num_heads
    dimension = channels // heads
    q = q.view(batch, length, heads, dimension).transpose(1, 2)
    k = k.view(batch, length, heads, dimension).transpose(1, 2)
    v = v.view(batch, length, heads, dimension).transpose(1, 2)
    raw = q @ k.transpose(-2, -1)
    scaled = raw / math.sqrt(dimension)
    allowed = ~block.causal_mask[:length, :length][None, None]
    probabilities = torch.softmax(scaled.masked_fill(~allowed, float("-inf")), dim=-1)
    attended = probabilities @ v
    attended = attended.transpose(1, 2).contiguous().view(batch, length, channels)
    output = F.linear(
        attended,
        block.attention.out_proj.weight,
        block.attention.out_proj.bias,
    )
    return q, k, raw, scaled, probabilities, output


@torch.inference_mode()
def attention_retrieval_metrics(model, implementation: str, examples: list[tuple]) -> dict:
    model.eval()
    inputs = torch.tensor([row[0] for row in examples], dtype=torch.long)
    hidden = model.embeddings(inputs)
    layer_rows = []
    for layer_index, block in enumerate(model.blocks):
        normalized = block.norm1(hidden)
        if implementation == "custom":
            q, k, raw, scaled, probabilities, attention_output = _custom_attention(
                block, normalized
            )
            post_attention = hidden + attention_output
            mlp_output = block.feed_forward(block.norm2(post_attention))
        elif implementation == "reference":
            q, k, raw, scaled, probabilities, attention_output = _reference_attention(
                block, normalized
            )
            post_attention = hidden + attention_output
            mlp_output = block.feed_forward(block.norm2(post_attention))
        else:
            raise KeyError(implementation)
        hidden = post_attention + mlp_output
        query_position = inputs.size(1) - 1
        query_probabilities = probabilities[:, :, query_position, :]
        query_raw = raw[:, :, query_position, :]
        query_scaled = scaled[:, :, query_position, :]
        query_q = q[:, :, query_position, :]
        correct_key = torch.tensor([
            row[2]["correct_key_position"] for row in examples
        ], dtype=torch.long)
        correct_value = torch.tensor([
            row[2]["correct_value_position"] for row in examples
        ], dtype=torch.long)
        batch_indices = torch.arange(inputs.size(0))[:, None]
        head_indices = torch.arange(query_probabilities.size(1))[None, :]
        key_mass = query_probabilities[batch_indices, head_indices, correct_key[:, None]]
        value_mass = query_probabilities[batch_indices, head_indices, correct_value[:, None]]
        correct_mass = key_mass + value_mass
        correct_logits = torch.maximum(
            query_scaled[batch_indices, head_indices, correct_key[:, None]],
            query_scaled[batch_indices, head_indices, correct_value[:, None]],
        )
        incorrect_mask = torch.ones_like(query_scaled, dtype=torch.bool)
        incorrect_mask[batch_indices, head_indices, correct_key[:, None]] = False
        incorrect_mask[batch_indices, head_indices, correct_value[:, None]] = False
        best_incorrect = query_scaled.masked_fill(~incorrect_mask, float("-inf")).max(-1).values
        margin = correct_logits - best_incorrect
        order = query_scaled.argsort(dim=-1, descending=True)
        key_rank = (order == correct_key[:, None, None]).nonzero(as_tuple=False)[:, 2] + 1
        value_rank = (order == correct_value[:, None, None]).nonzero(as_tuple=False)[:, 2] + 1
        best_rank = torch.minimum(
            key_rank.view(inputs.size(0), -1),
            value_rank.view(inputs.size(0), -1),
        )
        maximum_positions = query_probabilities.argmax(-1)
        maximum_is_correct = (
            (maximum_positions == correct_key[:, None])
            | (maximum_positions == correct_value[:, None])
        )
        entropy = -(
            query_probabilities * query_probabilities.clamp_min(1e-12).log()
        ).sum(-1) / math.log(query_probabilities.size(-1))
        top3 = query_probabilities.topk(3, dim=-1).values.sum(-1)
        heads = []
        for head in range(query_probabilities.size(1)):
            heads.append({
                "head": head,
                "normalized_entropy": float(entropy[:, head].mean()),
                "max_attention_probability": float(
                    query_probabilities[:, head].max(-1).values.mean()
                ),
                "top_3_attention_mass": float(top3[:, head].mean()),
                "correct_key_mass": float(key_mass[:, head].mean()),
                "correct_value_mass": float(value_mass[:, head].mean()),
                "correct_key_value_mass": float(correct_mass[:, head].mean()),
                "incorrect_position_mass": float(1 - correct_mass[:, head].mean()),
                "max_position_is_correct_rate": float(
                    maximum_is_correct[:, head].float().mean()
                ),
                "correct_position_mean_rank": float(best_rank[:, head].float().mean()),
                "q_rms": _rms(query_q[:, head]),
                "k_rms": _rms(k[:, head]),
                "qk_dot_product_std": float(query_raw[:, head].float().std(unbiased=False)),
                "scaled_attention_logit_std": float(
                    query_scaled[:, head].float().std(unbiased=False)
                ),
                "attention_margin": float(margin[:, head].mean()),
            })
        layer_rows.append({
            "layer": layer_index,
            "heads": heads,
            "mean": {
                name: sum(row[name] for row in heads) / len(heads)
                for name in (
                    "normalized_entropy",
                    "max_attention_probability",
                    "top_3_attention_mass",
                    "correct_key_mass",
                    "correct_value_mass",
                    "correct_key_value_mass",
                    "incorrect_position_mass",
                    "max_position_is_correct_rate",
                    "correct_position_mean_rank",
                    "q_rms",
                    "k_rms",
                    "qk_dot_product_std",
                    "scaled_attention_logit_std",
                    "attention_margin",
                )
            },
        })
    sequence_length = inputs.size(1)
    return {
        "examples": len(examples),
        "sequence_length": sequence_length,
        "chance_correct_key_value_mass": 2 / sequence_length,
        "chance_correct_position_rank": (sequence_length + 1) / 2,
        "layers": layer_rows,
    }
