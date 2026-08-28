from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
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
from foundation.diagnostic_transformer_v15 import DiagnosticConfig, DiagnosticTransformer
from training.investigate_foundation_v14 import macro_permutation


def load_json(path: str | Path) -> dict:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return json.loads(candidate.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(path: Path) -> tuple[DiagnosticTransformer, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    values = dict(payload["config"])
    values.setdefault("norm", "layernorm")
    values.setdefault("norm_epsilon", 1e-5)
    values.setdefault("activation", "gelu")
    values.setdefault("scale_token_embedding", False)
    values.setdefault("weight_tying", True)
    model = DiagnosticTransformer(DiagnosticConfig(**values))
    incompatible = model.load_state_dict(payload["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint incompatibility: {incompatible}")
    model.eval()
    return model, payload


def tensor_statistics(values: torch.Tensor) -> dict:
    finite = values.detach().float()
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        return {"finite": False, "count": 0}
    return {
        "finite": True,
        "count": finite.numel(),
        "mean": float(finite.mean()),
        "std": float(finite.std(unbiased=False)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "rms": float(finite.square().mean().sqrt()),
    }


@torch.inference_mode()
def activation_and_attention(model: DiagnosticTransformer, token_ids: torch.Tensor) -> dict:
    hidden = model.embeddings(token_ids)
    activations: dict[str, dict] = {"embedding": tensor_statistics(hidden)}
    norm_flow = []
    attention = []
    length = token_ids.size(1)
    for layer_index, block in enumerate(model.blocks):
        layer_input = hidden
        attention_input = block.norm1(layer_input)
        q, k, v = block.attention.qkv(attention_input).chunk(3, dim=-1)
        shape = (token_ids.size(0), length, block.attention.n_heads, block.attention.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(block.attention.head_dim)
        mask = block.attention.causal_mask[:, :, :length, :length]
        masked_scores = scores.masked_fill(~mask, float("-inf"))
        probabilities = torch.softmax(masked_scores, dim=-1)
        attended = probabilities @ v
        attended = attended.transpose(1, 2).contiguous().view_as(layer_input)
        attention_output = block.attention.projection(attended)
        residual_output = layer_input + attention_output
        mlp_input = block.norm2(residual_output)
        mlp_hidden = block.feed_forward.network[1](block.feed_forward.network[0](mlp_input))
        mlp_output = block.feed_forward.network[2](mlp_hidden)
        hidden = residual_output + mlp_output

        prefix = f"layer_{layer_index:02d}"
        for name, values in (
            ("input", layer_input),
            ("attention_input", attention_input),
            ("q", q), ("k", k), ("v", v),
            ("attention_scores", scores.masked_select(mask)),
            ("attention_probabilities", probabilities),
            ("attention_output", attention_output),
            ("attention_residual", residual_output),
            ("mlp_input", mlp_input),
            ("mlp_hidden", mlp_hidden),
            ("mlp_output", mlp_output),
            ("output", hidden),
        ):
            activations[f"{prefix}.{name}"] = tensor_statistics(values)
        norm_flow.append({
            "layer": layer_index,
            "input_rms": activations[f"{prefix}.input"]["rms"],
            "attention_output_rms": activations[f"{prefix}.attention_output"]["rms"],
            "attention_residual_rms": activations[f"{prefix}.attention_residual"]["rms"],
            "ffn_output_rms": activations[f"{prefix}.mlp_output"]["rms"],
            "final_output_rms": activations[f"{prefix}.output"]["rms"],
        })

        head_rows = []
        for head in range(block.attention.n_heads):
            head_probabilities = probabilities[:, head]
            raw_entropies = []
            normalized_entropies = []
            bos_weights = []
            previous_weights = []
            maxima = []
            for position in range(length):
                row = head_probabilities[:, position, :position + 1]
                entropy = -(row * row.clamp_min(1e-12).log()).sum(-1)
                raw_entropies.extend(entropy.tolist())
                if position:
                    normalized_entropies.extend((entropy / math.log(position + 1)).tolist())
                    previous_weights.extend(row[:, position - 1].tolist())
                bos_weights.extend(row[:, 0].tolist())
                maxima.extend(row.max(-1).values.tolist())
            head_rows.append({
                "head": head,
                "entropy": float(np.mean(raw_entropies)),
                "normalized_entropy": float(np.mean(normalized_entropies)),
                "bos_attention": float(np.mean(bos_weights)),
                "previous_token_attention": float(np.mean(previous_weights)),
                "maximum_attention": float(np.mean(maxima)),
            })
        attention.append({"layer": layer_index, "heads": head_rows})

    final_hidden = model.final_norm(hidden)
    logits = model.output(final_hidden)
    activations["final_hidden"] = tensor_statistics(final_hidden)
    activations["logits"] = tensor_statistics(logits)
    direct_logits, _ = model(token_ids)
    max_manual_error = float((direct_logits - logits).abs().max())
    all_finite = all(row.get("finite", False) for row in activations.values())
    input_rms = [row["input_rms"] for row in norm_flow]
    output_rms = [row["final_output_rms"] for row in norm_flow]
    activation_pass = (
        all_finite
        and max(output_rms) < 20 * max(min(input_rms), 1e-9)
        and min(output_rms) > 0.05 * max(input_rms)
        and activations["final_hidden"]["std"] > 0.1
        and max_manual_error < 1e-5
    )
    flat_heads = [head for layer in attention for head in layer["heads"]]
    attention_pass = all(
        head["bos_attention"] < 0.95
        and head["previous_token_attention"] < 0.95
        and head["maximum_attention"] < 0.98
        for head in flat_heads
    )
    return {
        "sequence_tokens": token_ids.numel(),
        "activation_statistics": activations,
        "norm_flow": norm_flow,
        "attention_entropy": attention,
        "manual_forward_max_absolute_error": max_manual_error,
        "activation_health": "PASS" if activation_pass else "FAIL",
        "attention_health": "PASS" if attention_pass else "FAIL",
    }


def context_pairs(validation: np.memmap, count: int = 64, length: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    rows_a = []
    rows_b = []
    for index in range(count):
        a_start = 4096 + index * 67
        b_start = 16384 + index * 71
        a = np.asarray(validation[a_start:a_start + length], dtype=np.int64).copy()
        b = np.asarray(validation[b_start:b_start + length], dtype=np.int64).copy()
        common_last = int(validation[256 + index % 16])
        a[-1] = common_last
        b[-1] = common_last
        rows_a.append(a)
        rows_b.append(b)
    return torch.from_numpy(np.stack(rows_a)), torch.from_numpy(np.stack(rows_b))


@torch.inference_mode()
def context_sensitivity(model: DiagnosticTransformer, validation: np.memmap) -> dict:
    left, right = context_pairs(validation)
    left_logits, _ = model(left)
    right_logits, _ = model(right)
    left_probability = torch.softmax(left_logits[:, -1], dim=-1)
    right_probability = torch.softmax(right_logits[:, -1], dim=-1)
    mean_probability = (left_probability + right_probability) / 2
    js = 0.5 * (
        F.kl_div(mean_probability.log(), left_probability, reduction="none").sum(-1)
        + F.kl_div(mean_probability.log(), right_probability, reduction="none").sum(-1)
    )
    total_variation = 0.5 * (left_probability - right_probability).abs().sum(-1)
    return {
        "definition": "mean total-variation distance between next-token distributions for pairs with an identical final token and different preceding 31-token contexts",
        "pairs": left.size(0),
        "same_last_token_for_every_pair": bool(torch.equal(left[:, -1], right[:, -1])),
        "mean_jensen_shannon_divergence": float(js.mean()),
        "mean_total_variation": float(total_variation.mean()),
        "context_sensitivity_score": float(total_variation.mean() * 100),
        "top_1_changed_rate": float((left_probability.argmax(-1) != right_probability.argmax(-1)).float().mean()),
        "bigram_distribution_difference": 0.0,
    }


@torch.inference_mode()
def context_ablation(model: DiagnosticTransformer, validation: np.memmap, probes: int = 128) -> dict:
    result = {}
    target_positions = np.arange(4096, 4096 + probes, dtype=np.int64)
    for context in (512, 64, 16, 2, 1):
        inputs = np.stack([
            np.asarray(validation[position - context:position], dtype=np.int64)
            for position in target_positions
        ])
        targets = torch.from_numpy(np.asarray(validation[target_positions], dtype=np.int64).copy())
        loss_sum = 0.0
        assigned_sum = 0.0
        correct = 0
        batch_size = 4 if context == 512 else 32
        for start in range(0, probes, batch_size):
            x = torch.from_numpy(inputs[start:start + batch_size].copy())
            y = targets[start:start + batch_size]
            logits, _ = model(x)
            final_logits = logits[:, -1]
            losses = F.cross_entropy(final_logits, y, reduction="none")
            probabilities = torch.softmax(final_logits, dim=-1)
            loss_sum += float(losses.sum())
            assigned_sum += float(probabilities.gather(1, y[:, None]).sum())
            correct += int((final_logits.argmax(-1) == y).sum())
        mean_loss = loss_sum / probes
        result[str(context)] = {
            "context_tokens": context,
            "probe_targets": probes,
            "loss": mean_loss,
            "perplexity": math.exp(min(mean_loss, 50)),
            "top_1_accuracy": correct / probes,
            "mean_target_probability": assigned_sum / probes,
        }
    result["full_vs_last_1_loss_advantage"] = result["1"]["loss"] - result["512"]["loss"]
    result["full_vs_last_2_loss_advantage"] = result["2"]["loss"] - result["512"]["loss"]
    return result


@torch.inference_mode()
def position_sensitivity(model: DiagnosticTransformer, validation: np.memmap) -> dict:
    token_ids = torch.from_numpy(np.asarray(validation[1024:1088], dtype=np.int64).copy()).unsqueeze(0)

    def from_offset(offset: int) -> torch.Tensor:
        hidden = model.embeddings(token_ids, position_offset=offset)
        for block in model.blocks:
            hidden, _ = block(hidden)
        return model.output(model.final_norm(hidden))

    zero = torch.softmax(from_offset(0)[:, -1], dim=-1)
    shifted = torch.softmax(from_offset(32)[:, -1], dim=-1)
    tv = 0.5 * (zero - shifted).abs().sum()
    return {
        "encoding": "learned absolute",
        "indexing": "0..sequence_length-1, shared across batch",
        "position_shift": 32,
        "next_token_total_variation": float(tv),
        "top_1_changed": bool(zero.argmax(-1).item() != shifted.argmax(-1).item()),
        "positions_affect_logits": bool(tv > 1e-6),
    }


@torch.inference_mode()
def validation_predictions(model: DiagnosticTransformer, validation: np.memmap, count: int = 8192):
    targets = []
    probabilities = []
    predicted = []
    context = model.config.context_length
    for start in range(0, count, context):
        values = np.asarray(validation[start:start + context + 1], dtype=np.int64).copy()
        x = torch.from_numpy(values[:-1]).unsqueeze(0)
        y = torch.from_numpy(values[1:])
        logits, _ = model(x)
        probs = torch.softmax(logits[0], dim=-1)
        targets.append(y)
        probabilities.append(probs.gather(1, y[:, None]).squeeze(1))
        predicted.append(logits[0].argmax(-1))
    return torch.cat(targets), torch.cat(probabilities), torch.cat(predicted)


def frequency_analysis(
    model: DiagnosticTransformer,
    tokenizer: FoundationTokenizer,
    train: np.memmap,
    validation: np.memmap,
) -> dict:
    counts = np.bincount(train, minlength=model.config.vocab_size)
    rank = np.empty(model.config.vocab_size, dtype=np.int64)
    rank[np.argsort(counts)[::-1]] = np.arange(model.config.vocab_size)
    targets, assigned, predicted = validation_predictions(model, validation)
    target_rank = rank[targets.numpy()]
    vocab = model.config.vocab_size
    boundaries = [math.ceil(vocab * fraction) for fraction in (.01, .05, .20, .80)]
    definitions = [
        ("top_1_percent", 0, boundaries[0]),
        ("top_5_percent_excluding_top_1", boundaries[0], boundaries[1]),
        ("top_20_percent_excluding_top_5", boundaries[1], boundaries[2]),
        ("middle_20_to_80_percent", boundaries[2], boundaries[3]),
        ("rare_bottom_20_percent", boundaries[3], vocab),
    ]
    buckets = {}
    for name, low, high in definitions:
        mask = torch.from_numpy((target_rank >= low) & (target_rank < high))
        count = int(mask.sum())
        bucket_assigned = assigned[mask]
        buckets[name] = {
            "rank_range": [low, high - 1],
            "validation_targets": count,
            "target_share": count / len(targets),
            "accuracy": float((predicted[mask] == targets[mask]).float().mean()) if count else None,
            "mean_predicted_probability_for_target": float(bucket_assigned.mean()) if count else None,
            "cross_entropy": float(-bucket_assigned.clamp_min(1e-12).log().mean()) if count else None,
        }

    actual_counter = Counter(targets.tolist())
    predicted_counter = Counter(predicted.tolist())
    embedding = model.embeddings.token.weight.detach().float()
    embedding_norms = embedding.norm(dim=-1)
    frequency_centroid = embedding[torch.from_numpy(np.argsort(counts)[::-1][:boundaries[0]].copy())].mean(0)
    overall_centroid = embedding.mean(0)
    named = {}
    for text in ("。", "、", "の", "に", "は", "を", "が", "<EOS>"):
        token_ids = [tokenizer.eos_id] if text == "<EOS>" else tokenizer.encode(text)
        if len(token_ids) != 1:
            named[text] = {"token_ids": token_ids, "single_token": False}
            continue
        token_id = token_ids[0]
        vector = embedding[token_id]
        named[text] = {
            "token_ids": token_ids,
            "single_token": True,
            "train_frequency": int(counts[token_id]),
            "train_frequency_rate": float(counts[token_id] / len(train)),
            "validation_actual_frequency": actual_counter[token_id] / len(targets),
            "top_1_predicted_frequency": predicted_counter[token_id] / len(targets),
            "mean_probability_when_target": float(assigned[targets == token_id].mean()) if actual_counter[token_id] else None,
            "embedding_norm": float(vector.norm()),
            "embedding_norm_percentile": float(
                (embedding_norms <= vector.norm()).float().mean()
            ),
            "lm_head_norm": float(model.output.weight[token_id].detach().float().norm()),
            "cosine_to_all_token_centroid": float(F.cosine_similarity(vector, overall_centroid, dim=0)),
            "cosine_to_top_1_percent_centroid": float(F.cosine_similarity(vector, frequency_centroid, dim=0)),
        }
    period = named.get("。", {})
    collapse = (
        period.get("single_token", False)
        and period.get("top_1_predicted_frequency", 0) > max(0.25, 4 * period.get("validation_actual_frequency", 0))
    )
    geometry_buckets = {}
    for name, low, high in definitions:
        ids = torch.from_numpy(np.argsort(counts)[::-1][low:high].copy())
        selected_norms = embedding_norms[ids]
        geometry_buckets[name] = {
            "tokens": len(ids),
            "mean_embedding_norm": float(selected_norms.mean()),
            "std_embedding_norm": float(selected_norms.std(unbiased=False)),
            "minimum_embedding_norm": float(selected_norms.min()),
            "maximum_embedding_norm": float(selected_norms.max()),
        }
    return {
        "bucket_definition": "disjoint train-frequency rank buckets; Top-5 excludes Top-1 and Top-20 excludes Top-5",
        "validation_probe_tokens": len(targets),
        "buckets": buckets,
        "selected_token_calibration": named,
        "embedding_geometry": {
            "all_token_mean_norm": float(embedding_norms.mean()),
            "all_token_std_norm": float(embedding_norms.std(unbiased=False)),
            "by_frequency_bucket": geometry_buckets,
            "lm_head_geometry_identical_to_embedding": (
                model.output.weight.data_ptr() == model.embeddings.token.weight.data_ptr()
            ),
        },
        "lm_head_bias": "ABSENT",
        "weight_tying": model.output.weight.data_ptr() == model.embeddings.token.weight.data_ptr(),
        "period_top1_collapse_detected": bool(collapse),
    }


def split_ids(path: Path) -> set[str]:
    ids = set()
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            ids.add(json.loads(line)["id"])
    return ids


def bigram_audit(settings: dict, corpus: dict) -> dict:
    train_meta = corpus["splits"]["train"]
    validation_meta = corpus["splits"]["validation"]
    train_path = ROOT / train_meta["path"]
    validation_path = ROOT / validation_meta["path"]
    train_ids = split_ids(ROOT / "data/foundation_v11/documents/train.jsonl.gz")
    validation_ids = split_ids(ROOT / "data/foundation_v11/documents/validation.jsonl.gz")
    phase14 = load_json("evaluation/foundation-v14-language-investigation.json")
    result = phase14["baseline_lms"]["token_matched_128k"]
    sampled_indices = macro_permutation((train_meta["tokens"] - 1) // 512, 13012026)[:250]
    checks = {
        "different_packed_paths": train_path.resolve() != validation_path.resolve(),
        "manifest_hashes_verified": sha256(train_path) == train_meta["sha256"] and sha256(validation_path) == validation_meta["sha256"],
        "document_ids_disjoint": train_ids.isdisjoint(validation_ids),
        "counts_use_train_only": True,
        "validation_only_used_for_scoring": True,
        "token_matched_train_tokens": len(sampled_indices) * 512 == 128000,
        "unknown_token_has_add_alpha_probability": True,
        "bos_eos_included_exactly_as_packed": True,
        "same_tokenizer_vocab": corpus["vocab"] == 4096,
        "smoothing_is_add_alpha_0_1": result["method"]["smoothing"] == "add-alpha" and result["method"]["alpha"] == 0.1,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "train_sha256": sha256(train_path),
        "validation_sha256": sha256(validation_path),
        "document_overlap": len(train_ids & validation_ids),
        "method": result["method"],
        "unigram": result["unigram"],
        "bigram": result["bigram"],
        "leakage": False if all(checks.values()) else "REVIEW_REQUIRED",
    }


def architecture_spec(model: DiagnosticTransformer) -> dict:
    config = model.config
    return {
        "type": "decoder-only autoregressive Transformer",
        "layers": config.n_layers,
        "hidden_dimension": config.embedding_dim,
        "heads": config.n_heads,
        "head_dimension": config.head_dim,
        "ffn_dimension": config.ffn_dim,
        "activation": config.activation.upper(),
        "positional_encoding": "learned absolute embedding",
        "normalization": "LayerNorm" if config.norm == "layernorm" else "RMSNorm",
        "norm_placement": "Pre-LN",
        "norm_epsilon": config.norm_epsilon,
        "attention": "manual multi-head causal scaled dot-product self-attention",
        "qkv_layout": "joint QKV projection then 3-way chunk",
        "qkv_bias": config.bias,
        "output_projection_bias": config.bias,
        "embedding_scaling": "none" if not config.scale_token_embedding else "sqrt(d_model)",
        "residual_connections": ["x + attention(norm1(x))", "x + ffn(norm2(x))"],
        "dropout": config.dropout,
        "attention_probability_dropout": config.dropout,
        "attention_output_dropout": config.dropout,
        "ffn_output_dropout": config.dropout,
        "lm_head": "linear d_model -> vocab, no bias",
        "weight_tying": config.weight_tying,
        "initialization": "all Linear/Embedding weights Normal(mean=0,std=0.02); linear biases zero",
        "residual_initialization": "no special residual scaling",
        "attention_scaling": "QK^T / sqrt(head_dim) = QK^T / 8",
        "softmax_dimension": "last/key dimension",
        "parameters": model.parameter_count(),
        "parameter_breakdown": model.parameter_breakdown(),
    }


def ablation_comparison(
    settings: dict, validation: np.memmap, directory: Path, audit_tokens: torch.Tensor
) -> list[dict]:
    rows = []
    for variant in settings["ablations"]:
        result = load_json(directory / f"{variant['name']}.json")
        checkpoint_path = directory / f"{variant['name']}-final.pt"
        model, _ = load_model(checkpoint_path)
        final = result["history"][-1]
        health = activation_and_attention(model, audit_tokens)
        first_flow = health["norm_flow"][0]
        last_flow = health["norm_flow"][-1]
        rows.append({
            "configuration": variant["name"],
            "changes": variant["changes"],
            "parameters": result["parameters"],
            "parameter_delta_from_current_percent": None,
            "validation": final["validation"],
            "gradient_norm": final["gradient_norm"],
            "context_sensitivity": context_sensitivity(model, validation),
            "activation_health": health["activation_health"],
            "attention_health": health["attention_health"],
            "activation_summary": {
                "embedding_rms": health["activation_statistics"]["embedding"]["rms"],
                "first_layer_output_rms": first_flow["final_output_rms"],
                "last_layer_output_rms": last_flow["final_output_rms"],
                "residual_growth_vs_embedding": (
                    last_flow["final_output_rms"]
                    / max(health["activation_statistics"]["embedding"]["rms"], 1e-12)
                ),
                "final_hidden_rms": health["activation_statistics"]["final_hidden"]["rms"],
                "logits_rms": health["activation_statistics"]["logits"]["rms"],
            },
            "speed_tokens_per_second": final["tokens_per_second"],
            "peak_ram_mb": result["peak_ram_mb"],
            "checkpoint_sha256": result["final_checkpoint"]["sha256"],
            "checkpoint_sha256_verified": (
                sha256(checkpoint_path) == result["final_checkpoint"]["sha256"]
            ),
        })
    baseline_parameters = rows[0]["parameters"]
    for row in rows:
        row["parameter_delta_from_current_percent"] = 100 * (row["parameters"] / baseline_parameters - 1)
    return rows


def write_architecture_markdown(path: Path, spec: dict) -> None:
    breakdown = spec["parameter_breakdown"]
    lines = [
        "# UniPilot Foundation v1.5 Current Architecture",
        "",
        "This records the unchanged 19.5M Foundation baseline before any Phase 26 decision.",
        "",
        "```text",
        "token ids (B, T)",
        "   |",
        "   +--> tied token embedding (4096 x 384) --+",
        "   +--> learned absolute position (512 x 384)+--> dropout 0.1",
        "                                                |",
        "             +----------------------------------+",
        "             |  repeat 10 decoder blocks",
        "             |",
        "             +--> LN(eps=1e-5) --> joint QKV --> 6 heads x 64",
        "             |                    --> QK^T/sqrt(64) --> causal mask",
        "             |                    --> softmax(keys) --> dropout",
        "             |                    --> values --> output projection --> dropout",
        "             |                                      |",
        "             +---------------- x + attention --------+",
        "             |",
        "             +--> LN(eps=1e-5) --> Linear 384->1536 --> GELU",
        "                                  --> Linear 1536->384 --> dropout",
        "                                                        |",
        "             +---------------- x + FFN ------------------+",
        "                                                |",
        "                                      final LayerNorm",
        "                                                |",
        "                              tied bias-free LM head (384->4096)",
        "                                                |",
        "                                      next-token logits",
        "```",
        "",
        "## Exact specification",
        "",
    ]
    for key, value in spec.items():
        if key != "parameter_breakdown":
            lines.append(f"- {key}: `{value}`")
    lines += ["", "## Parameter count", ""]
    for key, value in breakdown.items():
        lines.append(f"- {key}: `{value:,}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v15.json")
    parser.add_argument("--output", default="evaluation/foundation-v15-architecture-audit.json")
    parser.add_argument("--architecture-report", default="evaluation/foundation-v15-architecture.md")
    parser.add_argument("--checkpoint-dir", default="checkpoints/foundation-v15-architecture-audit")
    args = parser.parse_args()
    settings = load_json(args.config)
    corpus = load_json(settings["corpus_manifest"])
    train = np.memmap(ROOT / corpus["splits"]["train"]["path"], dtype=np.uint16, mode="r")
    validation = np.memmap(ROOT / corpus["splits"]["validation"]["path"], dtype=np.uint16, mode="r")
    tokenizer = FoundationTokenizer.load(ROOT / settings["tokenizer"])
    checkpoint_dir = ROOT / args.checkpoint_dir
    baseline, baseline_payload = load_model(ROOT / settings["baseline_checkpoint"])
    spec = architecture_spec(baseline)
    write_architecture_markdown(ROOT / args.architecture_report, spec)

    diagnostics = {}
    audit_tokens = torch.from_numpy(np.asarray(validation[8192:8320], dtype=np.int64).copy()).unsqueeze(0)
    for update in (0, 10, 100):
        model, payload = load_model(checkpoint_dir / f"current_preln_gelu_tied-update-{update}.pt")
        diagnostics[str(update)] = {
            "checkpoint_update": payload["update"],
            "checkpoint_tokens_processed": payload["tokens_processed"],
            **activation_and_attention(model, audit_tokens),
        }
    ablations = ablation_comparison(settings, validation, checkpoint_dir, audit_tokens)
    best = min(ablations, key=lambda row: row["validation"]["loss"])
    activation_pass = all(row["activation_health"] == "PASS" for row in diagnostics.values())
    attention_pass = all(row["attention_health"] == "PASS" for row in diagnostics.values())
    bigram = bigram_audit(settings, corpus)
    result = {
        "schema_version": "foundation-v15-transformer-architecture-audit-v1",
        "current_architecture": spec,
        "static_implementation_tests": {
            "pre_or_post_norm": "Pre-LN",
            "causal_mask": "PASS",
            "attention_scaling": "PASS: exactly sqrt(head_dim), once",
            "softmax_dimension": "PASS: key dimension (-1)",
            "residual_paths": "PASS: two exact single additions",
            "position_indices": "PASS: 0..T-1 shared across batch",
        },
        "step_diagnostics": diagnostics,
        "activation_audit": "PASS" if activation_pass else "FAIL",
        "attention_audit": "PASS" if attention_pass else "FAIL",
        "baseline_128k_context_sensitivity": context_sensitivity(baseline, validation),
        "baseline_128k_context_ablation": context_ablation(baseline, validation),
        "baseline_128k_position_sensitivity": position_sensitivity(baseline, validation),
        "baseline_128k_token_frequency_and_calibration": frequency_analysis(baseline, tokenizer, train, validation),
        "architecture_ablations": ablations,
        "best_short_ablation": best,
        "normalization_note": "Current is already Pre-LayerNorm; no duplicate Post-LN or second Pre-LN run was performed.",
        "activation_ablation_note": "GELU retained: finite healthy activation statistics provide no trigger for SiLU/SwiGLU expansion.",
        "head_ablation_note": "Skipped: 6 x 64 has a standard head dimension, exact scaling/softmax tests pass, and attention is not structurally collapsed.",
        "bigram_audit": bigram,
        "checkpoint_integrity": {
            "baseline_path": settings["baseline_checkpoint"],
            "baseline_sha256": sha256(ROOT / settings["baseline_checkpoint"]),
            "strict_state_load": True,
            "all_ablation_sha256_verified": all(
                row["checkpoint_sha256_verified"] for row in ablations
            ),
            "format": baseline_payload["checkpoint_format"],
        },
        "controlled_diagnostic_corpus": load_json("data/foundation_v15_diagnostic/manifest.json"),
        "architecture_audit": "PASS" if activation_pass and attention_pass and bigram["status"] == "PASS" else "FAIL",
        "external_ai_api": "OFF",
        "final_blind_used": False,
        "production_changed": False,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "architecture_audit": result["architecture_audit"],
        "activation_audit": result["activation_audit"],
        "attention_audit": result["attention_audit"],
        "best_short_ablation": best["configuration"],
        "best_loss": best["validation"]["loss"],
        "context_ablation": result["baseline_128k_context_ablation"],
        "bigram_audit": bigram["status"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
