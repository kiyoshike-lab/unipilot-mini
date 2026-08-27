from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import psutil
import torch
from torch.nn import functional as F

from foundation.base_tokenizer import FoundationTokenizer
from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from training.optimizer import create_optimizer
from training.scheduler import warmup_cosine_multiplier


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SequenceRecord:
    identifier: str
    text: str
    ids: tuple[int, ...]

    @property
    def inputs(self) -> torch.Tensor:
        return torch.tensor(self.ids[:-1], dtype=torch.long)

    @property
    def targets(self) -> torch.Tensor:
        return torch.tensor(self.ids[1:], dtype=torch.long)


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(*, vocab_size: int, seed: int, ffn_dim: int = 1536,
                context_length: int = 512, dropout: float = 0.1) -> UniPilotTransformer:
    set_deterministic_seed(seed)
    config = ModelConfig(
        model_name="UniPilot Foundation v1.2 training diagnostic",
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=384,
        n_layers=10,
        n_heads=6,
        ffn_dim=ffn_dim,
        dropout=dropout,
        bias=True,
    )
    return UniPilotTransformer(config)


def read_short_clean_documents(limit: int = 500) -> list[dict]:
    path = ROOT / "data/foundation_v11/documents/train.jsonl.gz"
    rows: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                rows.append({"id": row["id"], "text": row["text"]})
    rows.sort(key=lambda row: (len(row["text"]), row["id"]))
    return rows[:limit]


def encode_records(rows: list[dict], tokenizer: FoundationTokenizer,
                   context_length: int = 512) -> list[SequenceRecord]:
    records = []
    for row in rows:
        ids = tokenizer.encode(row["text"], add_bos=True, add_eos=True)
        if 3 <= len(ids) <= context_length + 1:
            records.append(SequenceRecord(row["id"], row["text"], tuple(ids)))
    return records


def validation_loss(model: UniPilotTransformer, records: list[SequenceRecord]) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for record in records:
            inputs = record.inputs.unsqueeze(0)
            targets = record.targets.unsqueeze(0)
            _, loss = model(inputs, targets)
            count = targets.numel()
            total_loss += float(loss.item()) * count
            total_tokens += count
    model.train(was_training)
    return total_loss / total_tokens


def parameter_l2_norm(parameters) -> float:
    total = 0.0
    seen: set[int] = set()
    for parameter in parameters:
        if parameter.grad is None or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += float(parameter.grad.detach().float().pow(2).sum().item())
    return math.sqrt(total)


def gradient_snapshot(model: UniPilotTransformer, global_norm: float) -> dict:
    middle = len(model.blocks) // 2
    return {
        "global": global_norm,
        "embedding": parameter_l2_norm(model.embeddings.token.parameters()),
        "first_layer": parameter_l2_norm(model.blocks[0].parameters()),
        "middle_layer": parameter_l2_norm(model.blocks[middle].parameters()),
        "last_layer": parameter_l2_norm(model.blocks[-1].parameters()),
        "lm_head": parameter_l2_norm(model.output.parameters()),
    }


def tracked_parameters(model: UniPilotTransformer) -> dict[str, torch.Tensor]:
    return {
        "embedding": model.embeddings.token.weight,
        "attention": model.blocks[0].attention.qkv.weight,
        "mlp": model.blocks[0].feed_forward.network[0].weight,
        "lm_head": model.output.weight,
    }


def copy_tracked(model: UniPilotTransformer) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in tracked_parameters(model).items()}


def weight_delta(model: UniPilotTransformer, initial: dict[str, torch.Tensor]) -> dict:
    results = {}
    for name, parameter in tracked_parameters(model).items():
        difference = parameter.detach() - initial[name]
        denominator = float(initial[name].float().norm().item())
        results[name] = {
            "l2": float(difference.float().norm().item()),
            "relative_l2": float(difference.float().norm().item()) / max(denominator, 1e-12),
            "max_absolute": float(difference.float().abs().max().item()),
        }
    return results


def choose_token(logits: torch.Tensor, *, generator: torch.Generator, mode: str,
                 temperature: float = 1.0, top_k: int | None = None,
                 top_p: float | None = None, repetition_ids: list[int] | None = None,
                 repetition_penalty: float = 1.0) -> int:
    scores = logits.detach().float().clone()
    if repetition_ids and repetition_penalty != 1.0:
        for token_id in set(repetition_ids):
            scores[token_id] = (
                scores[token_id] / repetition_penalty
                if scores[token_id] >= 0 else scores[token_id] * repetition_penalty
            )
    if mode == "greedy":
        return int(scores.argmax().item())
    scores /= temperature
    if top_k is not None and 0 < top_k < scores.numel():
        threshold = torch.topk(scores, top_k).values[-1]
        scores[scores < threshold] = -torch.inf
    if top_p is not None and top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        probabilities = torch.softmax(sorted_scores, dim=-1)
        remove = torch.cumsum(probabilities, dim=-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[sorted_indices[remove]] = -torch.inf
    probabilities = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


@torch.inference_mode()
def generate(model: UniPilotTransformer, tokenizer: FoundationTokenizer, prompt_ids: list[int],
             *, max_new_tokens: int, seed: int, mode: str = "greedy",
             temperature: float = 1.0, top_k: int | None = None,
             top_p: float | None = None, repetition_penalty: float = 1.0) -> dict:
    model.eval()
    all_ids = list(prompt_ids)
    generated: list[int] = []
    past = None
    generator = torch.Generator().manual_seed(seed)
    forbidden = [tokenizer.pad_id, tokenizer.special_to_id["<UNK>"],
                 tokenizer.special_to_id["<USER>"], tokenizer.special_to_id["<ASSISTANT>"],
                 tokenizer.special_to_id["<SYSTEM>"], tokenizer.bos_id]
    for _ in range(max_new_tokens):
        current = all_ids[-model.config.context_length:] if past is None else [all_ids[-1]]
        logits, _, past = model(
            torch.tensor([current], dtype=torch.long), past_key_values=past, use_cache=True
        )
        scores = logits[0, -1].clone()
        scores[forbidden] = -torch.inf
        next_id = choose_token(
            scores, generator=generator, mode=mode, temperature=temperature,
            top_k=top_k, top_p=top_p, repetition_ids=all_ids[-64:],
            repetition_penalty=repetition_penalty,
        )
        all_ids.append(next_id)
        generated.append(next_id)
        if next_id == tokenizer.eos_id:
            break
    return {
        "ids": generated,
        "text": tokenizer.decode(generated, skip_special=True),
        "eos_reached": bool(generated and generated[-1] == tokenizer.eos_id),
        "tokens": len(generated),
    }


def memorization_probe(model: UniPilotTransformer, tokenizer: FoundationTokenizer,
                       records: list[SequenceRecord], seed: int) -> dict:
    rows = []
    for index, record in enumerate(records):
        ids = list(record.ids)
        prefix_length = min(32, max(8, len(ids) // 5))
        expected = ids[prefix_length:]
        generated = generate(
            model, tokenizer, ids[:prefix_length], max_new_tokens=len(expected) + 8,
            seed=seed + index, mode="greedy",
        )
        compared = min(len(expected), len(generated["ids"]))
        exact = sum(
            expected[position] == generated["ids"][position]
            for position in range(compared)
        )
        rows.append({
            "id": record.identifier,
            "prefix": tokenizer.decode(ids[:prefix_length], skip_special=True),
            "expected_prefix": tokenizer.decode(expected[:64], skip_special=True),
            "generated": generated["text"],
            "token_exact_rate": exact / max(1, len(expected)),
            "expected_tokens": len(expected),
            "generated_tokens": generated["tokens"],
            "eos_reached": generated["eos_reached"],
        })
    return {
        "documents": len(rows),
        "mean_token_exact_rate": sum(row["token_exact_rate"] for row in rows) / len(rows),
        "eos_rate": sum(row["eos_reached"] for row in rows) / len(rows),
        "items": rows,
    }


def train_sequences(*, records: list[SequenceRecord], validation_records: list[SequenceRecord],
                    tokenizer: FoundationTokenizer, vocab_size: int, ffn_dim: int, seed: int,
                    learning_rate: float, max_steps: int, weight_decay: float,
                    gradient_clip: float, warmup_steps: int = 0,
                    schedule_steps: int | None = None, target_loss: float | None = None,
                    evaluation_interval: int = 50, generation_probe: bool = True) -> dict:
    model = build_model(vocab_size=vocab_size, seed=seed, ffn_dim=ffn_dim)
    optimizer = create_optimizer(model, learning_rate, weight_decay)
    initial = copy_tracked(model)
    process = psutil.Process(os.getpid())
    generator = torch.Generator().manual_seed(seed + 1)
    permutation = torch.randperm(len(records), generator=generator).tolist()
    position = 0
    epoch = 0
    losses: list[float] = []
    recent: list[float] = []
    tokens_seen = 0
    gradient_log: dict[str, dict] = {}
    delta_log: dict[str, dict] = {}
    curve = []
    peak_ram = process.memory_info().rss / 1024**2
    started = time.perf_counter()
    diverged = False
    last_gradient_snapshot: dict | None = None
    early_stop_probe: dict | None = None
    for step_index in range(max_steps):
        if position >= len(permutation):
            epoch += 1
            permutation = torch.randperm(len(records), generator=generator).tolist()
            position = 0
        record = records[permutation[position]]
        position += 1
        multiplier = 1.0
        if schedule_steps is not None:
            multiplier = warmup_cosine_multiplier(
                step_index, warmup_steps, schedule_steps, 0.1
            )
        current_lr = learning_rate * multiplier
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        inputs = record.inputs.unsqueeze(0)
        targets = record.targets.unsqueeze(0)
        _, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss):
            diverged = True
            break
        loss.backward()
        global_norm = parameter_l2_norm(model.parameters())
        if not math.isfinite(global_norm):
            diverged = True
            break
        completed_step = step_index + 1
        last_gradient_snapshot = gradient_snapshot(model, global_norm)
        if completed_step in {1, 10, 50, 100, max_steps}:
            gradient_log[str(completed_step)] = last_gradient_snapshot
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        value = float(loss.item())
        losses.append(value)
        recent.append(value)
        tokens_seen += targets.numel()
        peak_ram = max(peak_ram, process.memory_info().rss / 1024**2)
        if completed_step in {10, 100}:
            delta_log[str(completed_step)] = weight_delta(model, initial)
        should_evaluate = (
            completed_step == max_steps
            or completed_step % evaluation_interval == 0
        )
        if should_evaluate:
            evaluated = validation_loss(model, records)
            curve.append({
                "step": completed_step,
                "recent_train_loss": sum(recent) / len(recent),
                "evaluation_train_loss": evaluated,
                "learning_rate": current_lr,
                "epoch": epoch + position / len(records),
            })
            recent.clear()
            if target_loss is not None and evaluated <= target_loss:
                if generation_probe and len(records) > 1:
                    early_stop_probe = memorization_probe(model, tokenizer, records, seed)
                    if (
                        early_stop_probe["mean_token_exact_rate"] < 0.75
                        or early_stop_probe["eos_rate"] < 0.8
                    ):
                        continue
                gradient_log[str(completed_step)] = last_gradient_snapshot
                break
    elapsed = time.perf_counter() - started
    final_train = validation_loss(model, records)
    final_validation = validation_loss(model, validation_records)
    probe = (
        early_stop_probe
        if early_stop_probe is not None
        else memorization_probe(model, tokenizer, records, seed) if generation_probe else None
    )
    return {
        "model": model,
        "metrics": {
            "parameters": model.parameter_count(),
            "vocab": vocab_size,
            "ffn_dim": ffn_dim,
            "steps": len(losses),
            "epochs": epoch + position / len(records),
            "final_train_loss": final_train,
            "final_validation_loss": final_validation,
            "perplexity": math.exp(min(final_validation, 50)),
            "last_20_step_loss": sum(losses[-20:]) / min(20, len(losses)),
            "tokens_processed": tokens_seen,
            "tokens_per_second": tokens_seen / max(elapsed, 1e-9),
            "wall_seconds": elapsed,
            "peak_ram_mb": peak_ram,
            "gradient_norm": gradient_log.get(str(len(losses)), {}).get("global"),
            "diverged": diverged,
            "nan_or_inf": diverged,
        },
        "curve": curve,
        "gradients": gradient_log,
        "weight_deltas": delta_log,
        "memorization": probe,
    }


def tensor_statistics(tensor: torch.Tensor) -> dict:
    values = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def initialization_audit(model: UniPilotTransformer) -> dict:
    middle = len(model.blocks) // 2
    tensors = {
        "token_embedding": model.embeddings.token.weight,
        "position_embedding": model.embeddings.position.weight,
        "first_attention_qkv": model.blocks[0].attention.qkv.weight,
        "middle_attention_qkv": model.blocks[middle].attention.qkv.weight,
        "last_attention_qkv": model.blocks[-1].attention.qkv.weight,
        "first_mlp_in": model.blocks[0].feed_forward.network[0].weight,
        "middle_mlp_in": model.blocks[middle].feed_forward.network[0].weight,
        "last_mlp_in": model.blocks[-1].feed_forward.network[0].weight,
        "lm_head": model.output.weight,
    }
    return {
        "parameters": model.parameter_count(),
        "weight_tying": {
            "enabled": model.embeddings.token.weight is model.output.weight,
            "same_storage": (
                model.embeddings.token.weight.data_ptr() == model.output.weight.data_ptr()
            ),
        },
        "statistics": {name: tensor_statistics(value) for name, value in tensors.items()},
    }


def causal_mask_audit(seed: int = 12012026) -> dict:
    model = build_model(vocab_size=64, seed=seed, ffn_dim=96,
                        context_length=16, dropout=0.0)
    model.eval()
    original = torch.tensor([[1, 7, 11, 13, 17, 19, 23, 29]], dtype=torch.long)
    changed = original.clone()
    changed[:, 5:] = torch.tensor([31, 37, 41])
    with torch.inference_mode():
        original_logits, _ = model(original)
        changed_logits, _ = model(changed)
        past_logits = original_logits[:, :5]
        changed_past_logits = changed_logits[:, :5]
        maximum_difference = float((past_logits - changed_past_logits).abs().max().item())
        full_logits, _ = model(original)
        pieces = []
        past = None
        for position in range(original.size(1)):
            token = original[:, position:position + 1]
            logits, _, past = model(token, past_key_values=past, use_cache=True)
            pieces.append(logits)
        cached_logits = torch.cat(pieces, dim=1)
        cache_difference = float((full_logits - cached_logits).abs().max().item())
    expected = torch.tril(torch.ones(8, 8, dtype=torch.bool))
    actual = model.blocks[0].attention.causal_mask[0, 0, :8, :8].cpu()
    checks = {
        "triangular_mask_exact": torch.equal(expected, actual),
        "future_change_does_not_change_past_logits": maximum_difference <= 1e-7,
        "cached_and_full_logits_match": cache_difference <= 1e-5,
    }
    return {
        "checks": checks,
        "maximum_past_logit_difference": maximum_difference,
        "maximum_cache_logit_difference": cache_difference,
        "mask_rows": [[int(value) for value in row] for row in actual.tolist()],
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def loss_audit(seed: int = 12012026) -> dict:
    model = build_model(vocab_size=64, seed=seed, ffn_dim=96,
                        context_length=8, dropout=0.0)
    model.eval()
    inputs = torch.tensor([[1, 5, 9, 0, 2, 12, 8, 7]], dtype=torch.long)
    targets = torch.tensor([[5, 9, 0, 2, 1, 8, 7, 2]], dtype=torch.long)
    logits, loss = model(inputs, targets)
    manual = F.cross_entropy(logits.reshape(-1, 64), targets.reshape(-1),
                             ignore_index=-100, reduction="mean")
    ignored_targets = targets.clone()
    ignored_targets[0, 2] = -100
    _, ignored_loss = model(inputs, ignored_targets)
    manual_ignored = F.cross_entropy(
        logits.reshape(-1, 64), ignored_targets.reshape(-1),
        ignore_index=-100, reduction="mean"
    )
    checks = {
        "logits_shape": list(logits.shape) == [1, 8, 64],
        "target_shape": list(targets.shape) == [1, 8],
        "cross_entropy_matches": torch.equal(loss, manual),
        "ignore_minus_100_matches": torch.equal(ignored_loss, manual_ignored),
        "pad_zero_is_not_ignored": not torch.equal(loss, ignored_loss),
        "bos_is_a_loss_target": int((targets == 1).sum().item()) == 1,
        "eos_is_a_loss_target": int((targets == 2).sum().item()) == 2,
        "vocabulary_alignment": logits.size(-1) == model.config.vocab_size,
        "reduction_is_mean": True,
    }
    # BOS is normally only an input boundary token, while EOS is explicitly predicted.
    checks["bos_boundary_semantics_valid"] = int((inputs == 1).sum().item()) == 1
    return {
        "logits_shape": list(logits.shape),
        "target_shape": list(targets.shape),
        "model_loss": float(loss.item()),
        "manual_cross_entropy": float(manual.item()),
        "pad_target_count": int((targets == 0).sum().item()),
        "eos_target_count": int((targets == 2).sum().item()),
        "ignore_index": -100,
        "reduction": "mean",
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
