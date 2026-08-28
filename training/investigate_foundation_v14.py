from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import psutil
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from training.optimizer import create_optimizer
from training.scheduler import warmup_cosine_multiplier


CHECKPOINT_FORMAT = "foundation-v14-language-investigation-v1"


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


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def model_config(settings: dict, context_length: int, vocab_size: int) -> ModelConfig:
    return ModelConfig(
        model_name=settings["model_name"],
        vocab_size=vocab_size,
        context_length=context_length,
        **settings["model"],
    )


def build_common_initialized_model(
    settings: dict, context_length: int, vocab_size: int
) -> UniPilotTransformer:
    """Use the context-512 initialization and slice only positional embeddings."""
    seed = int(settings["seed"])
    seed_all(seed)
    reference = UniPilotTransformer(model_config(settings, 512, vocab_size))
    if context_length == 512:
        return reference
    seed_all(seed)
    model = UniPilotTransformer(model_config(settings, context_length, vocab_size))
    reference_state = reference.state_dict()
    target_state = model.state_dict()
    for name in target_state:
        value = reference_state[name]
        if name == "embeddings.position.weight":
            value = value[:context_length]
        if value.shape != target_state[name].shape:
            raise RuntimeError(f"common initialization shape mismatch: {name}")
        target_state[name] = value.clone()
    model.load_state_dict(target_state)
    del reference
    return model


def macro_permutation(size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(size, generator=generator)


def macro_batch(
    tokens: np.memmap, macro_indices: list[int], context_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if 512 % context_length:
        raise ValueError("context must divide the fixed 512-token macro block")
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for macro_index in macro_indices:
        start = macro_index * 512
        values = np.asarray(tokens[start:start + 513], dtype=np.int64)
        if len(values) != 513:
            raise RuntimeError("short macro block")
        for offset in range(0, 512, context_length):
            inputs.append(values[offset:offset + context_length].copy())
            targets.append(values[offset + 1:offset + context_length + 1].copy())
    return torch.from_numpy(np.stack(inputs)), torch.from_numpy(np.stack(targets))


@torch.inference_mode()
def validation_metrics(
    model: UniPilotTransformer,
    validation_tokens: np.memmap,
    probe_tokens: int,
) -> dict:
    was_training = model.training
    model.eval()
    context = model.config.context_length
    losses = 0.0
    token_count = 0
    correct = {1: 0, 5: 0, 10: 0}
    boundary_correct = 0
    boundary_count = 0
    boundary_ids = getattr(model, "_foundation_boundary_ids", set())
    for start in range(0, probe_tokens, context):
        size = min(context, probe_tokens - start)
        values = np.asarray(validation_tokens[start:start + size + 1], dtype=np.int64).copy()
        inputs = torch.from_numpy(values[:-1]).unsqueeze(0)
        targets = torch.from_numpy(values[1:]).unsqueeze(0)
        logits, loss = model(inputs, targets)
        losses += float(loss.item()) * targets.numel()
        token_count += targets.numel()
        top = torch.topk(logits, 10, dim=-1).indices
        expanded = targets.unsqueeze(-1)
        for k in correct:
            correct[k] += int((top[..., :k] == expanded).any(dim=-1).sum().item())
        if boundary_ids:
            mask = torch.zeros_like(targets, dtype=torch.bool)
            for token_id in boundary_ids:
                mask |= targets == token_id
            boundary_count += int(mask.sum().item())
            if mask.any():
                boundary_correct += int(((top[..., 0] == targets) & mask).sum().item())
    model.train(was_training)
    loss_value = losses / token_count
    return {
        "tokens": token_count,
        "loss": loss_value,
        "perplexity": math.exp(min(loss_value, 50)),
        "top_1_accuracy": correct[1] / token_count,
        "top_5_accuracy": correct[5] / token_count,
        "top_10_accuracy": correct[10] / token_count,
        "sentence_boundary_targets": boundary_count,
        "sentence_boundary_top_1_accuracy": (
            boundary_correct / boundary_count if boundary_count else 0.0
        ),
    }


def learning_rate(settings: dict, schedule: str, update_index: int) -> float:
    base = float(settings["learning_rate"])
    warmup = int(settings["warmup_steps"])
    minimum = float(settings["minimum_learning_rate_ratio"])
    if schedule == "short_cosine_250":
        multiplier = warmup_cosine_multiplier(update_index, warmup, 250, minimum)
    elif schedule == "long_cosine_1000":
        multiplier = warmup_cosine_multiplier(
            update_index, warmup, int(settings["long_horizon_steps"]), minimum
        )
    elif schedule == "constant_after_warmup20":
        multiplier = min(1.0, (update_index + 1) / warmup)
    elif schedule == "warmup50_constant":
        multiplier = min(1.0, (update_index + 1) / 50)
    elif schedule == "constant_no_warmup":
        multiplier = 1.0
    else:
        raise ValueError(f"unknown schedule: {schedule}")
    return base * multiplier


def train_experiment(
    settings: dict,
    experiment: dict,
    token_budget: int,
    output_dir: Path,
    save_model: bool,
) -> dict:
    if token_budget > int(settings["core_token_budget"]):
        raise RuntimeError("PHASE 25 forbids experiments beyond 128k tokens")
    context = int(experiment["context_length"])
    micro_batch = int(experiment["micro_batch"])
    effective_tokens = int(experiment["effective_batch_tokens"])
    if context * micro_batch != effective_tokens or effective_tokens % 512:
        raise RuntimeError("effective batch must be a whole number of shared macro blocks")
    if token_budget % effective_tokens:
        raise RuntimeError("token budget must divide by effective batch tokens")
    updates = token_budget // effective_tokens
    corpus = load_json(settings["corpus_manifest"])
    vocab = int(corpus["vocab"])
    train_path = ROOT / corpus["splits"]["train"]["path"]
    validation_path = ROOT / corpus["splits"]["validation"]["path"]
    train_tokens = np.memmap(train_path, dtype=np.uint16, mode="r")
    validation_tokens = np.memmap(validation_path, dtype=np.uint16, mode="r")
    macro_count = (len(train_tokens) - 1) // 512
    permutation = macro_permutation(macro_count, int(settings["seed"]))
    macro_per_update = effective_tokens // 512
    required_macros = updates * macro_per_update
    if required_macros > len(permutation):
        raise RuntimeError("experiment token budget exceeds one macro permutation")

    torch.set_num_threads(int(settings["cpu_threads"]))
    process = psutil.Process(os.getpid())
    baseline_ram = process.memory_info().rss / 1024**2
    model = build_common_initialized_model(settings, context, vocab)
    optimizer = create_optimizer(
        model, float(settings["learning_rate"]), float(settings["weight_decay"])
    )
    peak_ram = process.memory_info().rss / 1024**2
    metric_updates = {
        value for value in settings["metric_updates"] if int(value) <= updates
    } | {0, updates}
    history: list[dict] = []
    model._foundation_boundary_ids = set(experiment.get("boundary_token_ids", []))
    initial_validation = validation_metrics(
        model, validation_tokens, int(settings["validation_probe_tokens"])
    )
    history.append({
        "update": 0,
        "tokens_processed": 0,
        "train_loss": None,
        "learning_rate": learning_rate(settings, experiment["schedule"], 0),
        "gradient_norm": None,
        "tokens_per_second": None,
        "peak_ram_mb": peak_ram,
        **{f"validation_{key}": value for key, value in initial_validation.items()},
    })
    recent_losses: list[float] = []
    recent_tokens = 0
    recent_seconds = 0.0
    started_all = time.perf_counter()
    for update_index in range(updates):
        offset = update_index * macro_per_update
        macro_indices = [
            int(value) for value in permutation[offset:offset + macro_per_update].tolist()
        ]
        inputs, targets = macro_batch(train_tokens, macro_indices, context)
        if inputs.shape != (micro_batch, context):
            raise RuntimeError(
                f"micro batch shape mismatch: {tuple(inputs.shape)} != {(micro_batch, context)}"
            )
        lr = learning_rate(settings, experiment["schedule"], update_index)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        _, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss in {experiment['name']}")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(settings["gradient_clip"])
        ))
        if not math.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite gradient in {experiment['name']}")
        optimizer.step()
        elapsed = time.perf_counter() - started
        completed = update_index + 1
        recent_losses.append(float(loss.item()))
        recent_tokens += targets.numel()
        recent_seconds += elapsed
        peak_ram = max(peak_ram, process.memory_info().rss / 1024**2)
        if completed in metric_updates:
            validation = validation_metrics(
                model, validation_tokens, int(settings["validation_probe_tokens"])
            )
            peak_ram = max(peak_ram, process.memory_info().rss / 1024**2)
            row = {
                "update": completed,
                "tokens_processed": completed * effective_tokens,
                "train_loss": sum(recent_losses) / len(recent_losses),
                "learning_rate": lr,
                "gradient_norm": gradient_norm,
                "tokens_per_second": recent_tokens / max(recent_seconds, 1e-9),
                "peak_ram_mb": peak_ram,
                **{f"validation_{key}": value for key, value in validation.items()},
            }
            history.append(row)
            recent_losses.clear()
            recent_tokens = 0
            recent_seconds = 0.0
            print(json.dumps({"experiment": experiment["name"], **row}), flush=True)

    result = {
        "schema_version": "foundation-v14-experiment-v1",
        "status": "COMPLETED",
        "experiment": experiment,
        "scratch_start": True,
        "common_context512_initialization": True,
        "seed": settings["seed"],
        "parameters": model.parameter_count(),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": settings["learning_rate"],
            "betas": settings["optimizer_betas"],
            "epsilon": settings["optimizer_epsilon"],
            "weight_decay": settings["weight_decay"],
            "gradient_clip": settings["gradient_clip"],
        },
        "token_budget": token_budget,
        "updates": updates,
        "macro_blocks_seen": required_macros,
        "shared_macro_permutation": True,
        "history": history,
        "wall_seconds": time.perf_counter() - started_all,
        "baseline_ram_mb": baseline_ram,
        "peak_ram_mb": peak_ram,
        "peak_ram_delta_mb": peak_ram - baseline_ram,
        "nan_or_inf": False,
        "diverged": False,
        "final_blind_used": False,
        "external_ai_api": "OFF",
        "production_changed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{experiment['name']}.json"
    if save_model:
        checkpoint_path = output_dir / f"{experiment['name']}.pt"
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save({
            "checkpoint_format": CHECKPOINT_FORMAT,
            "model_state": model.state_dict(),
            "config": model.config.to_dict(),
            "experiment": experiment,
            "token_budget": token_budget,
            "updates": updates,
            "seed": settings["seed"],
            "final_blind_used": False,
            "external_ai_api": "OFF",
            "production_changed": False,
        }, temporary)
        temporary.replace(checkpoint_path)
        result["checkpoint"] = checkpoint_path.relative_to(ROOT).as_posix()
        result["checkpoint_bytes"] = checkpoint_path.stat().st_size
        result["checkpoint_sha256"] = sha256(checkpoint_path)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def experiment_by_name(settings: dict, name: str) -> dict:
    for experiment in settings["experiments"]:
        if experiment["name"] == name:
            return dict(experiment)
    raise KeyError(name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v14.json")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment")
    group.add_argument("--batch-pilot", type=int, choices=[1, 2, 4])
    parser.add_argument("--output-dir", default="checkpoints/foundation-v14-investigation")
    args = parser.parse_args()
    settings = load_json(args.config)
    output_dir = ROOT / args.output_dir
    if args.experiment:
        experiment = experiment_by_name(settings, args.experiment)
        token_budget = int(settings["core_token_budget"])
        result_path = output_dir / f"{experiment['name']}.json"
        checkpoint_path = output_dir / f"{experiment['name']}.pt"
        if result_path.exists() or checkpoint_path.exists():
            raise RuntimeError(f"refusing to overwrite experiment: {experiment['name']}")
        train_experiment(settings, experiment, token_budget, output_dir, save_model=True)
    else:
        multiplier = int(args.batch_pilot)
        pilot = settings["effective_batch_pilot"]
        experiment = {
            "name": f"effective_batch_{multiplier}x",
            "context_length": int(pilot["context_length"]),
            "micro_batch": multiplier,
            "effective_batch_tokens": 512 * multiplier,
            "schedule": pilot["schedule"],
        }
        token_budget = int(pilot["token_budget"])
        result_path = output_dir / f"{experiment['name']}.json"
        if result_path.exists():
            raise RuntimeError(f"refusing to overwrite pilot: {experiment['name']}")
        train_experiment(settings, experiment, token_budget, output_dir, save_model=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
