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

from foundation.diagnostic_transformer_v15 import DiagnosticConfig, DiagnosticTransformer
from training.investigate_foundation_v14 import macro_batch, macro_permutation
from training.optimizer import create_optimizer


CHECKPOINT_FORMAT = "foundation-v15-architecture-ablation-v1"


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


def variant(settings: dict, name: str) -> dict:
    for row in settings["ablations"]:
        if row["name"] == name:
            return row
    raise KeyError(name)


def build_config(settings: dict, variant_row: dict, vocab_size: int) -> DiagnosticConfig:
    values = dict(settings["architecture"])
    values.update(variant_row["changes"])
    return DiagnosticConfig(
        model_name=f"UniPilot Foundation v1.5 {variant_row['name']}",
        vocab_size=vocab_size,
        **values,
    )


def build_model(settings: dict, variant_row: dict, vocab_size: int) -> DiagnosticTransformer:
    seed_all(int(settings["seed"]))
    return DiagnosticTransformer(build_config(settings, variant_row, vocab_size))


@torch.inference_mode()
def validation_metrics(
    model: DiagnosticTransformer, validation_tokens: np.memmap, probe_tokens: int = 8192
) -> dict:
    model.eval()
    context = model.config.context_length
    loss_sum = 0.0
    total = 0
    correct = {1: 0, 5: 0, 10: 0}
    for start in range(0, probe_tokens, context):
        size = min(context, probe_tokens - start)
        values = np.asarray(
            validation_tokens[start:start + size + 1], dtype=np.int64
        ).copy()
        inputs = torch.from_numpy(values[:-1]).unsqueeze(0)
        targets = torch.from_numpy(values[1:]).unsqueeze(0)
        logits, loss = model(inputs, targets)
        count = targets.numel()
        loss_sum += float(loss.item()) * count
        total += count
        top = torch.topk(logits, 10, dim=-1).indices
        for k in correct:
            correct[k] += int(
                (top[..., :k] == targets.unsqueeze(-1)).any(dim=-1).sum().item()
            )
    value = loss_sum / total
    return {
        "tokens": total,
        "loss": value,
        "perplexity": math.exp(min(value, 50)),
        "top_1_accuracy": correct[1] / total,
        "top_5_accuracy": correct[5] / total,
        "top_10_accuracy": correct[10] / total,
    }


def learning_rate(settings: dict, update_index: int) -> float:
    warmup = int(settings["warmup_updates"])
    return float(settings["learning_rate"]) * min(1.0, (update_index + 1) / warmup)


def save_model_checkpoint(
    path: Path,
    model: DiagnosticTransformer,
    variant_row: dict,
    update: int,
    tokens_processed: int,
    seed: int,
) -> dict:
    payload = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "model_state": model.state_dict(),
        "config": model.config.to_dict(),
        "variant": variant_row,
        "update": update,
        "tokens_processed": tokens_processed,
        "seed": seed,
        "scratch_start": True,
        "external_ai_api": "OFF",
        "production_changed": False,
        "final_blind_used": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "update": update,
        "tokens_processed": tokens_processed,
    }


def train(settings: dict, variant_row: dict, output_dir: Path) -> dict:
    token_budget = int(settings["ablation_token_budget"])
    if token_budget != 65_536:
        raise RuntimeError("PHASE 26 ablation budget must remain 65,536 tokens")
    effective_tokens = int(settings["effective_batch_tokens"])
    updates = token_budget // effective_tokens
    corpus = load_json(settings["corpus_manifest"])
    train_tokens = np.memmap(
        ROOT / corpus["splits"]["train"]["path"], dtype=np.uint16, mode="r"
    )
    validation_tokens = np.memmap(
        ROOT / corpus["splits"]["validation"]["path"], dtype=np.uint16, mode="r"
    )
    permutation = macro_permutation(
        (len(train_tokens) - 1) // 512, int(settings["seed"])
    )
    torch.set_num_threads(int(settings["cpu_threads"]))
    process = psutil.Process(os.getpid())
    baseline_ram = process.memory_info().rss / 1024**2
    model = build_model(settings, variant_row, int(corpus["vocab"]))
    optimizer = create_optimizer(
        model, float(settings["learning_rate"]), float(settings["weight_decay"])
    )
    peak_ram = process.memory_info().rss / 1024**2
    history = []
    checkpoints = []
    metric_updates = set(settings["ablation_metric_updates"]) | {0, updates}
    current_variant = variant_row["name"] == "current_preln_gelu_tied"
    if current_variant:
        checkpoints.append(save_model_checkpoint(
            output_dir / f"{variant_row['name']}-update-0.pt",
            model, variant_row, 0, 0, int(settings["seed"]),
        ))
    initial = validation_metrics(model, validation_tokens)
    history.append({
        "update": 0,
        "tokens_processed": 0,
        "learning_rate": learning_rate(settings, 0),
        "train_loss": None,
        "gradient_norm": None,
        "tokens_per_second": None,
        "peak_ram_mb": peak_ram,
        "validation": initial,
    })
    recent_losses = []
    recent_tokens = 0
    recent_seconds = 0.0
    started_all = time.perf_counter()
    for update_index in range(updates):
        macro_index = int(permutation[update_index].item())
        inputs, targets = macro_batch(train_tokens, [macro_index], 512)
        lr = learning_rate(settings, update_index)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        _, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite ablation loss: {variant_row['name']}")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(settings["gradient_clip"])
        ))
        if not math.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite gradient: {variant_row['name']}")
        optimizer.step()
        elapsed = time.perf_counter() - started
        completed = update_index + 1
        recent_losses.append(float(loss.item()))
        recent_tokens += targets.numel()
        recent_seconds += elapsed
        peak_ram = max(peak_ram, process.memory_info().rss / 1024**2)
        if current_variant and completed in {10, 100}:
            checkpoints.append(save_model_checkpoint(
                output_dir / f"{variant_row['name']}-update-{completed}.pt",
                model, variant_row, completed, completed * effective_tokens,
                int(settings["seed"]),
            ))
        if completed in metric_updates:
            evaluated = validation_metrics(model, validation_tokens)
            row = {
                "update": completed,
                "tokens_processed": completed * effective_tokens,
                "learning_rate": lr,
                "train_loss": sum(recent_losses) / len(recent_losses),
                "gradient_norm": gradient_norm,
                "tokens_per_second": recent_tokens / max(recent_seconds, 1e-9),
                "peak_ram_mb": peak_ram,
                "validation": evaluated,
            }
            history.append(row)
            recent_losses.clear()
            recent_tokens = 0
            recent_seconds = 0.0
            print(json.dumps({"variant": variant_row["name"], **row}), flush=True)
    final_checkpoint = save_model_checkpoint(
        output_dir / f"{variant_row['name']}-final.pt",
        model, variant_row, updates, token_budget, int(settings["seed"]),
    )
    if current_variant:
        checkpoints.append(final_checkpoint)
    result = {
        "schema_version": "foundation-v15-ablation-result-v1",
        "status": "COMPLETED",
        "name": variant_row["name"],
        "changes": variant_row["changes"],
        "config": model.config.to_dict(),
        "parameters": model.parameter_count(),
        "parameter_breakdown": model.parameter_breakdown(),
        "token_budget": token_budget,
        "updates": updates,
        "effective_batch_tokens": effective_tokens,
        "same_training_macroblocks": True,
        "scratch_start": True,
        "history": history,
        "final_checkpoint": final_checkpoint,
        "diagnostic_checkpoints": checkpoints,
        "wall_seconds": time.perf_counter() - started_all,
        "baseline_ram_mb": baseline_ram,
        "peak_ram_mb": peak_ram,
        "nan_or_inf": False,
        "diverged": False,
        "external_ai_api": "OFF",
        "production_changed": False,
        "final_blind_used": False,
    }
    result_path = output_dir / f"{variant_row['name']}.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v15.json")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-dir", default="checkpoints/foundation-v15-architecture-audit")
    args = parser.parse_args()
    settings = load_json(args.config)
    variant_row = variant(settings, args.variant)
    output_dir = ROOT / args.output_dir
    result_path = output_dir / f"{variant_row['name']}.json"
    if result_path.exists() or (output_dir / f"{variant_row['name']}-final.pt").exists():
        raise RuntimeError(f"refusing to overwrite v1.5 ablation: {variant_row['name']}")
    train(settings, variant_row, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
