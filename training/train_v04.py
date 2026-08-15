from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import subprocess
import time

import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from tokenizer.tokenizer import BPETokenizer
from training.dataset_v03 import CurriculumDataset, dynamic_collate
from training.optimizer import create_optimizer
from training.scheduler import warmup_cosine_multiplier


def eos_weighted_loss(logits: torch.Tensor, targets: torch.Tensor, eos_id: int, eos_weight: float) -> torch.Tensor:
    flat_targets = targets.reshape(-1)
    per_token = F.cross_entropy(logits.reshape(-1, logits.size(-1)), flat_targets, ignore_index=-100, reduction="none")
    valid = flat_targets != -100
    weights = torch.ones_like(per_token)
    weights[flat_targets == eos_id] = eos_weight
    return (per_token[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1)


def git_commit() -> str:
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: return "unknown"


@torch.inference_mode()
def validation(model, loader, device, eos_id, eos_weight, batches=30):
    model.eval(); values = []
    for index, (inputs, targets, _, _) in enumerate(loader):
        if index >= batches: break
        logits, _ = model(inputs.to(device)); values.append(eos_weighted_loss(logits, targets.to(device), eos_id, eos_weight).item())
    model.train(); return sum(values) / max(1, len(values))


def save(path: Path, model, optimizer, step, loss, settings, args, stats):
    manifest = {"model": "UniPilot Mini v0.4", "parameters": model.parameter_count(), "experiment_id": args.experiment_id,
                "base_checkpoint": args.base_checkpoint, "dataset_version": settings["dataset_version"],
                "evaluation_version": settings["evaluation_version"], "eos_weight": args.eos_weight, "step": step,
                "seed": settings["seed"], "generation": settings["generation"], "git_commit": git_commit(),
                "model_config": model.config.to_dict(), "optimizer": {"name": "AdamW", "learning_rate": optimizer.param_groups[0]["lr"],
                "weight_decay": settings["weight_decay"]}, "metrics": stats}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "step": step, "loss": loss,
                "config": model.config.to_dict(), "tokenizer_version": "unipilot-byte-bpe-v02-512", "v04_manifest": manifest}, path)
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-v04.json"); parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--eos-weight", type=float, choices=[1.0, 1.5, 2.0], required=True)
    parser.add_argument("--max-steps", type=int, default=500); parser.add_argument("--base-checkpoint")
    parser.add_argument("--resume-run"); parser.add_argument("--output-dir", required=True); parser.add_argument("--device", default="auto")
    args = parser.parse_args(); settings = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.max_steps > 2000: raise ValueError("v0.4 experiments are capped at 2000 steps")
    args.base_checkpoint = args.base_checkpoint or settings["base_checkpoint"]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    random.seed(settings["seed"]); torch.manual_seed(settings["seed"])
    tokenizer = BPETokenizer.load(settings["tokenizer"]); config = ModelConfig(**settings["model"])
    model = UniPilotTransformer(config).to(device); source = args.resume_run or args.base_checkpoint
    payload = torch.load(source, map_location=device, weights_only=False); model.load_state_dict(payload["model_state"])
    optimizer = create_optimizer(model, settings["learning_rate"], settings["weight_decay"]); step = 0
    if args.resume_run:
        optimizer.load_state_dict(payload["optimizer_state"]); step = int(payload["step"])
        old = payload.get("v04_manifest", {})
        if old.get("eos_weight") != args.eos_weight: raise ValueError("resume EOS weight mismatch")
    train = CurriculumDataset(Path(settings["dataset"]) / "train.jsonl", tokenizer, config.context_length, True)
    valid = CurriculumDataset(Path(settings["dataset"]) / "validation.jsonl", tokenizer, config.context_length, True)
    generator = torch.Generator().manual_seed(settings["seed"])
    loader = DataLoader(train, batch_size=settings["batch_size"], shuffle=True, collate_fn=dynamic_collate, generator=generator)
    val_loader = DataLoader(valid, batch_size=1, shuffle=False, collate_fn=dynamic_collate); iterator = iter(loader)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True); log = output / "training_log.csv"
    fields = ["step", "loss", "validation_loss", "learning_rate", "gradient_norm", "gradient_clipping_count", "nan_count", "inf_count", "tokens_per_second", "step_time_seconds", "memory_usage_mb"]
    mode = "a" if args.resume_run and log.exists() else "w"; clipping = nan = inf = 0; recent = []; process = psutil.Process()
    with log.open(mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if mode == "w": writer.writeheader()
        while step < args.max_steps:
            try: inputs, targets, _, _ = next(iterator)
            except StopIteration: iterator = iter(loader); inputs, targets, _, _ = next(iterator)
            local = step; lr = settings["learning_rate"] * warmup_cosine_multiplier(local, settings["warmup_steps"], 2000, min_ratio=.1)
            for group in optimizer.param_groups: group["lr"] = lr
            started = time.perf_counter(); optimizer.zero_grad(set_to_none=True); inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs); loss = eos_weighted_loss(logits, targets, tokenizer.eos_id, args.eos_weight)
            if torch.isnan(loss): nan += 1; raise RuntimeError("NaN loss")
            if torch.isinf(loss): inf += 1; raise RuntimeError("Inf loss")
            loss.backward(); norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])); clipping += norm > settings["gradient_clip"]
            optimizer.step(); step += 1; elapsed = time.perf_counter() - started; recent.append(loss.item()); recent = recent[-100:]
            if step in settings["checkpoint_steps"] or step == args.max_steps:
                val = validation(model, val_loader, device, tokenizer.eos_id, args.eos_weight)
                stats = {"step": step, "loss": sum(recent) / len(recent), "validation_loss": val, "learning_rate": lr,
                         "gradient_norm": norm, "gradient_clipping_count": clipping, "nan_count": nan, "inf_count": inf,
                         "tokens_per_second": int((targets != -100).sum()) / max(elapsed, 1e-9), "step_time_seconds": elapsed,
                         "memory_usage_mb": process.memory_info().rss / 1024**2}
                writer.writerow(stats); file.flush(); print(json.dumps(stats), flush=True)
                model.config.model_name = f"UniPilot Mini v0.4-{args.experiment_id}-{step}"
                save(output / f"checkpoint-step-{step}.pt", model, optimizer, step, val, settings, args, stats)


if __name__ == "__main__": main()
