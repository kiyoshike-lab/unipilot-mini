from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import subprocess
import time

import psutil
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from tokenizer.tokenizer import BPETokenizer
from training.dataset_v03 import CurriculumDataset, dynamic_collate
from training.optimizer import create_optimizer
from training.scheduler import warmup_cosine_multiplier
from training.train_v04 import eos_weighted_loss


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


@torch.inference_mode()
def validation(model, loader, device, eos_id, eos_weight) -> float:
    model.eval()
    values = []
    for inputs, targets, _, _ in loader:
        logits, _ = model(inputs.to(device))
        values.append(eos_weighted_loss(logits, targets.to(device), eos_id, eos_weight).item())
    model.train()
    return sum(values) / max(1, len(values))


def save(path: Path, model, optimizer, step: int, loss: float, settings: dict, args, stats: dict) -> None:
    manifest = {
        "model": "UniPilot Mini v0.7 candidate", "parameters": model.parameter_count(),
        "stage": "context-grounded direct answer", "experiment_id": args.experiment_id,
        "base_checkpoint": settings["base_checkpoint"], "dataset_version": settings["dataset_version"],
        "evaluation_version": settings["evaluation_version"], "step": step, "seed": settings["seed"],
        "git_commit": git_commit(), "model_config": model.config.to_dict(), "production_promoted": False,
        "pipeline_required": True, "metrics": stats,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "step": step,
                "loss": loss, "config": model.config.to_dict(), "tokenizer_version": "unipilot-byte-bpe-v02-512",
                "v07_manifest": manifest}, path)
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-v07.json")
    parser.add_argument("--experiment-id", default="grounded")
    parser.add_argument("--max-steps", type=int, choices=(500, 1000), required=True)
    parser.add_argument("--resume-run")
    parser.add_argument("--output-dir", default="checkpoints/v07-grounded")
    parser.add_argument("--eos-weight", type=float, default=1.5)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    tokenizer = BPETokenizer.load(settings["tokenizer"])
    config = ModelConfig(**settings["model"])
    model = UniPilotTransformer(config).to(device)
    source = args.resume_run or settings["base_checkpoint"]
    payload = torch.load(source, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    optimizer = create_optimizer(model, settings["learning_rate"], settings["weight_decay"])
    step = 0
    if args.resume_run:
        optimizer.load_state_dict(payload["optimizer_state"])
        step = int(payload["step"])
    del payload
    direct = CurriculumDataset(Path(settings["dataset"]) / "train.jsonl", tokenizer, config.context_length, True)
    replay = CurriculumDataset(settings["replay_dataset"], tokenizer, config.context_length, True)
    combined = ConcatDataset((direct, replay))
    ratio = settings["replay_ratio"]
    weights = [((1 - ratio) / len(direct))] * len(direct) + [(ratio / len(replay))] * len(replay)
    generator = torch.Generator().manual_seed(settings["seed"])
    sampler = WeightedRandomSampler(weights, num_samples=max(len(direct), args.max_steps), replacement=True, generator=generator)
    loader = DataLoader(combined, batch_size=settings["batch_size"], sampler=sampler, collate_fn=dynamic_collate)
    valid = CurriculumDataset(Path(settings["dataset"]) / "validation.jsonl", tokenizer, config.context_length, True)
    val_loader = DataLoader(valid, batch_size=1, shuffle=False, collate_fn=dynamic_collate)
    iterator = iter(loader)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log = output / "training_log.csv"
    fields = ["step", "loss", "validation_loss", "learning_rate", "gradient_norm", "gradient_clipping_count",
              "nan_count", "inf_count", "tokens_per_second", "step_time_seconds", "memory_usage_mb"]
    mode = "a" if args.resume_run and log.exists() else "w"
    clipping = nan = inf = 0
    recent = []
    process = psutil.Process()
    with log.open(mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        while step < args.max_steps:
            try:
                inputs, targets, _, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                inputs, targets, _, _ = next(iterator)
            lr = settings["learning_rate"] * warmup_cosine_multiplier(step, settings["warmup_steps"], 1000, min_ratio=0.2)
            for group in optimizer.param_groups:
                group["lr"] = lr
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            loss = eos_weighted_loss(logits, targets, tokenizer.eos_id, args.eos_weight)
            if torch.isnan(loss):
                nan += 1
                raise RuntimeError("NaN loss")
            if torch.isinf(loss):
                inf += 1
                raise RuntimeError("Inf loss")
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"]))
            clipping += int(norm > settings["gradient_clip"])
            optimizer.step()
            step += 1
            elapsed = time.perf_counter() - started
            recent.append(loss.item())
            recent = recent[-100:]
            if step in settings["checkpoint_steps"] or step == args.max_steps:
                val = validation(model, val_loader, device, tokenizer.eos_id, args.eos_weight)
                stats = {"step": step, "loss": sum(recent) / len(recent), "validation_loss": val,
                         "learning_rate": lr, "gradient_norm": norm, "gradient_clipping_count": clipping,
                         "nan_count": nan, "inf_count": inf,
                         "tokens_per_second": int((targets != -100).sum()) / max(elapsed, 1e-9),
                         "step_time_seconds": elapsed, "memory_usage_mb": process.memory_info().rss / 1024**2}
                writer.writerow(stats)
                file.flush()
                print(json.dumps(stats), flush=True)
                model.config.model_name = f"UniPilot Mini v0.7-{args.experiment_id}-{step}"
                save(output / f"checkpoint-step-{step}.pt", model, optimizer, step, val, settings, args, stats)


if __name__ == "__main__":
    main()
