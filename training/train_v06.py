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
from torch.utils.data import DataLoader

from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from tokenizer.tokenizer import BPETokenizer
from training.dataset_v03 import CurriculumDataset, dynamic_collate
from training.optimizer import create_optimizer
from training.scheduler import warmup_cosine_multiplier
from training.train_v04 import eos_weighted_loss


ALLOWED_STOPS = (500, 1000, 2000, 5000, 10000)


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


def schedule_weights(settings: dict, step: int) -> dict[str, float]:
    # Complete A-E introduction within the first 500-step gate. Later gates use
    # the final replay-heavy mix instead of postponing accuracy examples.
    fraction = min(1.0, (step + 1) / 500)
    for entry in settings["stage_schedule"]:
        if fraction <= entry["until_fraction"]:
            return entry["weights"]
    return settings["stage_schedule"][-1]["weights"]


def choose_stage(rng: random.Random, weights: dict[str, float]) -> str:
    point = rng.random() * sum(weights.values())
    total = 0.0
    for stage, weight in weights.items():
        total += weight
        if point <= total:
            return stage
    return next(reversed(weights))


def save(path: Path, model, optimizer, step: int, loss: float, settings: dict, args, stats: dict) -> None:
    manifest = {
        "model": "UniPilot Mini v0.6 candidate", "parameters": model.parameter_count(),
        "stage": "quality-preserving curriculum A-E", "experiment_id": args.experiment_id,
        "base_checkpoint": settings["base_checkpoint"], "dataset_version": settings["dataset_version"],
        "evaluation_version": settings["evaluation_version"], "eos_weight": args.eos_weight,
        "step": step, "seed": settings["seed"], "git_commit": git_commit(),
        "model_config": model.config.to_dict(), "production_promoted": False, "metrics": stats,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "step": step,
        "loss": loss, "config": model.config.to_dict(), "tokenizer_version": "unipilot-byte-bpe-v02-512",
        "v06_manifest": manifest,
    }, path)
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-v06.json")
    parser.add_argument("--experiment-id", default="v06-preserve")
    parser.add_argument("--max-steps", type=int, choices=ALLOWED_STOPS, required=True)
    parser.add_argument("--resume-run")
    parser.add_argument("--output-dir", default="checkpoints/v06-preserve")
    parser.add_argument("--eos-weight", type=float, default=1.5)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    rng = random.Random(settings["seed"])
    tokenizer = BPETokenizer.load(settings["tokenizer"])
    config = ModelConfig(**settings["model"])
    model = UniPilotTransformer(config).to(device)
    source = args.resume_run or settings["base_checkpoint"]
    payload = torch.load(source, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    optimizer = create_optimizer(model, settings["learning_rate"], settings["weight_decay"])
    step = 0
    if args.resume_run:
        if "optimizer_state" not in payload:
            raise ValueError("resume checkpoint has no optimizer_state")
        optimizer.load_state_dict(payload["optimizer_state"])
        step = int(payload["step"])
        if step >= args.max_steps:
            raise ValueError("resume checkpoint step must be below max-steps")
    del payload

    loaders, iterators = {}, {}
    for stage in "ABCDE":
        dataset = CurriculumDataset(Path(settings["dataset"]) / "stages" / f"stage_{stage.lower()}.jsonl",
                                    tokenizer, config.context_length, True)
        if not len(dataset):
            raise ValueError(f"curriculum stage {stage} is empty")
        generator = torch.Generator().manual_seed(settings["seed"] + ord(stage))
        loaders[stage] = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True,
                                    collate_fn=dynamic_collate, generator=generator)
        iterators[stage] = iter(loaders[stage])
    valid = CurriculumDataset(Path(settings["dataset"]) / "instruction" / "validation.jsonl",
                              tokenizer, config.context_length, True)
    val_loader = DataLoader(valid, batch_size=1, shuffle=False, collate_fn=dynamic_collate)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log = output / "training_log.csv"
    fields = ["step", "loss", "validation_loss", "learning_rate", "gradient_norm", "gradient_clipping_count",
              "nan_count", "inf_count", "tokens_per_second", "step_time_seconds", "memory_usage_mb", "stage_counts"]
    mode = "a" if args.resume_run and log.exists() else "w"
    clipping = nan = inf = 0
    recent: list[float] = []
    stage_counts = CounterLike()
    process = psutil.Process()
    with log.open(mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        while step < args.max_steps:
            weights = schedule_weights(settings, step)
            stage = choose_stage(rng, weights)
            try:
                inputs, targets, _, _ = next(iterators[stage])
            except StopIteration:
                iterators[stage] = iter(loaders[stage])
                inputs, targets, _, _ = next(iterators[stage])
            stage_counts[stage] += 1
            lr = settings["learning_rate"] * warmup_cosine_multiplier(
                step, settings["warmup_steps"], 10000, min_ratio=0.2)
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
                stats = {
                    "step": step, "loss": sum(recent) / len(recent), "validation_loss": val, "learning_rate": lr,
                    "gradient_norm": norm, "gradient_clipping_count": clipping, "nan_count": nan, "inf_count": inf,
                    "tokens_per_second": int((targets != -100).sum()) / max(elapsed, 1e-9),
                    "step_time_seconds": elapsed, "memory_usage_mb": process.memory_info().rss / 1024**2,
                    "stage_counts": dict(stage_counts),
                }
                writer.writerow({**stats, "stage_counts": json.dumps(stats["stage_counts"], sort_keys=True)})
                file.flush()
                print(json.dumps(stats), flush=True)
                model.config.model_name = f"UniPilot Mini v0.6-{args.experiment_id}-{step}"
                save(output / f"checkpoint-step-{step}.pt", model, optimizer, step, val, settings, args, stats)


class CounterLike(dict):
    def __missing__(self, key):
        return 0


if __name__ == "__main__":
    main()
