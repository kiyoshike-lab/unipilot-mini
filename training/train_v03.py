from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import shutil
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


def resolve_device(requested: str) -> str:
    if requested == "auto": return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    return requested


def git_commit() -> str:
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: return "unknown"


def stage_for_step(stages: list[dict], step: int) -> dict:
    for stage in stages:
        if stage["start_step"] <= step < stage["end_step"]: return stage
    return stages[-1]


def learning_rate(stage: dict, global_step: int) -> float:
    local = global_step - stage["start_step"]
    duration = stage["end_step"] - stage["start_step"]
    return stage["learning_rate"] * warmup_cosine_multiplier(local, stage["warmup_steps"], duration, min_ratio=0.1)


def make_loader(stage: dict, split: str, tokenizer, config, batch_size: int, max_records: int, seed: int):
    dataset = CurriculumDataset(Path(stage["data_dir"]) / f"{split}.jsonl", tokenizer, config.context_length,
                                stage["assistant_only_loss"], max_records=max_records)
    generator = torch.Generator().manual_seed(seed + ord(stage["name"]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == "train", num_workers=0,
                      collate_fn=dynamic_collate, generator=generator), dataset


@torch.inference_mode()
def validation_loss(model, loader, device, batches):
    model.eval(); losses = []
    for index, (inputs, targets, _, _) in enumerate(loader):
        if index >= batches: break
        _, loss = model(inputs.to(device), targets.to(device)); losses.append(loss.item())
    model.train(); return sum(losses) / max(1, len(losses))


def save_v03_checkpoint(path: Path, model, optimizer, global_step: int, loss: float, settings: dict, stage: dict,
                        initialization: str, stats: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"model": "UniPilot Mini v0.3", "parameters": model.parameter_count(), "tokenizer": "bpe-512",
                "tokenizer_version": "unipilot-byte-bpe-v02-512", "dataset": settings["dataset_version"],
                "stage": stage["name"], "step": global_step, "experiment_id": settings["experiment_id"],
                "initialization": initialization, "random_seed": settings["seed"], "model_config": model.config.to_dict(),
                "optimizer": {"name": "AdamW", "weight_decay": settings["weight_decay"], "learning_rate": optimizer.param_groups[0]["lr"]},
                "scheduler": {"name": "per-stage warmup cosine", "stage_config": stage},
                "generation": settings["generation"], "git_commit": git_commit(), "metrics": stats}
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scheduler_state": {"global_step": global_step},
                "epoch": 0, "step": global_step, "loss": loss, "config": model.config.to_dict(),
                "tokenizer_version": "unipilot-byte-bpe-v02-512", "v03_manifest": manifest}, path)
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="UniPilot Mini v0.3 curriculum trainer")
    parser.add_argument("--config", default="configs/unipilot-v03.json")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--initialization", choices=["scratch-v03", "resume-v02"], default="scratch-v03")
    parser.add_argument("--v02-checkpoint", default="checkpoints/unipilot-v02-step-1000/checkpoint-step-1000.pt")
    parser.add_argument("--resume-run", help="resume an interrupted v0.3 run and preserve its global step")
    parser.add_argument("--output-dir", default="checkpoints/v03-scratch-001")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args(); settings = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.max_steps > settings["stages"][-1]["end_step"]:
        raise ValueError("This curriculum is validated through 5000 steps. Use a dedicated extended config for longer training.")
    device = resolve_device(args.device or settings["device"]); random.seed(settings["seed"]); torch.manual_seed(settings["seed"])
    tokenizer = BPETokenizer.load(settings["tokenizer"]); config = ModelConfig(**settings["model"])
    if tokenizer.vocab_size != 512 or config.vocab_size != 512: raise ValueError("v0.3 fixes tokenizer and model vocabulary at 512")
    model = UniPilotTransformer(config).to(device); optimizer = create_optimizer(model, settings["stages"][0]["learning_rate"], settings["weight_decay"])
    amp_enabled = device == "cuda" and settings["precision"] in ["auto", "fp16"]
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    global_step = 0
    initialization = args.initialization
    source_checkpoint = args.resume_run or (args.v02_checkpoint if args.initialization == "resume-v02" else None)
    if source_checkpoint:
        payload = torch.load(source_checkpoint, map_location=device, weights_only=False); model.load_state_dict(payload["model_state"])
        if payload.get("optimizer_state"): optimizer.load_state_dict(payload["optimizer_state"])
        if args.resume_run: global_step = int(payload["step"]); initialization = payload.get("v03_manifest", {}).get("initialization", initialization)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training_log.csv"; fields = ["step", "stage", "train_loss", "stage_validation_loss", "general_validation_loss",
        "university_validation_loss", "conversation_validation_loss", "learning_rate", "gradient_norm", "gradient_clipping_count",
        "nan_count", "inf_count", "tokens_per_second", "step_time_seconds", "memory_usage_mb", "eta_seconds"]
    process = psutil.Process(); max_records = settings["max_records_per_stage"] if args.max_records is None else args.max_records
    loader_cache = {}; validation_cache = {}; iterator = None; current_stage_name = None; recent_losses = []; clipping_count = nan_count = inf_count = 0
    validation_history = []; checkpoint_steps = set(settings["checkpoint_steps"]) | {stage["end_step"] for stage in settings["stages"]}
    mode = "a" if args.resume_run and log_path.exists() else "w"
    with log_path.open(mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if mode == "w": writer.writeheader()
        while global_step < args.max_steps:
            stage = stage_for_step(settings["stages"], global_step)
            if stage["name"] != current_stage_name:
                if stage["name"] not in loader_cache:
                    train_loader, train_set = make_loader(stage, "train", tokenizer, config, settings["batch_size"], max_records, settings["seed"])
                    val_loader, val_set = make_loader(stage, "validation", tokenizer, config, settings["batch_size"], 200, settings["seed"])
                    loader_cache[stage["name"]] = (train_loader, val_loader, len(train_set), len(val_set))
                iterator = iter(loader_cache[stage["name"]][0]); current_stage_name = stage["name"]
                print(json.dumps({"event": "stage_start", "stage": stage["name"], "step": global_step,
                                  "train_samples": loader_cache[stage["name"]][2], "validation_samples": loader_cache[stage["name"]][3]}), flush=True)
            lr = learning_rate(stage, global_step)
            for group in optimizer.param_groups: group["lr"] = lr
            started = time.perf_counter(); optimizer.zero_grad(set_to_none=True); token_count = 0; micro_losses = []
            for _ in range(settings["gradient_accumulation"]):
                try: inputs, targets, _, _ = next(iterator)
                except StopIteration:
                    iterator = iter(loader_cache[stage["name"]][0]); inputs, targets, _, _ = next(iterator)
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp_enabled):
                    _, loss = model(inputs, targets); scaled = loss / settings["gradient_accumulation"]
                if torch.isnan(loss): nan_count += 1; raise RuntimeError("NaN loss detected; training stopped")
                if torch.isinf(loss): inf_count += 1; raise RuntimeError("Inf loss detected; training stopped")
                scaler.scale(scaled).backward(); micro_losses.append(loss.item()); token_count += (targets != -100).sum().item()
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"]))
            if not math.isfinite(grad_norm): raise RuntimeError("non-finite gradient norm detected; training stopped")
            if grad_norm > settings["gradient_clip"]: clipping_count += 1
            scaler.step(optimizer); scaler.update(); global_step += 1; elapsed = time.perf_counter() - started
            current_loss = sum(micro_losses) / len(micro_losses); recent_losses.append(current_loss)
            if len(recent_losses) > 100: recent_losses.pop(0)
            if global_step > 100 and current_loss > 3 * sum(recent_losses) / len(recent_losses): raise RuntimeError("loss explosion detected")
            should_eval = global_step % settings["eval_interval"] == 0 or global_step in checkpoint_steps or global_step == args.max_steps
            stats = {}
            if should_eval:
                stage_val = validation_loss(model, loader_cache[stage["name"]][1], device, settings["validation_batches"])
                separated = {}
                for check_stage in settings["stages"]:
                    if check_stage["name"] in loader_cache: val_loader = loader_cache[check_stage["name"]][1]
                    else:
                        if check_stage["name"] not in validation_cache:
                            validation_cache[check_stage["name"]], _ = make_loader(check_stage, "validation", tokenizer, config, 1, 50, settings["seed"])
                        val_loader = validation_cache[check_stage["name"]]
                    separated[check_stage["name"]] = validation_loss(model, val_loader, device, min(10, settings["validation_batches"]))
                if validation_history and stage_val > validation_history[-1] * 1.5:
                    print(json.dumps({"warning": "validation_loss_increase", "previous": validation_history[-1], "current": stage_val}), flush=True)
                validation_history.append(stage_val)
                eta = (args.max_steps - global_step) * elapsed
                stats = {"step": global_step, "stage": stage["name"], "train_loss": sum(recent_losses) / len(recent_losses),
                         "stage_validation_loss": stage_val, "general_validation_loss": separated["A"],
                         "university_validation_loss": separated["B"], "conversation_validation_loss": separated["C"],
                         "learning_rate": lr, "gradient_norm": grad_norm, "gradient_clipping_count": clipping_count,
                         "nan_count": nan_count, "inf_count": inf_count, "tokens_per_second": token_count / max(elapsed, 1e-9),
                         "step_time_seconds": elapsed, "memory_usage_mb": (torch.cuda.max_memory_allocated() if device == "cuda" else process.memory_info().rss) / 1024**2,
                         "eta_seconds": eta}
                writer.writerow(stats); file.flush(); print(json.dumps(stats), flush=True)
            if global_step in checkpoint_steps or global_step == args.max_steps:
                model.config.model_name = f"UniPilot Mini v0.3-{stage['name'].lower()}-{global_step}"
                checkpoint_dir = output / f"stage-{stage['name'].lower()}"; checkpoint_path = checkpoint_dir / f"checkpoint-step-{global_step}.pt"
                save_v03_checkpoint(checkpoint_path, model, optimizer, global_step, stats.get("stage_validation_loss", current_loss), settings, stage, initialization, stats)
                best_path = output / "best-checkpoints.json"
                best = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
                if not best.get("validation") or stats.get("stage_validation_loss", float("inf")) < best["validation"]["value"]:
                    best["validation"] = {"value": stats.get("stage_validation_loss"), "checkpoint": str(checkpoint_path)}
                best.setdefault("relevance", {"value": None, "checkpoint": None}); best.setdefault("human", {"value": None, "checkpoint": None})
                best_path.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "step": global_step, "output": str(output)}), flush=True)


if __name__ == "__main__": main()
