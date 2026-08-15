from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import time

import psutil
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from tokenizer.tokenizer import BPETokenizer
from training.checkpoint import save_checkpoint
from training.dataset import V02LanguageModelDataset
from training.optimizer import create_optimizer
from training.scheduler import warmup_cosine_multiplier


def resolve_device(requested: str) -> str:
    if requested == "auto": return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    return requested


def auto_batch(device: str) -> tuple[int, int]:
    if device == "cpu": return 1, 1
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if memory_gb < 6: return 1, 8
    if memory_gb < 10: return 2, 4
    return 4, 2


def make_sampler(dataset: V02LanguageModelDataset, mix: dict, sample_count: int, seed: int):
    counts = dataset.kind_counts
    weights = [mix.get(kind, 0) / max(1, counts.get(kind, 1)) for _, _, kind in dataset.samples]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, sample_count, replacement=True, generator=generator)


@torch.inference_mode()
def evaluate(model, loader, device, max_batches):
    model.eval(); losses = []
    for index, (inputs, targets, _) in enumerate(loader):
        if index >= max_batches: break
        _, loss = model(inputs.to(device), targets.to(device)); losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def main():
    parser = argparse.ArgumentParser(description="Staged UniPilot Mini v0.2 training")
    parser.add_argument("--config", default="configs/unipilot-v02.json")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--max-records", type=int, help="0 uses the complete 45,000-row training split")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--stage", choices=["all", "general", "university", "conversation"], default="all")
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text(encoding="utf-8"))
    train_cfg = settings["training"]
    max_steps = args.max_steps or train_cfg["max_steps"]
    device = resolve_device(args.device or train_cfg["device"])
    auto_bs, auto_accum = auto_batch(device)
    batch_size = args.batch_size or (auto_bs if train_cfg["batch_size"] == "auto" else int(train_cfg["batch_size"]))
    accumulation = args.gradient_accumulation or (auto_accum if train_cfg["batch_size"] == "auto" else train_cfg["gradient_accumulation"])
    output_dir = Path(args.output_dir or f"checkpoints/unipilot-v02-step-{max_steps}")
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(train_cfg["seed"]); torch.manual_seed(train_cfg["seed"])
    tokenizer = BPETokenizer.load(settings["data"]["tokenizer"])
    config = ModelConfig(**settings["model"])
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(f"v0.2 requires {config.vocab_size} tokenizer tokens, got {tokenizer.vocab_size}")
    kinds = {"general"} if args.stage == "general" else ({"university_text"} if args.stage == "university" else ({"dialogue"} if args.stage == "conversation" else None))
    staged_records = min(train_cfg["max_train_records_for_staged_run"], max(500, max_steps * batch_size * accumulation))
    max_records = staged_records if args.max_records is None else args.max_records
    train_set = V02LanguageModelDataset(settings["data"]["train"], tokenizer, config.context_length, train_cfg["assistant_only_loss"], kinds, max_records)
    val_set = V02LanguageModelDataset(settings["data"]["validation"], tokenizer, config.context_length, train_cfg["assistant_only_loss"], kinds, train_cfg["max_validation_records"])
    micro_steps = max_steps * accumulation * batch_size
    sampler = make_sampler(train_set, train_cfg["mix"], micro_steps, train_cfg["seed"]) if args.stage == "all" else None
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, shuffle=sampler is None, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    model = UniPilotTransformer(config).to(device)
    optimizer = create_optimizer(model, train_cfg["learning_rate"], train_cfg["weight_decay"])
    global_step = 0
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"]); optimizer.load_state_dict(payload["optimizer_state"])
        global_step = int(payload["step"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: warmup_cosine_multiplier(step + global_step, train_cfg["warmup_steps"], max_steps))
    amp = device == "cuda" and train_cfg["precision"] in {"auto", "fp16"}
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    log_path = output_dir / "training_log.csv"
    fields = ["step", "train_loss", "validation_loss", "perplexity", "learning_rate", "tokens_per_second", "step_time_ms", "memory_usage_mb"]
    process = psutil.Process(); iterator = iter(train_loader); recent_losses = []
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        initial_val = evaluate(model, val_loader, device, train_cfg["validation_batches"])
        writer.writerow({"step": global_step, "train_loss": "", "validation_loss": initial_val, "perplexity": math.exp(min(20, initial_val)),
                         "learning_rate": optimizer.param_groups[0]["lr"], "tokens_per_second": 0, "step_time_ms": 0,
                         "memory_usage_mb": process.memory_info().rss / 1024**2}); file.flush()
        print(json.dumps({"phase": "before_training", "step": global_step, "validation_loss": initial_val, "batch_size": batch_size,
                          "gradient_accumulation": accumulation, "device": device, "parameters": model.parameter_count(),
                          "training_samples_loaded": len(train_set), "validation_samples_loaded": len(val_set)}))
        while global_step < max_steps:
            started = time.perf_counter(); optimizer.zero_grad(set_to_none=True); tokens = 0; micro_losses = []
            try:
                for _ in range(accumulation):
                    inputs, targets, _ = next(iterator); inputs, targets = inputs.to(device), targets.to(device)
                    with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                        _, loss = model(inputs, targets); scaled_loss = loss / accumulation
                    scaler.scale(scaled_loss).backward(); micro_losses.append(loss.item()); tokens += (targets != -100).sum().item()
            except torch.cuda.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                raise RuntimeError(f"CUDA OOM at batch_size={batch_size}. Re-run with --batch-size {max(1, batch_size // 2)} and increase --gradient-accumulation.") from error
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"])
            scaler.step(optimizer); scaler.update(); scheduler.step(); global_step += 1
            elapsed = time.perf_counter() - started; loss_value = sum(micro_losses) / len(micro_losses); recent_losses.append(loss_value)
            should_eval = global_step % train_cfg["eval_interval"] == 0 or global_step == max_steps
            if should_eval:
                val_loss = evaluate(model, val_loader, device, train_cfg["validation_batches"])
                row = {"step": global_step, "train_loss": sum(recent_losses) / len(recent_losses), "validation_loss": val_loss,
                       "perplexity": math.exp(min(20, val_loss)), "learning_rate": optimizer.param_groups[0]["lr"],
                       "tokens_per_second": tokens / max(elapsed, 1e-9), "step_time_ms": elapsed * 1000,
                       "memory_usage_mb": (torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else process.memory_info().rss / 1024**2)}
                writer.writerow(row); file.flush(); print(json.dumps(row)); recent_losses.clear()
            if global_step % train_cfg["checkpoint_interval"] == 0 or global_step == max_steps:
                model.config.model_name = f"UniPilot Mini v0.2-{global_step}"
                save_checkpoint(output_dir / f"checkpoint-step-{global_step}.pt", model, optimizer, scheduler, 0, global_step,
                                val_loss if should_eval else loss_value, model.config, "unipilot-byte-bpe-v02-512")
    print(json.dumps({"status": "complete", "step": global_step, "checkpoint": str(output_dir / f"checkpoint-step-{global_step}.pt")}))


if __name__ == "__main__": main()
