from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader

from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from tokenizer.tokenizer import BPETokenizer
from training.checkpoint import load_checkpoint, save_checkpoint
from training.dataset import LanguageModelDataset, load_documents, split_documents
from training.optimizer import create_optimizer
from training.scheduler import warmup_cosine_multiplier


@torch.no_grad()
def validation_loss(model, loader, device):
    model.eval(); losses = []
    for inputs, targets in loader:
        _, loss = model(inputs.to(device), targets.to(device)); losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def train(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    config = ModelConfig.load(args.config)
    tokenizer = BPETokenizer.load(args.tokenizer)
    if config.vocab_size != tokenizer.vocab_size:
        config.vocab_size = tokenizer.vocab_size
    documents = load_documents(args.dataset)
    train_docs, validation_docs = split_documents(documents, args.validation_ratio, args.seed)
    train_set = LanguageModelDataset(train_docs, tokenizer, config.context_length)
    validation_set = LanguageModelDataset(validation_docs, tokenizer, config.context_length)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, num_workers=0)
    model = UniPilotTransformer(config).to(device)
    optimizer = create_optimizer(model, args.learning_rate)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: warmup_cosine_multiplier(step, args.warmup_steps, total_steps))
    start_epoch = 0; global_step = 0
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch, global_step = state["epoch"] + 1, state["step"]
    amp_enabled = args.mixed_precision and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history_path = Path(args.output_dir) / "training_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and history_path.exists() else "w"
    with history_path.open(mode, newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=["epoch", "step", "train_loss", "validation_loss", "learning_rate", "step_time_ms", "tokens_per_sec"])
        if mode == "w":
            writer.writeheader()
            initial_val = validation_loss(model, validation_loader, device)
            writer.writerow({"epoch": -1, "step": global_step, "train_loss": "", "validation_loss": initial_val,
                             "learning_rate": optimizer.param_groups[0]["lr"], "step_time_ms": 0, "tokens_per_sec": 0})
            history_file.flush()
            print(json.dumps({"epoch": -1, "step": global_step, "validation_loss": initial_val, "phase": "before_training"}))
        for epoch in range(start_epoch, args.epochs):
            model.train(); epoch_losses = []; started = time.perf_counter(); token_count = 0
            for inputs, targets in train_loader:
                step_started = time.perf_counter(); inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp_enabled):
                    _, loss = model(inputs, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                scaler.step(optimizer); scaler.update(); scheduler.step()
                global_step += 1; epoch_losses.append(loss.item()); token_count += (targets != -100).sum().item()
                if args.max_steps and global_step >= args.max_steps: break
            elapsed = time.perf_counter() - started
            train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            val_loss = validation_loss(model, validation_loader, device)
            step_ms = elapsed * 1000 / max(1, len(epoch_losses))
            tokens_per_sec = token_count / max(elapsed, 1e-9)
            writer.writerow({"epoch": epoch, "step": global_step, "train_loss": train_loss, "validation_loss": val_loss,
                             "learning_rate": optimizer.param_groups[0]["lr"], "step_time_ms": step_ms, "tokens_per_sec": tokens_per_sec})
            history_file.flush()
            checkpoint_path = Path(args.output_dir) / f"checkpoint-step-{global_step}.pt"
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, global_step, val_loss, config)
            print(json.dumps({"epoch": epoch, "step": global_step, "train_loss": train_loss, "validation_loss": val_loss,
                              "tokens_per_sec": tokens_per_sec, "checkpoint": str(checkpoint_path)}, ensure_ascii=False))
            if args.max_steps and global_step >= args.max_steps: break
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train UniPilot Mini from random initialization")
    parser.add_argument("--config", default="configs/v0.1.json")
    parser.add_argument("--dataset", default="data/conversations")
    parser.add_argument("--tokenizer", default="tokenizer/vocab.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--resume")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
