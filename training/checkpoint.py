from __future__ import annotations

from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer, scheduler, epoch, step, loss, config, tokenizer_version="unipilot-byte-bpe-v1"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "epoch": epoch, "step": step, "loss": loss, "config": config.to_dict(),
        "tokenizer_version": tokenizer_version,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state"):
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state"):
        scheduler.load_state_dict(payload["scheduler_state"])
    return payload
