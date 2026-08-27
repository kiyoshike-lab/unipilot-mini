from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.packed_dataset import PackedTokenDataset
from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from training.optimizer import create_optimizer
from training.scheduler import warmup_cosine_multiplier


class StatefulBlockSampler:
    def __init__(self, size: int, seed: int, state: dict | None = None):
        self.size = size
        self.seed = seed
        self.generator = torch.Generator()
        self.epoch = 0
        self.position = 0
        self.permutation = torch.empty(0, dtype=torch.int64)
        if state is None:
            self.generator.manual_seed(seed)
            self._new_epoch()
        else:
            self.load_state_dict(state)

    def _new_epoch(self) -> None:
        self.permutation = torch.randperm(self.size, generator=self.generator)
        self.position = 0

    def next_index(self) -> int:
        if self.position >= self.size:
            self.epoch += 1
            self._new_epoch()
        value = int(self.permutation[self.position].item())
        self.position += 1
        return value

    def state_dict(self) -> dict:
        return {
            "size": self.size, "seed": self.seed, "epoch": self.epoch,
            "position": self.position, "permutation": self.permutation.clone(),
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["size"]) != self.size or int(state["seed"]) != self.seed:
            raise RuntimeError("Foundation v1.1 sampler state mismatch")
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])
        self.permutation = state["permutation"].to(dtype=torch.int64).clone()
        self.generator.set_state(state["generator_state"].cpu())


def random_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def validation_loss(model, dataset, device: str, batches: int = 16) -> float:
    model.eval()
    values = []
    with torch.inference_mode():
        for index in range(min(batches, len(dataset))):
            inputs, targets = dataset[index]
            _, loss = model(inputs.unsqueeze(0).to(device), targets.unsqueeze(0).to(device))
            values.append(float(loss.item()))
    model.train()
    return sum(values) / len(values)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def save_checkpoint_v2(path: Path, model, optimizer, sampler: StatefulBlockSampler,
                       settings: dict, corpus: dict, step: int, history: list[dict]) -> dict:
    scheduler_state = {
        "kind": "stateless_warmup_cosine", "global_step": step,
        "base_learning_rate": settings["learning_rate"],
        "warmup_steps": settings["warmup_steps"],
        "schedule_steps": settings["schedule_steps"], "minimum_ratio": .1,
    }
    manifest = {
        "schema_version": "foundation-v11-checkpoint-manifest-v2",
        "model": f"UniPilot Foundation v1.1 Clean 20M step {step}",
        "parameters": model.parameter_count(), "global_step": step,
        "model_config": model.config.to_dict(), "tokenizer": corpus["tokenizer"],
        "corpus_manifest": settings["corpus_manifest"],
        "train_corpus_tokens": corpus["splits"]["train"]["tokens"],
        "tokens_processed": step * settings["batch_size"] * model.config.context_length,
        "train_corpus_fraction": (
            step * settings["batch_size"] * model.config.context_length
            / corpus["splits"]["train"]["tokens"]
        ),
        "training_stage": settings["training_stage"], "scheduler_state": scheduler_state,
        "rng_states_saved": ["python", "numpy", "torch_cpu"] +
                            (["torch_cuda"] if torch.cuda.is_available() else []),
        "sampler_state_saved": True, "history": history,
        "initialization": settings["initialization"], "git_commit": git_head(),
        "final_blind_used": False, "external_ai_api": "OFF", "production_changed": False,
    }
    payload = {
        "checkpoint_format": "foundation-v11-v2", "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler_state,
        "global_step": step, "step": step, "random_state": random_state(),
        "sampler_state": sampler.state_dict(), "config": model.config.to_dict(),
        "foundation_v11_manifest": manifest,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    manifest["checkpoint_bytes"] = path.stat().st_size
    manifest["checkpoint_sha256"] = sha256(path)
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def train_run(*, settings: dict, max_steps: int, output_dir: Path,
              checkpoint_steps: set[int], resume: Path | None = None,
              device: str = "cpu", cpu_threads: int = 4) -> dict:
    if device == "cpu":
        torch.set_num_threads(max(1, cpu_threads))
    seed = int(settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    corpus = json.loads((ROOT / settings["corpus_manifest"]).read_text(encoding="utf-8"))
    config = ModelConfig(
        model_name="UniPilot Foundation v1.1 Clean 20M", vocab_size=corpus["vocab"],
        dropout=.1, bias=True, **settings["model"],
    )
    model = UniPilotTransformer(config).to(device)
    optimizer = create_optimizer(model, settings["learning_rate"], settings["weight_decay"])
    train_data = PackedTokenDataset(ROOT / corpus["splits"]["train"]["path"],
                                    config.context_length)
    validation_data = PackedTokenDataset(ROOT / corpus["splits"]["validation"]["path"],
                                         config.context_length)
    sampler = StatefulBlockSampler(len(train_data), seed)
    step = 0
    history: list[dict] = []
    if resume is not None:
        payload = torch.load(resume, map_location=device, weights_only=False)
        if payload.get("checkpoint_format") != "foundation-v11-v2":
            raise RuntimeError("Foundation v1.1 refuses non-v2 resume checkpoints")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        step = int(payload["global_step"])
        if step >= max_steps:
            raise RuntimeError("resume checkpoint must be earlier than max_steps")
        scheduler_state = payload["scheduler_state"]
        if int(scheduler_state["global_step"]) != step:
            raise RuntimeError("scheduler global step mismatch")
        sampler.load_state_dict(payload["sampler_state"])
        restore_random_state(payload["random_state"])
        history = list(payload["foundation_v11_manifest"].get("history", []))
    process = psutil.Process(os.getpid())
    step_losses: list[float] = []
    recent_losses: list[float] = []
    recent_speeds: list[float] = []
    last_norm = 0.0
    previous_validation = history[-1]["validation_loss"] if history else None
    log_path = output_dir / "training-log.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ("step", "train_loss", "validation_loss", "learning_rate", "gradient_norm",
              "tokens_per_second", "peak_ram_mb", "tokens_processed", "corpus_fraction")
    mode = "a" if resume is not None and log_path.exists() else "w"
    with log_path.open(mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        if step == 0 and 0 in checkpoint_steps:
            initial_validation = validation_loss(model, validation_data, device)
            row = {
                "step": 0, "train_loss": None, "validation_loss": initial_validation,
                "learning_rate": settings["learning_rate"] * warmup_cosine_multiplier(
                    0, settings["warmup_steps"], settings["schedule_steps"], .1),
                "gradient_norm": None, "tokens_per_second": None,
                "peak_ram_mb": process.memory_info().rss / 1024**2, "tokens_processed": 0,
                "corpus_fraction": 0.0,
            }
            history.append(row)
            writer.writerow(row)
            file.flush()
        while step < max_steps:
            index = sampler.next_index()
            inputs, targets = train_data[index]
            inputs = inputs.unsqueeze(0).to(device)
            targets = targets.unsqueeze(0).to(device)
            lr = settings["learning_rate"] * warmup_cosine_multiplier(
                step, settings["warmup_steps"], settings["schedule_steps"], .1
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            _, loss = model(inputs, targets)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("non-finite Foundation v1.1 loss")
            loss.backward()
            last_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings["gradient_clip"]
            ))
            if not math.isfinite(last_norm):
                raise RuntimeError("non-finite Foundation v1.1 gradient")
            optimizer.step()
            elapsed = time.perf_counter() - started
            step += 1
            value = float(loss.item())
            step_losses.append(value)
            recent_losses.append(value)
            recent_speeds.append(targets.numel() / max(elapsed, 1e-9))
            if step in checkpoint_steps:
                valid = validation_loss(model, validation_data, device)
                row = {
                    "step": step, "train_loss": sum(recent_losses) / len(recent_losses),
                    "validation_loss": valid, "learning_rate": lr,
                    "gradient_norm": last_norm,
                    "tokens_per_second": sum(recent_speeds) / len(recent_speeds),
                    "peak_ram_mb": process.memory_info().rss / 1024**2,
                    "tokens_processed": step * settings["batch_size"] * config.context_length,
                    "corpus_fraction": (
                        step * settings["batch_size"] * config.context_length
                        / corpus["splits"]["train"]["tokens"]
                    ),
                }
                if previous_validation is not None and valid > previous_validation + .5:
                    raise RuntimeError("Foundation v1.1 validation loss diverged")
                history.append(row)
                writer.writerow(row)
                file.flush()
                checkpoint = output_dir / f"checkpoint-step-{step}.pt"
                save_checkpoint_v2(checkpoint, model, optimizer, sampler, settings,
                                   corpus, step, history)
                previous_validation = valid
                recent_losses.clear()
                recent_speeds.clear()
                print(json.dumps(row, ensure_ascii=False), flush=True)
    return {"model": model, "optimizer": optimizer, "sampler": sampler,
            "history": history, "step_losses": step_losses,
            "checkpoint": output_dir / f"checkpoint-step-{max_steps}.pt"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v11.json")
    parser.add_argument("--max-steps", type=int, choices=(20, 40, 100), required=True)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir", default="checkpoints/foundation-v11-clean-100")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    settings = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    metric_steps = set(settings["metric_steps"])
    if args.max_steps in (20, 40):
        metric_steps = {0, 20, 40} & set(range(args.max_steps + 1))
    train_run(settings=settings, max_steps=args.max_steps, output_dir=ROOT / args.output_dir,
              checkpoint_steps=metric_steps, resume=ROOT / args.resume if args.resume else None,
              device="cpu", cpu_threads=args.cpu_threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
