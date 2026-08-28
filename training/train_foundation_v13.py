from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
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


CHECKPOINT_FORMAT = "foundation-v13-v3"


class StatefulBlockSampler:
    """Deterministic shuffled block sampler whose exact position is checkpointed."""

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
            "size": self.size,
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self.position,
            "permutation": self.permutation.clone(),
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["size"]) != self.size or int(state["seed"]) != self.seed:
            raise RuntimeError("Foundation v1.3 sampler state mismatch")
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])
        self.permutation = state["permutation"].to(dtype=torch.int64).clone()
        self.generator.set_state(state["generator_state"].cpu())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


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


def evaluate_validation(model: UniPilotTransformer, dataset: PackedTokenDataset,
                        device: str, batches: int) -> float:
    was_training = model.training
    model.eval()
    values = []
    with torch.inference_mode():
        for index in range(min(batches, len(dataset))):
            inputs, targets = dataset[index]
            _, loss = model(
                inputs.unsqueeze(0).to(device), targets.unsqueeze(0).to(device)
            )
            values.append(float(loss.item()))
    model.train(was_training)
    return sum(values) / len(values)


def scheduler_state(settings: dict, step: int) -> dict:
    return {
        "kind": "stateless_warmup_cosine",
        "global_step": step,
        "base_learning_rate": settings["learning_rate"],
        "warmup_steps": settings["warmup_steps"],
        "schedule_steps": settings["schedule_steps"],
        "minimum_ratio": settings["minimum_learning_rate_ratio"],
    }


def checkpoint_manifest(*, model: UniPilotTransformer, sampler: StatefulBlockSampler,
                        settings: dict, corpus: dict, step: int,
                        history: list[dict], resumed_from: str | None) -> dict:
    train_tokens = int(corpus["splits"]["train"]["tokens"])
    tokens_processed = step * settings["batch_size"] * model.config.context_length
    return {
        "schema_version": "foundation-v13-checkpoint-manifest-v3",
        "model": f"{settings['model_name']} step {step}",
        "parameters": model.parameter_count(),
        "global_step": step,
        "model_config": model.config.to_dict(),
        "tokenizer": corpus["tokenizer"],
        "corpus_manifest": settings["corpus_manifest"],
        "train_corpus_tokens": train_tokens,
        "tokens_processed": tokens_processed,
        "train_corpus_fraction": tokens_processed / train_tokens,
        "epoch_equivalent": tokens_processed / train_tokens,
        "training_stage": settings["training_stage"],
        "scheduler_state": scheduler_state(settings, step),
        "optimizer": {
            "name": settings["optimizer"],
            "learning_rate": settings["learning_rate"],
            "betas": settings["optimizer_betas"],
            "epsilon": settings["optimizer_epsilon"],
            "weight_decay": settings["weight_decay"],
            "gradient_clip": settings["gradient_clip"],
        },
        "rng_states_saved": ["python", "numpy", "torch_cpu"] + (
            ["torch_cuda"] if torch.cuda.is_available() else []
        ),
        "sampler_state_saved": True,
        "sampler_epoch": sampler.epoch,
        "sampler_position": sampler.position,
        "history": history,
        "initialization": settings["initialization"],
        "scratch_start": resumed_from is None,
        "resumed_from": resumed_from,
        "seed": settings["seed"],
        "git_commit_at_run_start": git_head(),
        "final_blind_used": False,
        "external_ai_api": "OFF",
        "production_changed": False,
    }


def save_checkpoint(path: Path, *, model: UniPilotTransformer, optimizer,
                    sampler: StatefulBlockSampler, settings: dict, corpus: dict,
                    step: int, history: list[dict], resumed_from: str | None) -> dict:
    manifest = checkpoint_manifest(
        model=model, sampler=sampler, settings=settings, corpus=corpus, step=step,
        history=history, resumed_from=resumed_from,
    )
    payload = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler_state(settings, step),
        "global_step": step,
        "step": step,
        "random_state": random_state(),
        "sampler_state": sampler.state_dict(),
        "config": model.config.to_dict(),
        "foundation_v13_manifest": manifest,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    manifest["checkpoint_bytes"] = path.stat().st_size
    manifest["checkpoint_sha256"] = sha256(path)
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_model(settings: dict, corpus: dict, device: str) -> UniPilotTransformer:
    config = ModelConfig(
        model_name=settings["model_name"],
        vocab_size=int(corpus["vocab"]),
        **settings["model"],
    )
    return UniPilotTransformer(config).to(device)


def train_run(*, settings: dict, max_steps: int, output_dir: Path,
              metric_steps: set[int], checkpoint_steps: set[int],
              metrics_output: Path | None = None, resume: Path | None = None,
              device: str = "cpu", cpu_threads: int = 4) -> dict:
    if max_steps > 250:
        raise RuntimeError("PHASE 24 forbids training beyond 250 steps")
    if device == "cpu":
        torch.set_num_threads(max(1, cpu_threads))
    seed = int(settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    corpus = json.loads((ROOT / settings["corpus_manifest"]).read_text(encoding="utf-8"))
    model = build_model(settings, corpus, device)
    optimizer = create_optimizer(model, settings["learning_rate"], settings["weight_decay"])
    train_data = PackedTokenDataset(
        ROOT / corpus["splits"]["train"]["path"], model.config.context_length
    )
    validation_data = PackedTokenDataset(
        ROOT / corpus["splits"]["validation"]["path"], model.config.context_length
    )
    sampler = StatefulBlockSampler(len(train_data), seed)
    step = 0
    history: list[dict] = []
    resumed_from = None
    if resume is not None:
        payload = torch.load(resume, map_location=device, weights_only=False)
        if payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
            raise RuntimeError("Foundation v1.3 refuses a non-v3 checkpoint")
        if payload["config"] != model.config.to_dict():
            raise RuntimeError("Foundation v1.3 model config mismatch on resume")
        expected_scheduler = scheduler_state(settings, int(payload["global_step"]))
        if payload["scheduler_state"] != expected_scheduler:
            raise RuntimeError("Foundation v1.3 scheduler state mismatch")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        step = int(payload["global_step"])
        if step >= max_steps:
            raise RuntimeError("resume checkpoint must be earlier than max_steps")
        sampler.load_state_dict(payload["sampler_state"])
        restore_random_state(payload["random_state"])
        history = list(payload["foundation_v13_manifest"].get("history", []))
        resumed_from = resume.relative_to(ROOT).as_posix() if resume.is_relative_to(ROOT) else str(resume)

    process = psutil.Process(os.getpid())
    peak_ram_mb = process.memory_info().rss / 1024**2
    recent_losses: list[float] = []
    recent_speeds: list[float] = []
    all_step_losses: list[float] = []
    previous_validation = history[-1]["validation_loss"] if history else None
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training-log.csv"
    fields = (
        "step", "train_loss", "validation_loss", "perplexity", "learning_rate",
        "gradient_norm", "tokens_per_second", "peak_ram_mb", "tokens_processed",
        "corpus_fraction", "corpus_percentage", "epoch_equivalent",
    )
    mode = "a" if resume is not None and log_path.exists() else "w"
    run_started = time.perf_counter()
    with log_path.open(mode, newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        if step == 0 and 0 in metric_steps:
            valid = evaluate_validation(
                model, validation_data, device, settings["validation_batches"]
            )
            peak_ram_mb = max(peak_ram_mb, process.memory_info().rss / 1024**2)
            first_lr = settings["learning_rate"] * warmup_cosine_multiplier(
                0, settings["warmup_steps"], settings["schedule_steps"],
                settings["minimum_learning_rate_ratio"],
            )
            row = {
                "step": 0,
                "train_loss": None,
                "validation_loss": valid,
                "perplexity": math.exp(valid),
                "learning_rate": first_lr,
                "gradient_norm": None,
                "tokens_per_second": None,
                "peak_ram_mb": peak_ram_mb,
                "tokens_processed": 0,
                "corpus_fraction": 0.0,
                "corpus_percentage": 0.0,
                "epoch_equivalent": 0.0,
            }
            history.append(row)
            writer.writerow(row)
            output.flush()
            previous_validation = valid

        while step < max_steps:
            index = sampler.next_index()
            inputs, targets = train_data[index]
            inputs = inputs.unsqueeze(0).to(device)
            targets = targets.unsqueeze(0).to(device)
            lr = settings["learning_rate"] * warmup_cosine_multiplier(
                step, settings["warmup_steps"], settings["schedule_steps"],
                settings["minimum_learning_rate_ratio"],
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            _, loss = model(inputs, targets)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"non-finite Foundation v1.3 loss at step {step + 1}")
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings["gradient_clip"]
            ))
            if not math.isfinite(gradient_norm):
                raise RuntimeError(f"non-finite Foundation v1.3 gradient at step {step + 1}")
            optimizer.step()
            elapsed = time.perf_counter() - started
            step += 1
            loss_value = float(loss.item())
            all_step_losses.append(loss_value)
            recent_losses.append(loss_value)
            recent_speeds.append(targets.numel() / max(elapsed, 1e-9))
            peak_ram_mb = max(peak_ram_mb, process.memory_info().rss / 1024**2)

            if step in metric_steps:
                valid = evaluate_validation(
                    model, validation_data, device, settings["validation_batches"]
                )
                peak_ram_mb = max(peak_ram_mb, process.memory_info().rss / 1024**2)
                tokens_processed = step * settings["batch_size"] * model.config.context_length
                train_tokens = int(corpus["splits"]["train"]["tokens"])
                fraction = tokens_processed / train_tokens
                row = {
                    "step": step,
                    "train_loss": sum(recent_losses) / len(recent_losses),
                    "validation_loss": valid,
                    "perplexity": math.exp(valid),
                    "learning_rate": lr,
                    "gradient_norm": gradient_norm,
                    "tokens_per_second": sum(recent_speeds) / len(recent_speeds),
                    "peak_ram_mb": peak_ram_mb,
                    "tokens_processed": tokens_processed,
                    "corpus_fraction": fraction,
                    "corpus_percentage": fraction * 100,
                    "epoch_equivalent": fraction,
                }
                if previous_validation is not None and valid > previous_validation + 0.5:
                    raise RuntimeError(
                        f"Foundation v1.3 validation divergence at step {step}: "
                        f"{previous_validation:.6f} -> {valid:.6f}"
                    )
                history.append(row)
                writer.writerow(row)
                output.flush()
                previous_validation = valid
                recent_losses.clear()
                recent_speeds.clear()
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if step in checkpoint_steps:
                    save_checkpoint(
                        output_dir / f"checkpoint-step-{step}.pt",
                        model=model, optimizer=optimizer, sampler=sampler,
                        settings=settings, corpus=corpus, step=step,
                        history=history, resumed_from=resumed_from,
                    )

    result = {
        "schema_version": "foundation-v13-training-curve-v1",
        "status": "COMPLETED",
        "project": settings["project"],
        "initialization": settings["initialization"],
        "resumed_from": resumed_from,
        "scratch_start": resumed_from is None,
        "parameters": model.parameter_count(),
        "model_config": model.config.to_dict(),
        "corpus_manifest": settings["corpus_manifest"],
        "tokenizer": corpus["tokenizer"],
        "train_corpus_tokens": corpus["splits"]["train"]["tokens"],
        "validation_corpus_tokens": corpus["splits"]["validation"]["tokens"],
        "validation_batches": settings["validation_batches"],
        "optimizer": {
            "name": settings["optimizer"],
            "learning_rate": settings["learning_rate"],
            "warmup_steps": settings["warmup_steps"],
            "schedule": "cosine",
            "schedule_steps": settings["schedule_steps"],
            "minimum_ratio": settings["minimum_learning_rate_ratio"],
            "betas": settings["optimizer_betas"],
            "epsilon": settings["optimizer_epsilon"],
            "weight_decay": settings["weight_decay"],
            "gradient_clip": settings["gradient_clip"],
        },
        "seed": settings["seed"],
        "max_steps": max_steps,
        "metric_steps": sorted(metric_steps),
        "checkpoint_steps": sorted(checkpoint_steps),
        "history": history,
        "best_validation": min(history, key=lambda row: row["validation_loss"]),
        "wall_seconds": time.perf_counter() - run_started,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": device,
            "cpu_threads": cpu_threads,
        },
        "nan_or_inf": False,
        "diverged": False,
        "final_blind_used": False,
        "external_ai_api": "OFF",
        "production_changed": False,
        "foundation_500_executed": False,
        "foundation_1000_executed": False,
        "standard_46m_executed": False,
    }
    if metrics_output is not None:
        metrics_output.parent.mkdir(parents=True, exist_ok=True)
        metrics_output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return {
        "model": model,
        "optimizer": optimizer,
        "sampler": sampler,
        "history": history,
        "step_losses": all_step_losses,
        "result": result,
        "checkpoint": output_dir / f"checkpoint-step-{max_steps}.pt",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v13.json")
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir", default="checkpoints/foundation-v13-clean-250")
    parser.add_argument("--metrics-output", default="evaluation/foundation-v13-training-curve.json")
    parser.add_argument("--cpu-threads", type=int)
    args = parser.parse_args()
    settings = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if args.max_steps != 250:
        raise RuntimeError("The PHASE 24 command only permits the formal 250-step run")
    output_dir = ROOT / args.output_dir
    resume = ROOT / args.resume if args.resume else None
    if resume is None and any(output_dir.glob("checkpoint-step-*.pt")):
        raise RuntimeError(f"Refusing to overwrite an existing scratch run: {output_dir}")
    train_run(
        settings=settings,
        max_steps=args.max_steps,
        output_dir=output_dir,
        metric_steps=set(settings["metric_steps"]),
        checkpoint_steps=set(settings["checkpoint_steps"]),
        metrics_output=ROOT / args.metrics_output,
        resume=resume,
        device="cpu",
        cpu_threads=args.cpu_threads or int(settings["cpu_threads"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
