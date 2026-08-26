from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import time

import psutil
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.packed_dataset import PackedTokenDataset
from foundation.base_tokenizer import FoundationTokenizer
from inference.sampling import apply_repetition_penalty
from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from training.optimizer import create_optimizer
from training.scheduler import warmup_cosine_multiplier


CHECKPOINT_STEPS = {50, 100, 250, 500}


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def validation_loss(model, loader, device: str, batches: int = 16) -> float:
    model.eval()
    values = []
    with torch.inference_mode():
        for index, (inputs, targets) in enumerate(loader):
            if index >= batches:
                break
            _, loss = model(inputs.to(device), targets.to(device))
            values.append(float(loss.item()))
    model.train()
    return sum(values) / max(1, len(values))


def natural_text(text: str) -> tuple[bool, float]:
    value = text.strip()
    japanese = len(re.findall(r"[ぁ-んァ-ヶー一-龥々]", value))
    grams = [value[index:index + 3] for index in range(max(0, len(value) - 2))]
    repetition = 0.0 if not grams else 1 - len(set(grams)) / len(grams)
    natural = len(value) >= 20 and japanese / max(1, len(value)) >= .35 and repetition < .35
    return natural, repetition


def bigram_overlap(left: str, right: str) -> float:
    def values(text: str) -> set[str]:
        normalized = re.sub(r"\s+", "", text)
        return {normalized[index:index + 2] for index in range(max(0, len(normalized) - 1))}
    a, b = values(left), values(right)
    return len(a & b) / max(1, len(a | b))


@torch.inference_mode()
def generate(model, tokenizer, prefix: str, max_new_tokens: int = 64) -> tuple[str, dict]:
    ids = tokenizer.encode(prefix, add_bos=True)
    generated: list[int] = []
    past = None
    started = time.perf_counter()
    first = None
    eos = False
    forbidden = [index for index in range(len(tokenizer.special_tokens))
                 if index != tokenizer.eos_id]
    for _ in range(max_new_tokens):
        current_ids = ids[-model.config.context_length:] if past is None else [ids[-1]]
        current = torch.tensor([current_ids], dtype=torch.long,
                               device=next(model.parameters()).device)
        logits, _, past = model(current, past_key_values=past, use_cache=True)
        scores = apply_repetition_penalty(logits[0, -1], ids[-64:], 1.1).clone()
        scores[forbidden] = -torch.inf
        next_id = int(scores.argmax().item())
        if first is None:
            first = time.perf_counter() - started
        ids.append(next_id)
        generated.append(next_id)
        if next_id == tokenizer.eos_id:
            eos = True
            break
    elapsed = time.perf_counter() - started
    return tokenizer.decode(generated, skip_special=True), {
        "tokens": len(generated), "first_token_seconds": first or 0.0,
        "total_seconds": elapsed, "tokens_per_second": len(generated) / max(elapsed, 1e-9),
        "eos_reached": eos,
    }


def validation_documents(limit: int = 8) -> list[dict]:
    path = ROOT / "data/foundation_v10/documents/validation.jsonl.gz"
    selected = []
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if len(row["text"]) >= 400:
                selected.append(row)
            if len(selected) >= limit:
                break
    return selected


@torch.inference_mode()
def base_probe(model, tokenizer) -> dict:
    model.eval()
    output = []
    for row in validation_documents():
        source = row["text"]
        boundary = min(160, max(60, len(source) // 4))
        prefix = source[:boundary]
        reference = source[boundary:boundary + 160]
        text, metrics = generate(model, tokenizer, prefix)
        natural, repetition = natural_text(text)
        output.append({
            "id": row["id"], "prefix": prefix, "reference_continuation": reference,
            "generated": text, "natural": natural, "repetition_rate": repetition,
            "reference_bigram_overlap": bigram_overlap(text, reference), **metrics,
        })
    model.train()
    return {
        "documents": len(output),
        "natural_japanese_rate": sum(row["natural"] for row in output) / len(output),
        "mean_repetition_rate": sum(row["repetition_rate"] for row in output) / len(output),
        "mean_reference_bigram_overlap": sum(row["reference_bigram_overlap"] for row in output) /
                                          len(output),
        "eos_rate": sum(row["eos_reached"] for row in output) / len(output),
        "mean_first_token_seconds": sum(row["first_token_seconds"] for row in output) / len(output),
        "mean_tokens_per_second": sum(row["tokens_per_second"] for row in output) / len(output),
        "rows": output,
    }


def save_checkpoint(output: Path, model, optimizer, step: int, architecture: str,
                    settings: dict, corpus: dict, stats: dict, probe: dict,
                    initial_validation_loss: float) -> dict:
    model.config.model_name = f"UniPilot Foundation v1.0 {architecture} step {step}"
    validation_improvement = initial_validation_loss - stats["validation_loss"]
    healthy_loss = (
        math.isfinite(stats["loss"])
        and math.isfinite(stats["validation_loss"])
        and validation_improvement >= .15
        and stats["validation_loss"] <= initial_validation_loss + .05
    )
    manifest = {
        "schema_version": "foundation-v10-checkpoint-manifest-v1",
        "model": model.config.model_name, "architecture": architecture,
        "parameters": model.parameter_count(), "step": step,
        "model_config": model.config.to_dict(), "initialization": settings["initialization"],
        "tokenizer": corpus["tokenizer"], "vocab": corpus["vocab"],
        "corpus_manifest": "data/foundation_v10/packed/vocab-%d/manifest.json" % corpus["vocab"],
        "corpus_tokens": corpus["total_tokens"], "training_stage": "A",
        "training_metrics": stats, "initial_validation_loss": initial_validation_loss,
        "validation_loss_improvement": validation_improvement, "development_probe": probe,
        "healthy_loss_curve": healthy_loss,
        "continue_to_500_recommended": bool(step == 100 and healthy_loss),
        "next_1000_requires_base_gate": True, "git_commit": git_head(),
        "seed": settings["seed"], "final_blind_used_for_training": False,
        "external_pretrained_model": False, "external_ai_api": "OFF",
        "production_changed": False,
    }
    path = output / f"checkpoint-step-{step}.pt"
    torch.save({
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "step": step, "architecture": architecture, "config": model.config.to_dict(),
        "initial_validation_loss": initial_validation_loss,
        "foundation_v10_manifest": manifest,
    }, path)
    manifest["training_checkpoint_bytes"] = path.stat().st_size
    manifest_path = output / f"checkpoint-step-{step}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v10.json")
    parser.add_argument("--architecture", choices=("20m", "30m", "46m"), required=True)
    parser.add_argument("--max-steps", type=int, choices=(50, 100, 500), required=True)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir", default="checkpoints/foundation-v10-sanity")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    settings = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    benchmark = json.loads((ROOT / "evaluation/foundation-v10-tokenizer-benchmark.json").read_text(
        encoding="utf-8"))
    vocab = int(benchmark["selected_vocab"])
    corpus_path = ROOT / f"data/foundation_v10/packed/vocab-{vocab}/manifest.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    tokenizer = FoundationTokenizer.load(ROOT / corpus["tokenizer"])
    dimensions = settings["model_candidates"][args.architecture]
    config = ModelConfig(
        model_name=f"UniPilot Foundation v1.0 {args.architecture}",
        vocab_size=vocab, dropout=.1, bias=True, **dimensions,
    )
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    if device == "cpu":
        torch.set_num_threads(max(1, args.cpu_threads))
    random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    model = UniPilotTransformer(config).to(device)
    optimizer = create_optimizer(model, settings["learning_rate"], settings["weight_decay"])
    step = 0
    initial_validation_loss = 0.0
    if args.resume:
        payload = torch.load(ROOT / args.resume, map_location=device, weights_only=False)
        if payload.get("architecture") != args.architecture:
            raise RuntimeError("resume architecture mismatch")
        step = int(payload["step"])
        if args.max_steps == 500 and step != 100:
            raise RuntimeError("500-step run must resume the exact step-100 checkpoint")
        if args.max_steps == 500 and not payload["foundation_v10_manifest"].get(
                "continue_to_500_recommended"):
            raise RuntimeError("step-100 healthy-loss gate did not pass")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        initial_validation_loss = float(payload["initial_validation_loss"])
        del payload
        gc.collect()
    train_data = PackedTokenDataset(
        ROOT / corpus["splits"]["train"]["path"], config.context_length)
    validation_data = PackedTokenDataset(
        ROOT / corpus["splits"]["validation"]["path"], config.context_length)
    generator = torch.Generator().manual_seed(settings["seed"])
    train_loader = DataLoader(train_data, batch_size=settings["batch_size"], shuffle=True,
                              generator=generator, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=1, shuffle=False, num_workers=0)
    if not args.resume:
        initial_validation_loss = validation_loss(model, validation_loader, device)
    iterator = iter(train_loader)
    # Recreate the seeded permutation and advance to the exact global step.  Without
    # this, a resumed 50->100 run would train on the first 50 blocks twice.
    if step:
        for _ in range(step):
            try:
                next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
    output = ROOT / args.output_dir / args.architecture
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training-log.csv"
    mode = "a" if args.resume and log_path.exists() else "w"
    fields = ("step", "loss", "validation_loss", "initial_validation_loss", "learning_rate",
              "gradient_norm", "gradient_clipping_count", "tokens_per_second",
              "step_time_seconds", "memory_usage_mb")
    recent: list[float] = []
    clipping = 0
    process = psutil.Process()
    with log_path.open(mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        while step < args.max_steps:
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, targets = next(iterator)
            lr = settings["learning_rate"] * warmup_cosine_multiplier(
                step, settings["warmup_steps"], settings["schedule_steps"], .1)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            inputs, targets = inputs.to(device), targets.to(device)
            started = time.perf_counter()
            _, loss = model(inputs, targets)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("non-finite Foundation v1.0 loss")
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"]))
            if not math.isfinite(norm):
                raise RuntimeError("non-finite Foundation v1.0 gradient")
            clipping += int(norm > settings["gradient_clip"])
            optimizer.step()
            step += 1
            elapsed = time.perf_counter() - started
            recent.append(float(loss.item()))
            recent = recent[-50:]
            if step in CHECKPOINT_STEPS or step == args.max_steps:
                validation = validation_loss(model, validation_loader, device)
                stats = {
                    "step": step, "loss": sum(recent) / len(recent),
                    "validation_loss": validation,
                    "initial_validation_loss": initial_validation_loss,
                    "learning_rate": lr, "gradient_norm": norm,
                    "gradient_clipping_count": clipping,
                    "tokens_per_second": targets.numel() / max(elapsed, 1e-9),
                    "step_time_seconds": elapsed,
                    "memory_usage_mb": process.memory_info().rss / 1024**2,
                }
                probe = base_probe(model, tokenizer)
                writer.writerow(stats)
                file.flush()
                manifest = save_checkpoint(output, model, optimizer, step, args.architecture,
                                           settings, corpus, stats, probe,
                                           initial_validation_loss)
                print(json.dumps({
                    "architecture": args.architecture, "parameters": model.parameter_count(),
                    "training": stats,
                    "probe": {key: value for key, value in probe.items() if key != "rows"},
                    "healthy_loss_curve": manifest["healthy_loss_curve"],
                    "continue_to_500_recommended": manifest["continue_to_500_recommended"],
                }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
