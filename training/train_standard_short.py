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
from torch.utils.data import DataLoader

from inference.generate import generate_text
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


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def loader(path: str, tokenizer, config: ModelConfig, batch_size: int, seed: int,
           shuffle: bool) -> DataLoader:
    dataset = CurriculumDataset(path, tokenizer, config.context_length, assistant_only=True)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dynamic_collate,
                      generator=generator, num_workers=0)


@torch.inference_mode()
def validation_loss(model, data, device: str, eos_id: int, eos_weight: float,
                    batches: int = 12) -> float:
    model.eval()
    values = []
    for index, (inputs, targets, _, _) in enumerate(data):
        if index >= batches:
            break
        logits, _ = model(inputs.to(device))
        values.append(eos_weighted_loss(logits, targets.to(device), eos_id, eos_weight).item())
    model.train()
    return sum(values) / max(1, len(values))


def meaningful_fragments(expected: str) -> set[str]:
    stop = {"ます", "です", "して", "から", "こと", "ため", "まず", "その", "確認", "大学"}
    return {expected[index:index + 2] for index in range(max(0, len(expected) - 1))
            if expected[index:index + 2] not in stop}


@torch.inference_mode()
def development_probe(model, tokenizer, device: str, validation_file: str) -> dict:
    rows = read_jsonl(validation_file)
    selected = [rows[index * len(rows) // 8] for index in range(8)]
    outputs = []
    model.eval()
    for row in selected:
        context = row.get("context") if isinstance(row.get("context"), str) else None
        context_block = f"<CONTEXT>\n{context}\n" if context else ""
        prompt = (f"<BOS><SYSTEM>\n{row['system']}\n{context_block}<USER>\n{row['user']}\n"
                  "<ASSISTANT>\n")
        text, metrics = generate_text(model, tokenizer, prompt, 64, temperature=0.0,
                                      top_k=40, top_p=0.9, repetition_penalty=1.1)
        grams = [text[index:index + 3] for index in range(max(0, len(text) - 2))]
        repetition = 0.0 if not grams else 1 - len(set(grams)) / len(grams)
        fragments = meaningful_fragments(row["assistant"])
        hits = sum(fragment in text for fragment in fragments)
        natural = (len(text.strip()) >= 20 and "�" not in text
                   and not any(ord(char) < 9 for char in text) and repetition < .35)
        outputs.append({
            "id": row["id"], "question": row["user"], "expected": row["assistant"], "text": text,
            "characters": len(text.strip()), "natural": natural, "expected_bigram_hits": hits,
            "repetition_rate": round(repetition, 4), **metrics,
        })
    model.train()
    count = len(outputs)
    formed = sum(row["natural"] for row in outputs) / count >= .75
    return {
        "questions": count,
        "natural_rate": sum(row["natural"] for row in outputs) / count,
        "mean_characters": sum(row["characters"] for row in outputs) / count,
        "mean_expected_bigram_hits": sum(row["expected_bigram_hits"] for row in outputs) / count,
        "eos_rate": sum(row["eos_reached"] for row in outputs) / count,
        "mean_tokens_per_second": sum(row["tokens_per_sec"] for row in outputs) / count,
        "text_generation_established": formed,
        "continue_to_500_recommended": formed,
        "rows": outputs,
        "scope": "Fixed validation development probe; independent Blind 200 remains sealed.",
    }


def save_checkpoint(output: Path, model, optimizer, step: int, loss: float, settings: dict,
                    stats: dict, probe: dict) -> dict:
    model.config.model_name = f"UniPilot Standard 50M short step {step}"
    manifest = {
        "model": model.config.model_name,
        "parameters": model.parameter_count(),
        "step": step,
        "initialization": "scratch-no-pretrained-model",
        "tokenizer_version": settings["tokenizer_version"],
        "dataset_version": settings["dataset_version"],
        "evaluation_version": settings["evaluation_version"],
        "seed": settings["seed"],
        "git_commit": git_commit(),
        "model_config": model.config.to_dict(),
        "training_metrics": stats,
        "development_generation_probe": probe,
        "continue_recommended": probe["continue_to_500_recommended"],
        "production_promoted": False,
        "production_changed": False,
        "external_pretrained_model": False,
        "external_ai_api": "OFF",
    }
    training_path = output / f"checkpoint-step-{step}.pt"
    inference_path = output / f"checkpoint-step-{step}-inference.pt"
    torch.save({
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "step": step,
        "loss": loss, "config": model.config.to_dict(), "tokenizer_version": settings["tokenizer_version"],
        "standard_short_manifest": manifest,
    }, training_path)
    torch.save({
        "model_state": model.state_dict(), "step": step, "config": model.config.to_dict(),
        "tokenizer_version": settings["tokenizer_version"], "standard_short_manifest": manifest,
    }, inference_path)
    manifest["training_checkpoint_bytes"] = training_path.stat().st_size
    manifest["inference_checkpoint_bytes"] = inference_path.stat().st_size
    manifest_path = output / f"checkpoint-step-{step}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-standard-50m-short.json")
    parser.add_argument("--max-steps", type=int, choices=(100, 500), required=True)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir", default="checkpoints/standard-50m-short")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    if device == "cpu":
        torch.set_num_threads(max(1, args.cpu_threads))
    random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    tokenizer = BPETokenizer.load(settings["tokenizer"])
    config = ModelConfig(**settings["model"])
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("Standard short tokenizer/model vocabulary mismatch")
    model = UniPilotTransformer(config).to(device)
    optimizer = create_optimizer(model, settings["learning_rate"], settings["weight_decay"])
    step = 0
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        if int(payload.get("step", -1)) != 100:
            raise ValueError("500-step continuation requires the exact step-100 checkpoint")
        manifest = payload.get("standard_short_manifest", {})
        if not manifest.get("continue_recommended"):
            raise RuntimeError("step-100 generation gate did not permit continuation")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        step = int(payload["step"])
        del payload
    train_loader = loader(settings["train_file"], tokenizer, config, settings["batch_size"],
                          settings["seed"], True)
    validation_loader = loader(settings["validation_file"], tokenizer, config, settings["batch_size"],
                               settings["seed"] + 1, False)
    iterator = iter(train_loader)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training-log.csv"
    fields = ("step", "loss", "validation_loss", "learning_rate", "gradient_norm",
              "gradient_clipping_count", "tokens_per_second", "step_time_seconds", "memory_usage_mb")
    mode = "a" if args.resume and log_path.exists() else "w"
    process = psutil.Process()
    recent: list[float] = []
    clipping = 0
    with log_path.open(mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        while step < args.max_steps:
            try:
                inputs, targets, _, _ = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, targets, _, _ = next(iterator)
            lr = settings["learning_rate"] * warmup_cosine_multiplier(
                step, settings["warmup_steps"], settings["schedule_steps"], .1)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            inputs, targets = inputs.to(device), targets.to(device)
            started = time.perf_counter()
            logits, _ = model(inputs)
            loss = eos_weighted_loss(logits, targets, tokenizer.eos_id, settings["eos_weight"])
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite Standard short loss")
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"]))
            if not math.isfinite(norm):
                raise RuntimeError("non-finite Standard short gradient norm")
            clipping += int(norm > settings["gradient_clip"])
            optimizer.step()
            step += 1
            elapsed = time.perf_counter() - started
            recent.append(float(loss.item()))
            recent = recent[-100:]
            if step % settings["evaluation_interval"] == 0 or step == args.max_steps:
                validation = validation_loss(model, validation_loader, device, tokenizer.eos_id,
                                             settings["eos_weight"])
                stats = {
                    "step": step, "loss": sum(recent) / len(recent), "validation_loss": validation,
                    "learning_rate": lr, "gradient_norm": norm, "gradient_clipping_count": clipping,
                    "tokens_per_second": int((targets != -100).sum()) / max(elapsed, 1e-9),
                    "step_time_seconds": elapsed, "memory_usage_mb": process.memory_info().rss / 1024**2,
                }
                probe = development_probe(model, tokenizer, device, settings["validation_file"])
                writer.writerow(stats)
                file.flush()
                manifest = save_checkpoint(output, model, optimizer, step, validation, settings, stats, probe)
                print(json.dumps({
                    "training": stats,
                    "probe": {key: value for key, value in probe.items() if key != "rows"},
                    "checkpoint": {key: manifest[key] for key in (
                        "training_checkpoint_bytes", "inference_checkpoint_bytes", "continue_recommended")},
                }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
