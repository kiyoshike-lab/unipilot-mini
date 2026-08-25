from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import re
import subprocess
import time

import psutil
import torch
from torch.utils.data import DataLoader

from inference.sampling import apply_repetition_penalty
from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from tokenizer.tokenizer import BPETokenizer
from training.dataset_v03 import CurriculumDataset, dynamic_collate
from training.optimizer import create_optimizer
from training.scheduler import warmup_cosine_multiplier
from training.train_v04 import eos_weighted_loss


SYSTEM = (
    "あなたは大学生を支援する完全ローカルのUniPilotです。最初に質問へ直接答え、理由、具体策、"
    "次の行動を示します。不足情報や大学固有制度は推測せず、確認方法を案内します。"
)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def stage_for_step(stages: list[dict], step: int) -> dict:
    return next(stage for stage in stages if stage["start_step"] <= step < stage["end_step"])


def make_loader(stage: dict, split: str, tokenizer, config: ModelConfig,
                batch_size: int, seed: int) -> DataLoader:
    dataset = CurriculumDataset(Path(stage["data_dir"]) / f"{split}.jsonl", tokenizer,
                                config.context_length, assistant_only=True)
    generator = torch.Generator().manual_seed(seed + sum(map(ord, stage["name"])))
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == "train",
                      collate_fn=dynamic_collate, generator=generator, num_workers=0)


@torch.inference_mode()
def safe_generate(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> tuple[str, dict]:
    ids = tokenizer.encode(prompt)
    generated: list[int] = []
    past = None
    started = time.perf_counter()
    first = None
    eos = False
    forbidden = [index for index in range(len(tokenizer.special_tokens)) if index != tokenizer.eos_id]
    for _ in range(max_new_tokens):
        current = torch.tensor([ids if past is None else [ids[-1]]], dtype=torch.long,
                               device=next(model.parameters()).device)
        logits, _, past = model(current, past_key_values=past, use_cache=True)
        scores = apply_repetition_penalty(logits[0, -1], ids[-64:], 1.1).clone()
        scores[forbidden] = -torch.inf
        next_id = int(torch.argmax(scores).item())
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
        "seconds": elapsed, "tokens_per_second": len(generated) / max(elapsed, 1e-9),
        "eos_reached": eos, "special_tokens_suppressed_except_eos": True,
    }


def natural_text(text: str) -> tuple[bool, float]:
    value = text.strip()
    japanese = sum(bool(re.match(r"[ぁ-んァ-ヶー一-龥々]", char)) for char in value)
    grams = [value[index:index + 3] for index in range(max(0, len(value) - 2))]
    repetition = 0.0 if not grams else 1 - len(set(grams)) / len(grams)
    return (len(value) >= 20 and japanese / max(1, len(value)) >= .30 and repetition < .40), repetition


@torch.inference_mode()
def base_probe(model, tokenizer, validation_file: str) -> dict:
    rows = read_jsonl(validation_file)
    selected = [rows[index * len(rows) // 8] for index in range(8)]
    outputs = []
    model.eval()
    for row in selected:
        source = row["text"]
        prefix = source[:max(30, len(source) // 3)]
        text, metrics = safe_generate(model, tokenizer, f"<BOS>{prefix}", 64)
        natural, repetition = natural_text(text)
        outputs.append({"id": row["id"], "prefix": prefix, "text": text,
                        "characters": len(text.strip()), "natural": natural,
                        "repetition_rate": round(repetition, 4), **metrics})
    model.train()
    rate = sum(row["natural"] for row in outputs) / len(outputs)
    return {
        "kind": "base_language_continuation", "questions": len(outputs), "natural_rate": rate,
        "mean_characters": sum(row["characters"] for row in outputs) / len(outputs),
        "mean_first_token_seconds": sum(row["first_token_seconds"] for row in outputs) / len(outputs),
        "mean_tokens_per_second": sum(row["tokens_per_second"] for row in outputs) / len(outputs),
        "text_generation_established": rate >= .75, "rows": outputs,
        "scope": "Base validation only; final Blind 1000 remains unopened.",
    }


@torch.inference_mode()
def instruction_probe(model, tokenizer, validation_file: str) -> dict:
    rows = read_jsonl(validation_file)
    selected = [rows[index * len(rows) // 8] for index in range(8)]
    outputs = []
    model.eval()
    for row in selected:
        prompt = f"<BOS><SYSTEM>\n{row.get('system', SYSTEM)}\n<USER>\n{row['user']}\n<ASSISTANT>\n"
        text, metrics = safe_generate(model, tokenizer, prompt, 96)
        natural, repetition = natural_text(text)
        outputs.append({"id": row["id"], "question": row["user"], "text": text,
                        "characters": len(text.strip()), "natural": natural,
                        "repetition_rate": round(repetition, 4), **metrics})
    model.train()
    return {
        "kind": "instruction_answer", "questions": len(outputs),
        "natural_rate": sum(row["natural"] for row in outputs) / len(outputs),
        "mean_characters": sum(row["characters"] for row in outputs) / len(outputs),
        "mean_first_token_seconds": sum(row["first_token_seconds"] for row in outputs) / len(outputs),
        "mean_tokens_per_second": sum(row["tokens_per_second"] for row in outputs) / len(outputs),
        "rows": outputs, "scope": "Instruction validation only; final Blind 1000 remains unopened.",
    }


@torch.inference_mode()
def validation_loss(model, loader, device: str, eos_id: int, eos_weight: float,
                    batches: int = 12) -> float:
    model.eval()
    values = []
    for index, (inputs, targets, _, _) in enumerate(loader):
        if index >= batches:
            break
        logits, _ = model(inputs.to(device))
        values.append(eos_weighted_loss(logits, targets.to(device), eos_id, eos_weight).item())
    model.train()
    return sum(values) / max(1, len(values))


def save(output: Path, model, optimizer, step: int, settings: dict, stats: dict, probe: dict) -> dict:
    model.config.model_name = f"UniPilot Standard Foundation v0.9 step {step}"
    manifest = {
        "model": model.config.model_name, "parameters": model.parameter_count(), "step": step,
        "initialization": "scratch-no-pretrained-model", "tokenizer_version": settings["tokenizer_version"],
        "dataset_version": settings["dataset_version"], "evaluation_version": settings["evaluation_version"],
        "seed": settings["seed"], "git_commit": git_head(), "model_config": model.config.to_dict(),
        "training_metrics": stats, "development_probe": probe,
        "continue_recommended": bool(step == 100 and probe["text_generation_established"]),
        "final_blind_opened": False, "production_promoted": False, "production_changed": False,
        "external_pretrained_model": False, "external_ai_api": "OFF",
    }
    train_path = output / f"checkpoint-step-{step}.pt"
    inference_path = output / f"checkpoint-step-{step}-inference.pt"
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "step": step, "config": model.config.to_dict(), "foundation_v09_manifest": manifest}, train_path)
    torch.save({"model_state": model.state_dict(), "step": step, "config": model.config.to_dict(),
                "foundation_v09_manifest": manifest}, inference_path)
    manifest["training_checkpoint_bytes"] = train_path.stat().st_size
    manifest["inference_checkpoint_bytes"] = inference_path.stat().st_size
    (output / f"checkpoint-step-{step}.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-foundation-v09-sanity.json")
    parser.add_argument("--max-steps", type=int, choices=(100, 500), required=True)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir", default="checkpoints/foundation-v09-sanity")
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
        raise ValueError("Foundation v0.9 tokenizer/model mismatch")
    model = UniPilotTransformer(config).to(device)
    optimizer = create_optimizer(model, settings["learning_rate"], settings["weight_decay"])
    step = 0
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        manifest = payload.get("foundation_v09_manifest", {})
        if int(payload.get("step", -1)) != 100 or not manifest.get("continue_recommended"):
            raise RuntimeError("500-step continuation requires a passing exact step-100 checkpoint")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        step = 100
        del payload
    loaders, validation_loaders, iterators = {}, {}, {}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training-log.csv"
    fields = ("step", "stage", "loss", "validation_loss", "learning_rate", "gradient_norm",
              "gradient_clipping_count", "tokens_per_second", "step_time_seconds", "memory_usage_mb")
    mode = "a" if args.resume and log_path.exists() else "w"
    recent: list[float] = []
    clipping = 0
    process = psutil.Process()
    with log_path.open(mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        while step < args.max_steps:
            stage = stage_for_step(settings["stages"], step)
            name = stage["name"]
            if name not in loaders:
                loaders[name] = make_loader(stage, "train", tokenizer, config, settings["batch_size"], settings["seed"])
                validation_loaders[name] = make_loader(stage, "validation", tokenizer, config,
                                                       settings["batch_size"], settings["seed"] + 1)
                iterators[name] = iter(loaders[name])
            try:
                inputs, targets, _, _ = next(iterators[name])
            except StopIteration:
                iterators[name] = iter(loaders[name])
                inputs, targets, _, _ = next(iterators[name])
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
                raise RuntimeError("non-finite Foundation v0.9 loss")
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"]))
            if not math.isfinite(norm):
                raise RuntimeError("non-finite Foundation v0.9 gradient")
            clipping += int(norm > settings["gradient_clip"])
            optimizer.step()
            step += 1
            elapsed = time.perf_counter() - started
            recent.append(float(loss.item()))
            recent = recent[-100:]
            if step % settings["evaluation_interval"] == 0 or step == args.max_steps:
                validation = validation_loss(model, validation_loaders[name], device,
                                             tokenizer.eos_id, settings["eos_weight"])
                stats = {"step": step, "stage": name, "loss": sum(recent) / len(recent),
                         "validation_loss": validation, "learning_rate": lr, "gradient_norm": norm,
                         "gradient_clipping_count": clipping,
                         "tokens_per_second": int((targets != -100).sum()) / max(elapsed, 1e-9),
                         "step_time_seconds": elapsed, "memory_usage_mb": process.memory_info().rss / 1024**2}
                probe = (base_probe(model, tokenizer, str(Path(stage["data_dir"]) / "validation.jsonl"))
                         if step <= 200 else instruction_probe(
                             model, tokenizer, "data/foundation_v09/instruction/validation.jsonl"))
                writer.writerow(stats)
                file.flush()
                manifest = save(output, model, optimizer, step, settings, stats, probe)
                print(json.dumps({"training": stats,
                                  "probe": {key: value for key, value in probe.items() if key != "rows"},
                                  "checkpoint": {key: manifest[key] for key in (
                                      "training_checkpoint_bytes", "inference_checkpoint_bytes",
                                      "continue_recommended")}}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
