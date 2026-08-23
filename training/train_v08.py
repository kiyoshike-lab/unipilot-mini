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


SYSTEM = (
    "あなたは大学生活支援に特化した完全ローカルのUniPilot Standardです。"
    "検索文脈に根拠がある場合はそれを使い、不足する情報は推測せず確認方法を案内します。"
)
PROBE_STOP_BIGRAMS = {"ます", "です", "して", "から", "こと", "ため", "確認", "大学", "場合", "情報", "進め", "その", "まず"}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def stage_for_step(stages: list[dict], step: int) -> dict:
    for stage in stages:
        if stage["start_step"] <= step < stage["end_step"]:
            return stage
    return stages[-1]


def loader_for(stage: dict, split: str, tokenizer, config, seed: int) -> DataLoader:
    dataset = CurriculumDataset(Path(stage["data_dir"]) / f"{split}.jsonl", tokenizer,
                                config.context_length, assistant_only=True)
    generator = torch.Generator().manual_seed(seed + ord(stage["name"]))
    return DataLoader(dataset, batch_size=1, shuffle=split == "train", collate_fn=dynamic_collate,
                      generator=generator, num_workers=0)


@torch.inference_mode()
def validation_loss(model, loader, device, eos_id: int, eos_weight: float, batches: int = 12) -> float:
    model.eval()
    values = []
    for index, (inputs, targets, _, _) in enumerate(loader):
        if index >= batches:
            break
        logits, _ = model(inputs.to(device))
        values.append(eos_weighted_loss(logits, targets.to(device), eos_id, eos_weight).item())
    model.train()
    return sum(values) / max(1, len(values))


@torch.inference_mode()
def blind_generation_probe(model, tokenizer, device: str) -> dict:
    blind = json.loads(Path("data/v08/blind/evaluation.json").read_text(encoding="utf-8"))
    selected = [blind[index] for index in range(0, len(blind), 66)][:8]
    rows = []
    model.eval()
    for item in selected:
        prompt = f"<BOS><SYSTEM>\n{SYSTEM}\n<CONTEXT>\n<USER>\n{item['prompt']}\n<ASSISTANT>\n"
        text, metrics = generate_text(model, tokenizer, prompt, 32, temperature=0.0, top_k=40, top_p=0.9,
                                      repetition_penalty=1.1)
        expected_fragments = [fragment for point in item["expected_key_points"]
                              for fragment in (point[index:index + 2] for index in range(max(0, len(point) - 1)))
                              if fragment not in PROBE_STOP_BIGRAMS]
        hits = sum(fragment in text for fragment in set(expected_fragments))
        rows.append({
            "id": item["id"], "category": item["category"], "text": text,
            "natural": bool(text.strip()) and "�" not in text,
            "complete": len(text.strip()) >= 12 and metrics["eos_reached"],
            "relevant": hits >= 4, "eos": metrics["eos_reached"],
            "tokens_per_second": metrics["tokens_per_sec"],
        })
    model.train()
    count = len(rows)
    return {
        "questions": count, "natural_rate": sum(row["natural"] for row in rows) / count,
        "completion_rate": sum(row["complete"] for row in rows) / count,
        "relevance_proxy_rate": sum(row["relevant"] for row in rows) / count,
        "eos_rate": sum(row["eos"] for row in rows) / count,
        "mean_tokens_per_second": sum(row["tokens_per_second"] for row in rows) / count,
        "rows": rows,
        "limit": "Eight-item automatic early-stop probe; not the 528-question blind evaluation or human score.",
    }


def save_checkpoint(path: Path, model, optimizer, step: int, loss: float, settings: dict,
                    stage: dict, stats: dict, probe: dict) -> None:
    manifest = {
        "model": "UniPilot Standard v0.8 candidate", "parameters": model.parameter_count(),
        "stage": stage["name"], "step": step, "initialization": "scratch-no-pretrained-model",
        "tokenizer_version": "unipilot-standard-byte-bpe-v08-1024", "dataset_version": settings["dataset_version"],
        "evaluation_version": settings["evaluation_version"], "seed": settings["seed"],
        "git_commit": git_commit(), "model_config": model.config.to_dict(), "training_metrics": stats,
        "blind_generation_probe": probe,
        "continue_recommended": (probe["relevance_proxy_rate"] >= 0.20 and probe["natural_rate"] >= 0.75
                                 and probe["completion_rate"] >= 0.25),
        "production_promoted": False, "render_free_target": False,
        "external_pretrained_model": False, "external_ai_api": "OFF",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "step": step,
        "loss": loss, "config": model.config.to_dict(),
        "tokenizer_version": "unipilot-standard-byte-bpe-v08-1024", "v08_manifest": manifest,
    }, path)
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unipilot-standard-v08.json")
    parser.add_argument("--max-steps", type=int, choices=(100, 500, 1000, 2000, 5000, 10000), required=True)
    parser.add_argument("--resume-run")
    parser.add_argument("--output-dir", default="checkpoints/standard-v08-scratch")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    tokenizer = BPETokenizer.load(settings["tokenizer"])
    config = ModelConfig(**settings["model"])
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("Standard tokenizer/model vocabulary mismatch")
    model = UniPilotTransformer(config).to(device)
    optimizer = create_optimizer(model, settings["learning_rate"], settings["weight_decay"])
    step = 0
    if args.resume_run:
        payload = torch.load(args.resume_run, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        step = int(payload["step"])
        del payload
    loaders, validation_loaders, iterators = {}, {}, {}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training_log.csv"
    fields = ("step", "stage", "loss", "validation_loss", "learning_rate", "gradient_norm",
              "gradient_clipping_count", "nan_count", "inf_count", "tokens_per_second",
              "step_time_seconds", "memory_usage_mb")
    mode = "a" if args.resume_run and log_path.exists() else "w"
    recent, clipping, nan, inf = [], 0, 0, 0
    process = psutil.Process()
    with log_path.open(mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        while step < args.max_steps:
            stage = stage_for_step(settings["stages"], step)
            active = [stage["name"], *stage.get("replay", {})]
            for name in active:
                if name not in loaders:
                    definition = next(row for row in settings["stages"] if row["name"] == name)
                    loaders[name] = loader_for(definition, "train", tokenizer, config, settings["seed"])
                    validation_loaders[name] = loader_for(definition, "validation", tokenizer, config, settings["seed"])
                    iterators[name] = iter(loaders[name])
            selector = random.Random(settings["seed"] + step * 1009).random()
            chosen, cumulative = stage["name"], 0.0
            for name, probability in stage.get("replay", {}).items():
                cumulative += probability
                if selector < cumulative:
                    chosen = name
                    break
            try:
                inputs, targets, _, _ = next(iterators[chosen])
            except StopIteration:
                iterators[chosen] = iter(loaders[chosen])
                inputs, targets, _, _ = next(iterators[chosen])
            local_step = step - stage["start_step"]
            duration = stage["end_step"] - stage["start_step"]
            lr = settings["learning_rate"] * warmup_cosine_multiplier(local_step, settings["warmup_steps"], duration, 0.1)
            for group in optimizer.param_groups:
                group["lr"] = lr
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            loss = eos_weighted_loss(logits, targets, tokenizer.eos_id, settings["eos_weight"])
            if torch.isnan(loss):
                nan += 1
                raise RuntimeError("NaN loss; Standard training stopped")
            if torch.isinf(loss):
                inf += 1
                raise RuntimeError("Inf loss; Standard training stopped")
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"]))
            if not math.isfinite(norm):
                raise RuntimeError("non-finite gradient norm; Standard training stopped")
            clipping += int(norm > settings["gradient_clip"])
            optimizer.step()
            step += 1
            elapsed = time.perf_counter() - started
            recent.append(loss.item())
            recent = recent[-100:]
            should_evaluate = step % settings["evaluation_interval"] == 0 or step == args.max_steps
            if should_evaluate:
                validation = validation_loss(model, validation_loaders[stage["name"]], device,
                                             tokenizer.eos_id, settings["eos_weight"])
                stats = {
                    "step": step, "stage": stage["name"], "loss": sum(recent) / len(recent),
                    "validation_loss": validation, "learning_rate": lr, "gradient_norm": norm,
                    "gradient_clipping_count": clipping, "nan_count": nan, "inf_count": inf,
                    "tokens_per_second": int((targets != -100).sum()) / max(elapsed, 1e-9),
                    "step_time_seconds": elapsed, "memory_usage_mb": process.memory_info().rss / 1024**2,
                }
                probe = blind_generation_probe(model, tokenizer, device)
                writer.writerow(stats)
                file.flush()
                print(json.dumps({"training": stats, "blind_probe": {k: v for k, v in probe.items() if k != "rows"}}), flush=True)
                model.config.model_name = f"UniPilot Standard v0.8-{stage['name'].lower()}-{step}"
                save_checkpoint(output / f"stage-{stage['name'].lower()}" / f"checkpoint-step-{step}.pt",
                                model, optimizer, step, validation, settings, stage, stats, probe)
                if step < args.max_steps and (probe["natural_rate"] < 0.25 or probe["relevance_proxy_rate"] == 0):
                    raise RuntimeError("blind generation early-stop gate failed; refusing longer training in this run")


if __name__ == "__main__":
    main()
