from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import psutil

from inference.generate import generate_text, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/sanity-100/checkpoint-step-100.pt")
    parser.add_argument("--history", default="checkpoints/sanity-100/training_history.csv")
    parser.add_argument("--output", default="evaluation/benchmark.json")
    args = parser.parse_args()
    process = psutil.Process(); before = process.memory_info().rss
    model, tokenizer, device, payload = load_model(args.checkpoint)
    after = process.memory_info().rss
    speeds = []
    samples = []
    for prompt in ["明日試験です", "大学の課題が終わらない", "単位が心配です"]:
        text, metrics = generate_text(model, tokenizer, f"<BOS><USER>\n{prompt}\n<ASSISTANT>\n", max_new_tokens=40)
        speeds.append(metrics["tokens_per_sec"]); samples.append({"prompt": prompt, "text": text, **metrics})
    training = {}
    history = Path(args.history)
    if history.exists():
        with history.open(encoding="utf-8") as file: rows = list(csv.DictReader(file))
        trained = [row for row in rows if row["train_loss"]]
        if trained:
            training = {"step_time_ms": float(trained[-1]["step_time_ms"]), "tokens_per_sec": float(trained[-1]["tokens_per_sec"])}
    result = {
        "model": model.config.model_name, "parameters": model.parameter_count(), "device": device,
        "checkpoint_bytes": Path(args.checkpoint).stat().st_size, "estimated_model_bytes_fp32": model.parameter_count() * 4,
        "process_rss_after_load_bytes": after, "model_load_rss_delta_bytes": max(0, after - before),
        "generation_tokens_per_sec_mean": statistics.mean(speeds), "training": training, "samples": samples,
        "checkpoint_step": payload.get("step"), "external_ai_api": "OFF",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__": main()
