from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import time

import psutil
import torch

from inference.generate import iter_generate_text, load_model
from inference.sampling import apply_repetition_penalty, sample_next_token


PROMPTS = {
    "short": "GPAって何？",
    "medium": "大学のテストに遅刻したらどうすればいい？大学ごとの差にも注意して短く教えてください。",
    "long": (
        "来週は試験が二つ、レポートが一つ、アルバイトが二日あります。"
        "科目名や大学固有のルールは決めつけず、今日からの一般的な優先順位と確認先を簡潔に教えてください。"
    ),
}
SYSTEM = "あなたは大学生活を支援する完全ローカルのUniPilot Miniです。情報がない場合は推測せず、確認方法を案内します。"


def rss_mb(process: psutil.Process) -> float:
    return process.memory_info().rss / 1024**2


@torch.inference_mode()
def legacy_profile(model, tokenizer, prompt: str, max_new_tokens: int) -> dict:
    process = psutil.Process()
    tokenize_started = time.perf_counter()
    ids = tokenizer.encode(prompt)
    tokenize_seconds = time.perf_counter() - tokenize_started
    prompt_tokens = len(ids)
    forward_seconds = 0.0
    sampling_seconds = 0.0
    peak_rss = rss_mb(process)
    eos_reached = False
    generated_ids: list[int] = []
    device = next(model.parameters()).device
    started = time.perf_counter()
    for _ in range(max_new_tokens):
        context = ids[-model.config.context_length:]
        forward_started = time.perf_counter()
        logits, _ = model(torch.tensor([context], dtype=torch.long, device=device))
        forward_seconds += time.perf_counter() - forward_started
        sampling_started = time.perf_counter()
        next_logits = apply_repetition_penalty(logits[0, -1], ids[-64:], 1.1)
        next_id = sample_next_token(next_logits, temperature=0.0, top_k=40, top_p=0.9)
        sampling_seconds += time.perf_counter() - sampling_started
        ids.append(next_id)
        generated_ids.append(next_id)
        peak_rss = max(peak_rss, rss_mb(process))
        if next_id == tokenizer.eos_id:
            eos_reached = True
    elapsed = time.perf_counter() - started
    return {
        "prompt_tokens": prompt_tokens,
        "requested_tokens": max_new_tokens,
        "generated_tokens": len(generated_ids),
        "tokenizer_seconds": tokenize_seconds,
        "model_forward_seconds": forward_seconds,
        "sampling_seconds": sampling_seconds,
        "generation_seconds": elapsed,
        "seconds_per_token": elapsed / max(1, len(generated_ids)),
        "tokens_per_second": len(generated_ids) / max(elapsed, 1e-9),
        "peak_rss_mb": peak_rss,
        "eos_reached": eos_reached,
        "text": tokenizer.decode(generated_ids, skip_special=True),
    }


def optimized_profile(model, tokenizer, prompt: str, max_new_tokens: int) -> dict:
    process = psutil.Process()
    tokenize_started = time.perf_counter()
    prompt_tokens = len(tokenizer.encode(prompt))
    tokenize_seconds = time.perf_counter() - tokenize_started
    peak = rss_mb(process); last = None; first_token_seconds = None
    for snapshot in iter_generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens,
                                       temperature=0.0, top_k=40, top_p=0.9, repetition_penalty=1.1,
                                       stop_on_eos=False):
        last = snapshot; peak = max(peak, rss_mb(process))
        if first_token_seconds is None: first_token_seconds = snapshot["seconds"]
    assert last is not None
    elapsed = float(last["seconds"])
    return {
        "prompt_tokens": prompt_tokens,
        "requested_tokens": max_new_tokens,
        "generated_tokens": int(last["tokens"]),
        "tokenizer_seconds": tokenize_seconds,
        "generation_seconds": elapsed,
        "first_token_seconds": first_token_seconds,
        "seconds_per_token": elapsed / max(1, int(last["tokens"])),
        "tokens_per_second": float(last["tokens_per_sec"]),
        "peak_rss_mb": peak,
        "eos_reached": bool(last["eos_reached"]),
        "text": last["text"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile UniPilot Mini CPU inference without external services.")
    parser.add_argument("--checkpoint", default="checkpoints/v04-eos15/checkpoint-step-2000.pt")
    parser.add_argument("--tokenizer", default="tokenizer/vocab-v02-512.json")
    parser.add_argument("--mode", choices=["legacy", "optimized"], default="legacy")
    parser.add_argument("--threads", type=int, default=0, help="0 keeps the current PyTorch setting")
    parser.add_argument("--output", default="evaluation/inference-benchmark.json")
    args = parser.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    process = psutil.Process()
    rss_before_load = rss_mb(process)
    load_started = time.perf_counter()
    model, tokenizer, device, payload = load_model(args.checkpoint, args.tokenizer, "cpu")
    load_seconds = time.perf_counter() - load_started
    rss_after_load = rss_mb(process)

    # One warm-up makes prompt-length comparisons independent of oneDNN initialization.
    with torch.inference_mode():
        warm = torch.tensor([[tokenizer.bos_id]], dtype=torch.long)
        model(warm)

    profiler = legacy_profile if args.mode == "legacy" else optimized_profile
    cases = []
    for label, requested in [("short", 5), ("medium", 20), ("long", 64)]:
        formatted = f"<BOS><SYSTEM>\n{SYSTEM}\n<USER>\n{PROMPTS[label]}\n<ASSISTANT>\n"
        cases.append({"name": label, **profiler(model, tokenizer, formatted, requested)})

    speeds = [case["tokens_per_second"] for case in cases]
    result = {
        "schema_version": 1,
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "checkpoint_size_mb": Path(args.checkpoint).stat().st_size / 1024**2,
        "model": model.config.model_name,
        "parameters": model.parameter_count(),
        "device": device,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cpu_count_logical": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "load_seconds": load_seconds,
        "rss_before_load_mb": rss_before_load,
        "rss_after_load_mb": rss_after_load,
        "model_load_rss_delta_mb": rss_after_load - rss_before_load,
        "peak_rss_mb": max([rss_after_load, *(case["peak_rss_mb"] for case in cases)]),
        "mean_tokens_per_second": statistics.fmean(speeds),
        "bottleneck": "full prompt and all prior generated tokens are recomputed through every Transformer layer for each token" if args.mode == "legacy" else "see per-case timings",
        "checkpoint_metadata": {"step": payload.get("step"), "loss": payload.get("loss")},
        "cases": cases,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
