from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import psutil
import torch

from model.config import ModelConfig
from model.transformer import UniPilotTransformer


ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(256, 512, 1024, 2048), required=True)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", default="evaluation/foundation-v09-context")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(509)
    process = psutil.Process()
    before = process.memory_info().rss / 1024**2
    config = ModelConfig(
        model_name=f"UniPilot Standard v0.9 context {args.context}", vocab_size=args.vocab,
        context_length=args.context, embedding_dim=512, n_layers=14, n_heads=8,
        ffn_dim=2048, dropout=0.0,
    )
    model = UniPilotTransformer(config).eval()
    after_model = process.memory_info().rss / 1024**2
    sequence = torch.randint(8, args.vocab, (1, args.context))
    started = time.perf_counter()
    model(sequence)
    full_seconds = time.perf_counter() - started
    after_forward = process.memory_info().rss / 1024**2
    ids = sequence[:, :min(96, args.context // 2)]
    past = None
    started = time.perf_counter()
    first = None
    for index in range(32):
        current = ids if index == 0 else ids[:, -1:]
        logits, _, past = model(current, past_key_values=past, use_cache=True)
        if first is None:
            first = time.perf_counter() - started
        ids = torch.cat((ids, logits[:, -1:].argmax(-1)), dim=1)
    generation_seconds = time.perf_counter() - started
    after_generation = process.memory_info().rss / 1024**2
    parameters = model.parameter_count()
    result = {
        "vocab": args.vocab, "context": args.context, "parameters": parameters,
        "full_context_forward_seconds": full_seconds,
        "full_context_tokens_per_second": args.context / full_seconds,
        "first_token_seconds": first,
        "generation_tokens_per_second": 32 / generation_seconds,
        "rss_before_mb": before, "rss_after_model_mb": after_model,
        "rss_after_full_context_mb": after_forward, "rss_after_generation_mb": after_generation,
        "peak_observed_rss_mb": max(after_model, after_forward, after_generation),
        "inference_checkpoint_mb_fp32": parameters * 4 / 1024**2,
        "kv_cache_mb_fp32": 2 * 14 * 8 * args.context * 64 * 4 / 1024**2,
        "cpu_threads": args.threads, "external_ai_api": "OFF",
    }
    output = ROOT / args.output_dir / f"context-{args.context}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
