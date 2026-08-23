from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import psutil
import torch

from model.config import ModelConfig
from model.transformer import UniPilotTransformer


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024**2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(256, 512, 1024), required=True)
    parser.add_argument("--output-dir", default="evaluation")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(808)
    before = rss_mb()
    config = ModelConfig(model_name=f"Standard v0.8 context-{args.context}", vocab_size=1024,
                         context_length=args.context, embedding_dim=512, n_layers=14, n_heads=8,
                         ffn_dim=2048, dropout=0.0)
    model = UniPilotTransformer(config).eval()
    after_model = rss_mb()
    inputs = torch.randint(8, config.vocab_size, (1, args.context))
    with torch.inference_mode():
        model(inputs[:, :16])
        started = time.perf_counter()
        model(inputs)
        elapsed = time.perf_counter() - started
    after_forward = rss_mb()
    result = {
        "context": args.context, "parameters": model.parameter_count(),
        "position_embedding_parameters": args.context * config.embedding_dim,
        "full_context_forward_seconds": elapsed,
        "full_context_forward_tokens_per_second": args.context / elapsed,
        "rss_before_mb": before, "rss_after_model_mb": after_model,
        "rss_after_full_forward_mb": after_forward, "full_forward_rss_delta_mb": after_forward - before,
        "attention_memory_relative_to_256": (args.context / 256) ** 2,
        "kv_cache_mb_fp32": 2 * config.n_layers * args.context * config.embedding_dim * 4 / 1024**2,
    }
    output = Path(args.output_dir) / f"context-standard-v08-{args.context}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
