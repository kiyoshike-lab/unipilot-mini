from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import tempfile
import time

import psutil
import torch

from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from tokenizer.tokenizer import BPETokenizer


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024**2


def blind_texts() -> list[str]:
    payload = json.loads(Path("data/v08/blind/evaluation.json").read_text(encoding="utf-8"))
    return [row["prompt"] + "".join(row["expected_key_points"]) for row in payload]


@torch.inference_mode()
def generation_probe(model: UniPilotTransformer, vocab_size: int) -> dict:
    model.eval()
    prompt = torch.randint(8, vocab_size, (1, 96))
    ids = prompt
    past = None
    first = None
    started = time.perf_counter()
    for index in range(32):
        current = ids if index == 0 else ids[:, -1:]
        logits, _, past = model(current, past_key_values=past, use_cache=True)
        if first is None:
            first = time.perf_counter() - started
        ids = torch.cat((ids, logits[:, -1:].argmax(-1)), dim=1)
    elapsed = time.perf_counter() - started
    return {
        "tokens": 32,
        "first_token_seconds": first,
        "total_seconds": elapsed,
        "tokens_per_second": 32 / elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=int, choices=(1024, 2048), required=True)
    parser.add_argument("--context", type=int, choices=(512, 1024), required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", default="evaluation/standard-50m-short")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(500826)
    tokenizer_path = Path(f"tokenizer/vocab-standard-v08-{args.vocab}.json")
    tokenizer = BPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != args.vocab:
        raise RuntimeError("tokenizer vocabulary size mismatch")

    texts = blind_texts()
    counts = [len(tokenizer.encode(text)) for text in texts]
    characters = [len(text) for text in texts]
    before = rss_mb()
    config = ModelConfig(
        model_name=f"UniPilot Standard short vocab-{args.vocab} context-{args.context}",
        vocab_size=args.vocab, context_length=args.context, embedding_dim=512,
        n_layers=14, n_heads=8, ffn_dim=2048, dropout=0.0, bias=True,
    )
    model = UniPilotTransformer(config)
    after_model = rss_mb()
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / "inference.pt"
        torch.save({"model_state": model.state_dict(), "config": config.to_dict()}, checkpoint)
        checkpoint_bytes = checkpoint.stat().st_size

    short_generation = generation_probe(model, tokenizer.vocab_size)
    after_generation = rss_mb()
    inputs = torch.randint(8, tokenizer.vocab_size, (1, args.context))
    with torch.inference_mode():
        model(inputs[:, :16])
        started = time.perf_counter()
        model(inputs)
        full_context_seconds = time.perf_counter() - started
    after_full_context = rss_mb()
    result = {
        "vocab_size": args.vocab, "context_length": args.context,
        "parameters": model.parameter_count(), "parameter_millions": model.parameter_count() / 1e6,
        "tokenizer": str(tokenizer_path).replace("\\", "/"),
        "tokens_per_japanese_character": sum(counts) / sum(characters),
        "mean_prompt_plus_keypoint_tokens": statistics.fmean(counts),
        "p95_prompt_plus_keypoint_tokens": sorted(counts)[int(.95 * (len(counts) - 1))],
        "exact_roundtrip_rate": sum(tokenizer.decode(tokenizer.encode(text)) == text for text in texts) / len(texts),
        "generation": short_generation,
        "full_context_forward_seconds": full_context_seconds,
        "full_context_tokens_per_second": args.context / full_context_seconds,
        "rss_before_mb": before, "rss_after_model_mb": after_model,
        "rss_after_generation_mb": after_generation, "rss_after_full_context_mb": after_full_context,
        "peak_observed_rss_mb": max(before, after_model, after_generation, after_full_context),
        "checkpoint_bytes_fp32": checkpoint_bytes, "checkpoint_mb_fp32": checkpoint_bytes / 1024**2,
        "kv_cache_mb_fp32": 2 * config.n_layers * args.context * config.embedding_dim * 4 / 1024**2,
        "cpu_threads": args.threads, "external_ai_api": "OFF",
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"vocab-{args.vocab}-context-{args.context}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    del model, inputs
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
