from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import tempfile
import time

import psutil
import torch

from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from training.optimizer import create_optimizer


CANDIDATES = {
    "a-45m": dict(embedding_dim=512, n_layers=14, n_heads=8, ffn_dim=2048),
    "b-51m": dict(embedding_dim=512, n_layers=16, n_heads=8, ffn_dim=2048),
    "c-58m": dict(embedding_dim=576, n_layers=14, n_heads=9, ffn_dim=2304),
}


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024**2


@torch.inference_mode()
def inference_probes(model, vocab_size: int) -> dict:
    model.eval()
    warm = torch.randint(8, vocab_size, (1, 16))
    model(warm)
    forward_inputs = torch.randint(8, vocab_size, (1, 256))
    started = time.perf_counter()
    model(forward_inputs)
    forward_seconds = time.perf_counter() - started
    prompt = torch.randint(8, vocab_size, (1, 96))
    ids, past = prompt, None
    first = None
    started = time.perf_counter()
    for index in range(16):
        current = ids if index == 0 else ids[:, -1:]
        logits, _, past = model(current, past_key_values=past, use_cache=True)
        if first is None:
            first = time.perf_counter() - started
        ids = torch.cat((ids, logits[:, -1:].argmax(-1)), dim=1)
    generation_seconds = time.perf_counter() - started
    return {
        "forward_sequence_tokens": 256,
        "forward_seconds": forward_seconds,
        "forward_tokens_per_second": 256 / forward_seconds,
        "generation_probe_tokens": 16,
        "generation_seconds": generation_seconds,
        "generation_tokens_per_second": 16 / generation_seconds,
        "first_token_seconds": first,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, choices=CANDIDATES)
    parser.add_argument("--output-dir", default="evaluation")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(8082026)
    before = rss_mb()
    values = CANDIDATES[args.candidate]
    config = ModelConfig(model_name=f"UniPilot Standard v0.8 {args.candidate}", vocab_size=1024,
                         context_length=512, dropout=0.0, bias=True, **values)
    model = UniPilotTransformer(config)
    parameters = model.parameter_count()
    after_model = rss_mb()
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint_path = Path(temporary) / "probe.pt"
        torch.save({"model_state": model.state_dict(), "config": config.to_dict()}, checkpoint_path)
        checkpoint_bytes = checkpoint_path.stat().st_size
    inference = inference_probes(model, config.vocab_size)
    after_inference = rss_mb()

    optimizer = create_optimizer(model, learning_rate=3e-4, weight_decay=0.01)
    inputs = torch.randint(8, config.vocab_size, (1, 128))
    targets = torch.randint(8, config.vocab_size, (1, 128))
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(inputs, targets)
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    training_step_seconds = time.perf_counter() - started
    after_step = rss_mb()
    head_dimension = config.embedding_dim // config.n_heads
    kv_cache_bytes = 2 * config.n_layers * config.context_length * config.embedding_dim * 4
    result = {
        "candidate": args.candidate, "config": config.to_dict(), "parameters": parameters,
        "parameter_millions": parameters / 1e6, "head_dimension": head_dimension,
        "context_512_supported": True, "inference": inference,
        "rss_before_mb": before, "rss_after_model_mb": after_model, "rss_after_inference_mb": after_inference,
        "rss_after_optimizer_step_mb": after_step, "model_rss_delta_mb": after_model - before,
        "training_rss_delta_mb": after_step - before, "training_sequence_length": 128,
        "training_step_seconds": training_step_seconds, "sanity_loss": float(loss),
        "gradient_norm": gradient_norm, "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_mb": checkpoint_bytes / 1024**2,
        "kv_cache_512_bytes_fp32": kv_cache_bytes, "kv_cache_512_mb_fp32": kv_cache_bytes / 1024**2,
        "estimated_kv_cache_1024_mb_fp32": 2 * kv_cache_bytes / 1024**2,
        "render_free_512mb_target": False,
    }
    del optimizer, inputs, targets, model
    gc.collect()
    output = Path(args.output_dir) / f"architecture-standard-v08-{args.candidate}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
