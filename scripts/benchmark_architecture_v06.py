from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import psutil
import torch

from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from training.optimizer import create_optimizer


CANDIDATES = {
    "20m": dict(embedding_dim=384, n_layers=11, n_heads=6, ffn_dim=1536),
    "50m": dict(embedding_dim=512, n_layers=16, n_heads=8, ffn_dim=2048),
    "100m": dict(embedding_dim=640, n_layers=20, n_heads=8, ffn_dim=2560),
    "200m": dict(embedding_dim=768, n_layers=28, n_heads=12, ffn_dim=3072),
}


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024**2


@torch.inference_mode()
def generation_speed(model, steps: int = 4) -> float:
    model.eval()
    ids = torch.randint(7, model.config.vocab_size, (1, 16))
    past = None
    started = time.perf_counter()
    for index in range(steps):
        model_input = ids if index == 0 else ids[:, -1:]
        logits, _, past = model(model_input, past_key_values=past, use_cache=True)
        ids = torch.cat((ids, logits[:, -1:].argmax(dim=-1)), dim=1)
    return steps / (time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description="One optimizer-step sanity test for one isolated size candidate.")
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(6062026)
    before = rss_mb()
    config = ModelConfig(model_name=f"UniPilot candidate {args.candidate}", vocab_size=512, context_length=256,
                         dropout=0.0, bias=True, **CANDIDATES[args.candidate])
    model = UniPilotTransformer(config)
    parameters = model.parameter_count()
    after_model = rss_mb()
    optimizer = create_optimizer(model, learning_rate=1e-5, weight_decay=0.01)
    inputs = torch.randint(7, config.vocab_size, (1, args.sequence_length))
    targets = torch.randint(7, config.vocab_size, (1, args.sequence_length))
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(inputs, targets)
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    step_seconds = time.perf_counter() - started
    after_step = rss_mb()
    del optimizer, inputs, targets
    gc.collect()
    speed = generation_speed(model)
    result = {
        "candidate": args.candidate, "config": config.to_dict(), "parameters": parameters,
        "parameter_millions": parameters / 1e6, "sanity_train_steps": 1, "sanity_loss": loss.item(),
        "gradient_norm": gradient_norm, "step_seconds": step_seconds, "inference_tokens_per_second_4_token_probe": speed,
        "rss_before_mb": before, "rss_after_model_mb": after_model, "rss_after_optimizer_step_mb": after_step,
        "rss_delta_optimizer_step_mb": after_step - before, "estimated_fp32_inference_checkpoint_mb": parameters * 4 / 1024**2,
        "render_free_512mb_feasible": args.candidate == "20m",
        "feasibility_basis": "Only 20m leaves measured runtime headroom under 512 MB; larger candidates are design probes, not production candidates.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
