from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys
import time
import torch

from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from tokenizer.tokenizer import BPETokenizer
from inference.sampling import apply_repetition_penalty, sample_next_token


def load_model(checkpoint_path: str, tokenizer_path: str = "tokenizer/vocab.json", device: str = "auto"):
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    configured_threads = os.getenv("UNIPILOT_CPU_THREADS")
    if resolved_device == "cpu" and (configured_threads or os.getenv("RENDER")):
        torch.set_num_threads(max(1, int(configured_threads or "1")))
    try:
        payload = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        payload = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    config = ModelConfig(**payload["config"])
    tokenizer = BPETokenizer.load(tokenizer_path)
    model = UniPilotTransformer(config).to(resolved_device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    # Inferenceでは学習用optimizer stateと重複model stateを保持しない
    payload.pop("optimizer_state", None)
    payload.pop("model_state", None)

    gc.collect()

    return model, tokenizer, resolved_device, payload


def iter_generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8,
                       top_k: int = 40, top_p: float = 0.95, repetition_penalty: float = 1.1,
                       stop_on_eos: bool = True):
    """Yield cumulative, UTF-8-safe decoded snapshots while using a per-request KV cache."""
    ids = tokenizer.encode(prompt)
    generated: list[int] = []
    eos_reached = False
    started = time.perf_counter()
    device = next(model.parameters()).device
    past_key_values = None
    cache_supported = hasattr(model, "blocks")
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            if cache_supported and past_key_values is not None and past_key_values[0][0].size(2) < model.config.context_length:
                model_input = torch.tensor([[ids[-1]]], dtype=torch.long, device=device)
            else:
                context = ids[-model.config.context_length:]
                model_input = torch.tensor([context], dtype=torch.long, device=device)
                past_key_values = None
            if cache_supported:
                logits, _, past_key_values = model(model_input, past_key_values=past_key_values, use_cache=True)
            else:
                logits, _ = model(model_input)
            next_logits = apply_repetition_penalty(logits[0, -1], ids[-64:], repetition_penalty)
            next_id = sample_next_token(next_logits, temperature, top_k, top_p)
            ids.append(next_id)
            generated.append(next_id)
            eos_reached = eos_reached or next_id == tokenizer.eos_id
            elapsed = time.perf_counter() - started
            yield {
                "text": tokenizer.decode(generated, skip_special=True),
                "token_id": next_id,
                "tokens": len(generated),
                "seconds": elapsed,
                "tokens_per_sec": len(generated) / max(elapsed, 1e-9),
                "eos_reached": eos_reached,
                "kv_cache": cache_supported,
            }
            if next_id == tokenizer.eos_id and stop_on_eos:
                break


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8,
                  top_k: int = 40, top_p: float = 0.95, repetition_penalty: float = 1.1,
                  stop_on_eos: bool = True):
    last = None
    for last in iter_generate_text(model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p,
                                   repetition_penalty, stop_on_eos):
        pass
    if last is None:
        return "", {"tokens": 0, "seconds": 0.0, "tokens_per_sec": 0.0, "eos_reached": False, "kv_cache": False}
    text = last.pop("text")
    last.pop("token_id", None)
    return text, last


def model_summary(model, device, payload):
    return (f"Model: {model.config.model_name}\nParameters: {model.parameter_count() / 1e6:.2f}M\n"
            f"Layers: {model.config.n_layers}\nHeads: {model.config.n_heads}\nContext: {model.config.context_length}\n"
            f"Device: {str(device).upper()}\nExternal AI API: OFF\nModel: Local\nStep: {payload.get('step', 0)}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/vocab.json")
    parser.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    model, tokenizer, device, payload = load_model(args.checkpoint, args.tokenizer, args.device)
    print(model_summary(model, device, payload))
    prompt = args.prompt or input("You: ")
    formatted = f"<BOS><USER>\n{prompt}\n<ASSISTANT>\n"
    text, metrics = generate_text(model, tokenizer, formatted, args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.repetition_penalty)
    print(f"\nUniPilot Mini:\n{text}\n\nGeneration: {metrics['tokens_per_sec']:.2f} tokens/sec")


if __name__ == "__main__":
    main()
