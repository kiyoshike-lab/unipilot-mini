from __future__ import annotations

import argparse
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
    payload = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    config = ModelConfig(**payload["config"])
    tokenizer = BPETokenizer.load(tokenizer_path)
    model = UniPilotTransformer(config).to(resolved_device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    # Inferenceでは学習用optimizer stateと重複model stateを保持しない
    payload.pop("optimizer_state", None)
    payload.pop("model_state", None)

    import gc
    gc.collect()

    if resolved_device == "cpu":
        torch.set_num_threads(1)

    return model, tokenizer, resolved_device, payload


@torch.inference_mode()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8,
                  top_k: int = 40, top_p: float = 0.95, repetition_penalty: float = 1.1):
    ids = tokenizer.encode(prompt)
    prompt_length = len(ids)
    eos_reached = False
    started = time.perf_counter()
    device = next(model.parameters()).device
    for _ in range(max_new_tokens):
        context = ids[-model.config.context_length:]
        logits, _ = model(torch.tensor([context], dtype=torch.long, device=device))
        next_logits = apply_repetition_penalty(logits[0, -1], ids[-64:], repetition_penalty)
        next_id = sample_next_token(next_logits, temperature, top_k, top_p)
        ids.append(next_id)
        if next_id == tokenizer.eos_id:
            eos_reached = True
            break
    elapsed = time.perf_counter() - started
    generated = ids[prompt_length:]
    return tokenizer.decode(generated, skip_special=True), {"tokens": len(generated), "seconds": elapsed,
        "tokens_per_sec": len(generated) / max(elapsed, 1e-9), "eos_reached": eos_reached}


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
