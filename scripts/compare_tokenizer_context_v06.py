from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from tokenizer.tokenizer import BPETokenizer
from training.dataset_v03 import SYSTEM_TEXT


def load_rows(path: str, limit: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


@torch.inference_mode()
def random_model_speed(vocab_size: int, steps: int = 4) -> float:
    config = ModelConfig(model_name=f"tokenizer probe {vocab_size}", vocab_size=vocab_size, context_length=256,
                         embedding_dim=384, n_layers=11, n_heads=6, ffn_dim=1536, dropout=0.0)
    model = UniPilotTransformer(config).eval()
    model(torch.randint(7, vocab_size, (1, 4)))
    ids = torch.randint(7, vocab_size, (1, 16))
    past = None
    started = time.perf_counter()
    for index in range(steps):
        model_input = ids if index == 0 else ids[:, -1:]
        logits, _, past = model(model_input, past_key_values=past, use_cache=True)
        ids = torch.cat((ids, logits[:, -1:].argmax(dim=-1)), dim=1)
    return steps / (time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/v06/instruction/train.jsonl")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--output", default="evaluation/tokenizer-context-candidates-v06.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = load_rows(args.dataset)
    sample = rows[:args.sample_size]
    train_texts = [value for row in sample for value in (row["user"], row["assistant"])]
    heldout = rows[-min(200, len(rows)):]
    heldout_texts = [value for row in heldout for value in (row["user"], row["assistant"])]
    tokenizer_results = []
    tokenizers = {}
    for size in (512, 1024, 2048, 4096):
        started = time.perf_counter()
        tokenizer = BPETokenizer()
        tokenizer.train(train_texts, size)
        train_seconds = time.perf_counter() - started
        tokenizers[size] = tokenizer
        counts = [len(tokenizer.encode(text)) for text in heldout_texts]
        characters = [len(text) for text in heldout_texts]
        roundtrip = [tokenizer.decode(tokenizer.encode(text)) == text for text in heldout_texts]
        byte_tokens = sum(token_id < 263 for text in heldout_texts for token_id in tokenizer.encode(text))
        all_tokens = sum(counts)
        tokenizer_results.append({
            "requested_vocab_size": size, "actual_vocab_size": tokenizer.vocab_size,
            "target_reached": tokenizer.vocab_size == size,
            "training_sample_texts": len(train_texts), "training_seconds": train_seconds,
            "mean_tokens": statistics.fmean(counts), "tokens_per_character": sum(counts) / max(1, sum(characters)),
            "byte_fragment_rate": byte_tokens / max(1, all_tokens), "exact_roundtrip_rate": sum(roundtrip) / len(roundtrip),
            "random_untrained_model_tokens_per_second": random_model_speed(tokenizer.vocab_size),
            "parameter_count_at_20m_architecture": UniPilotTransformer(ModelConfig(
                vocab_size=tokenizer.vocab_size, context_length=256, embedding_dim=384, n_layers=11,
                n_heads=6, ffn_dim=1536, dropout=0.0)).parameter_count(),
            "japanese_naturalness_note": "Compression and exact round-trip only; an untrained tokenizer cannot establish generated Japanese quality.",
        })
    current = BPETokenizer.load("tokenizer/vocab-v02-512.json")
    sequence_lengths = []
    for row in heldout:
        text = f"<BOS><SYSTEM>\n{SYSTEM_TEXT}\n<USER>\n{row['user']}\n<ASSISTANT>\n{row['assistant']}<EOS>"
        sequence_lengths.append(len(current.encode(text)))
    base_parameters = 19814784
    context_results = []
    for context in (256, 512, 1024):
        added_positions = (context - 256) * 384
        context_results.append({
            "context": context, "fit_rate": sum(length <= context for length in sequence_lengths) / len(sequence_lengths),
            "mean_sequence_tokens": statistics.fmean(sequence_lengths), "p95_sequence_tokens": sorted(sequence_lengths)[int(0.95 * (len(sequence_lengths) - 1))],
            "parameter_count": base_parameters + added_positions,
            "fp32_position_parameter_delta_mb": added_positions * 4 / 1024**2,
            "attention_score_memory_relative_to_256": (context / 256) ** 2,
        })
    report = {
        "tokenizer_candidates": tokenizer_results, "context_candidates": context_results,
        "mini_choice": {"vocab": 512, "context": 256,
                        "reason": "Required for exact v0.4 weight compatibility and the 512 MB production envelope."},
        "standard_candidate": {"vocab": 1024, "context": 512,
                               "reason": "Better compression/detail capacity, but requires scratch/converted training and more RAM."},
        "warning": "Random-model speed is an architecture cost probe, not a quality benchmark.",
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
