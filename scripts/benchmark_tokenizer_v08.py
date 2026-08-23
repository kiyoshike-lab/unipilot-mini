from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import psutil
import torch

from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from tokenizer.tokenizer import BPETokenizer


TERMS = (
    "GPA", "履修登録", "教授メール", "卒業論文", "単位認定", "生成AI", "情報リテラシー",
    "インターンシップ", "奨学金", "標準偏差", "参考文献", "研究室配属",
)


def load_texts(limit: int) -> tuple[list[str], list[str]]:
    paths = (
        "data/v08/curriculum/A/train.jsonl", "data/v08/curriculum/C/train.jsonl",
        "data/v08/curriculum/D/train.jsonl", "data/v08/curriculum/E/train.jsonl",
    )
    texts = []
    per_path = max(2, limit // len(paths))
    for path in paths:
        selected = 0
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            texts.extend([row["user"], row["assistant"]])
            selected += 2
            if selected >= per_path:
                break
    blind = json.loads(Path("data/v08/blind/evaluation.json").read_text(encoding="utf-8"))
    heldout = [row["prompt"] + "".join(row["expected_key_points"]) for row in blind]
    return texts[:limit], heldout


@torch.inference_mode()
def speed_probe(vocab_size: int, repeats: int = 3) -> tuple[float, float]:
    torch.manual_seed(808)
    config = ModelConfig(model_name="v08 tokenizer probe", vocab_size=vocab_size, context_length=512,
                         embedding_dim=512, n_layers=2, n_heads=8, ffn_dim=2048, dropout=0.0)
    model = UniPilotTransformer(config).eval()
    inputs = torch.randint(8, vocab_size, (1, 128))
    model(inputs)
    started = time.perf_counter()
    for _ in range(repeats):
        model(inputs)
    forward = repeats * inputs.numel() / (time.perf_counter() - started)
    past = None
    ids = inputs[:, :32]
    started = time.perf_counter()
    for index in range(12):
        current = ids if index == 0 else ids[:, -1:]
        logits, _, past = model(current, past_key_values=past, use_cache=True)
        ids = torch.cat((ids, logits[:, -1:].argmax(-1)), dim=1)
    generation = 12 / (time.perf_counter() - started)
    return forward, generation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-texts", type=int, default=1600)
    parser.add_argument("--output", default="evaluation/tokenizer-benchmark-v08.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    train, heldout = load_texts(args.training_texts)
    results = []
    process = psutil.Process()
    for size in (512, 1024, 2048):
        tokenizer = BPETokenizer()
        tokenizer.special_tokens.append("<CONTEXT>")
        tokenizer._refresh()
        started = time.perf_counter()
        tokenizer.train(train, size)
        train_seconds = time.perf_counter() - started
        output = Path(f"tokenizer/vocab-standard-v08-{size}.json")
        tokenizer.save(output)
        counts = [len(tokenizer.encode(text)) for text in heldout]
        characters = [len(text) for text in heldout]
        term_tokens = {term: len(tokenizer.encode(term)) for term in TERMS}
        forward, generation = speed_probe(tokenizer.vocab_size)
        embedding_output_parameters = tokenizer.vocab_size * 512
        results.append({
            "requested_vocab": size, "actual_vocab": tokenizer.vocab_size,
            "training_texts": len(train), "training_seconds": train_seconds,
            "tokens_per_japanese_character": sum(counts) / sum(characters),
            "mean_blind_prompt_plus_keypoint_tokens": statistics.fmean(counts),
            "p95_blind_prompt_plus_keypoint_tokens": sorted(counts)[int(0.95 * (len(counts) - 1))],
            "university_term_token_counts": term_tokens,
            "mean_university_term_tokens": statistics.fmean(term_tokens.values()),
            "embedding_output_parameters_tied": embedding_output_parameters,
            "embedding_delta_from_512_parameters": (tokenizer.vocab_size - 512) * 512,
            "embedding_delta_from_512_mb_fp32": (tokenizer.vocab_size - 512) * 512 * 4 / 1024**2,
            "two_layer_forward_tokens_per_second": forward,
            "two_layer_generation_tokens_per_second": generation,
            "tokenizer_file_bytes": output.stat().st_size,
            "rss_mb": process.memory_info().rss / 1024**2,
            "exact_roundtrip_rate": sum(tokenizer.decode(tokenizer.encode(text)) == text for text in heldout) / len(heldout),
        })
    selection = min(results, key=lambda row: (
        0 if row["actual_vocab"] == 1024 else 1,
        row["tokens_per_japanese_character"],
    ))
    report = {
        "results": results, "selected_vocab": selection["actual_vocab"],
        "decision": "1024 is the default balance target; 2048 is selected only if its compression gain exceeds its parameter and speed cost materially.",
        "external_pretrained_tokenizer": False,
    }
    # Apply the explicit material-gain gate after collecting all measurements.
    by_size = {row["actual_vocab"]: row for row in results}
    if 1024 in by_size and 2048 in by_size:
        gain = 1 - by_size[2048]["tokens_per_japanese_character"] / by_size[1024]["tokens_per_japanese_character"]
        report["compression_gain_2048_over_1024"] = gain
        report["selected_vocab"] = 2048 if gain >= 0.15 and by_size[2048]["two_layer_generation_tokens_per_second"] >= 0.9 * by_size[1024]["two_layer_generation_tokens_per_second"] else 1024
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
