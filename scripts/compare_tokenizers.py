"""Train scratch BPE candidates on the same sample and compare compression without replacing v0.1 vocab."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.analyze_tokenizer import load_texts, metrics
from tokenizer.tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/test/v02.jsonl")
    parser.add_argument("--train-dataset", default="data/train/v02.jsonl")
    parser.add_argument("--sizes", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--output", default="evaluation/tokenizer-candidates-v02.json")
    args = parser.parse_args()
    train_texts = load_texts(Path(args.train_dataset), args.sample_size)
    test_texts = load_texts(Path(args.dataset), args.sample_size)
    results = []
    for size in args.sizes:
        tokenizer = BPETokenizer(); tokenizer.train(train_texts, size)
        results.append(metrics(tokenizer, test_texts))
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
