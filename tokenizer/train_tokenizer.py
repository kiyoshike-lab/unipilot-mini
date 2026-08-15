from __future__ import annotations

import argparse
import json
from pathlib import Path
from tokenizer.tokenizer import BPETokenizer


def load_texts(paths: list[str]) -> list[str]:
    texts = []
    for pattern in paths:
        for path in Path().glob(pattern):
            if path.suffix == ".jsonl":
                for line in path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    texts.extend(str(value) for value in row.values() if isinstance(value, str))
            else:
                texts.extend(path.read_text(encoding="utf-8").splitlines())
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", default=["data/**/*.jsonl", "data/**/*.txt"])
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--output", default="tokenizer/vocab.json")
    args = parser.parse_args()
    texts = load_texts(args.input)
    tokenizer = BPETokenizer()
    tokenizer.train(texts, args.vocab_size)
    tokenizer.save(args.output)
    print(f"trained byte-level BPE: {tokenizer.vocab_size} tokens from {len(texts)} texts")


if __name__ == "__main__":
    main()
