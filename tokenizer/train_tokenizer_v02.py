from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizer.tokenizer import BPETokenizer


def representative_texts(path: Path, per_kind: int) -> list[str]:
    selected = {"general": [], "university_text": [], "dialogue": []}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line); kind = row["kind"]
        if len(selected[kind]) >= per_kind: continue
        if kind == "dialogue": selected[kind].extend([row["user"], row["assistant"]])
        else: selected[kind].append(row["text"])
        if all(len(values) >= per_kind for values in selected.values()): break
    return [text for values in selected.values() for text in values[:per_kind]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/train/v02.jsonl")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--samples-per-kind", type=int, default=300)
    parser.add_argument("--output", default="tokenizer/vocab-v02-512.json")
    args = parser.parse_args()
    texts = representative_texts(Path(args.dataset), args.samples_per_kind)
    tokenizer = BPETokenizer(); tokenizer.train(texts, args.vocab_size); tokenizer.save(args.output)
    print(json.dumps({"vocab_size": tokenizer.vocab_size, "training_texts": len(texts), "output": args.output}))


if __name__ == "__main__": main()
