from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from tokenizer.tokenizer import BPETokenizer


def load_texts(path: Path, limit: int = 5000):
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        texts.append(row.get("text") or row.get("user", ""))
        if len(texts) >= limit: break
    return texts


def metrics(tokenizer, texts):
    token_counts = [len(tokenizer.encode(text)) for text in texts]
    characters = [len(text) for text in texts]
    unknown = sum(tokenizer.encode(text).count(tokenizer.special_to_id["<UNK>"]) for text in texts)
    return {"sentences": len(texts), "vocab_size": tokenizer.vocab_size,
            "average_tokens_per_sentence": statistics.mean(token_counts),
            "average_tokens_per_character": sum(token_counts) / sum(characters),
            "unknown_token_usage": unknown, "compression_ratio_bytes_per_token": sum(len(text.encode("utf-8")) for text in texts) / sum(token_counts)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="tokenizer/vocab.json")
    parser.add_argument("--dataset", default="data/test/v02.jsonl")
    parser.add_argument("--output", default="evaluation/tokenizer-analysis-v02.json")
    args = parser.parse_args()
    result = metrics(BPETokenizer.load(args.tokenizer), load_texts(Path(args.dataset)))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
