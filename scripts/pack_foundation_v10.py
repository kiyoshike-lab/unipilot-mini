from __future__ import annotations

import argparse
from array import array
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.base_tokenizer import FoundationTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="evaluation/foundation-v10-tokenizer-benchmark.json")
    parser.add_argument("--output-dir", default="data/foundation_v10/packed")
    args = parser.parse_args()
    benchmark = json.loads((ROOT / args.benchmark).read_text(encoding="utf-8"))
    vocab = int(benchmark["selected_vocab"])
    tokenizer_paths = {
        1024: "tokenizer/foundation-v10-base-1024.json",
        2048: "tokenizer/foundation-v10-base-2048.json",
        4096: "tokenizer/foundation-v10-base-4096.json",
    }
    tokenizer_path = ROOT / tokenizer_paths[vocab]
    tokenizer = FoundationTokenizer.load(tokenizer_path)
    output = ROOT / args.output_dir / f"vocab-{vocab}"
    output.mkdir(parents=True, exist_ok=True)
    splits = {}
    total_by_source = Counter()
    total_by_category = Counter()
    for split in ("train", "validation", "test"):
        input_path = ROOT / f"data/foundation_v10/documents/{split}.jsonl.gz"
        output_path = output / f"{split}.bin"
        documents = characters = tokens = 0
        by_source = Counter()
        by_category = Counter()
        with gzip.open(input_path, "rt", encoding="utf-8") as source, output_path.open("wb") as target:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                ids = tokenizer.encode(row["text"], add_bos=True, add_eos=True)
                array("H", ids).tofile(target)
                documents += 1
                characters += len(row["text"])
                tokens += len(ids)
                by_source[row["source_type"]] += len(ids)
                by_category[row["category"]] += len(ids)
        total_by_source.update(by_source)
        total_by_category.update(by_category)
        splits[split] = {
            "path": output_path.relative_to(ROOT).as_posix(), "documents": documents,
            "characters": characters, "tokens": tokens, "bytes": output_path.stat().st_size,
            "sha256": sha256(output_path), "tokens_by_source": dict(by_source),
            "tokens_by_category": dict(by_category),
        }
    manifest = {
        "schema_version": "foundation-v10-packed-corpus-v1",
        "tokenizer": tokenizer_path.relative_to(ROOT).as_posix(), "vocab": vocab,
        "dtype": "uint16-native-little-endian", "document_boundary": "BOS ... EOS",
        "splits": splits, "total_documents": sum(row["documents"] for row in splits.values()),
        "total_characters": sum(row["characters"] for row in splits.values()),
        "total_tokens": sum(row["tokens"] for row in splits.values()),
        "tokens_by_source": dict(total_by_source), "tokens_by_category": dict(total_by_category),
        "pretraining_and_instruction_separated": True,
        "external_pretrained_model": False, "external_ai_api": "OFF",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
