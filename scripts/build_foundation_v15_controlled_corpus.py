from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.base_tokenizer import FoundationTokenizer


SENTENCE = re.compile(r"[^。！？\n]+[。！？]")
MARKUP = re.compile(r"(?:\[\[|\]\]|\{\{|\}\}|<[^>]+>|https?://)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidates(source: Path, tokenizer: FoundationTokenizer):
    with gzip.open(source, "rt", encoding="utf-8") as rows:
        for line in rows:
            document = json.loads(line)
            sentences = [match.group(0).strip() for match in SENTENCE.finditer(document["text"])]
            for sentence_index, sentence in enumerate(sentences):
                if MARKUP.search(sentence):
                    continue
                token_ids = tokenizer.encode(sentence)
                if not 20 <= len(token_ids) <= 200:
                    continue
                yield {
                    "document_id": document["id"],
                    "title": document["title"],
                    "category": document["category"],
                    "source_type": document["source_type"],
                    "source": document["source"],
                    "source_url": document["source_url"],
                    "license": document["license"],
                    "sentence_index": sentence_index,
                    "text": sentence,
                    "characters": len(sentence),
                    "tokens": len(token_ids),
                    "content_sha256": hashlib.sha256(sentence.encode("utf-8")).hexdigest(),
                    "token_ids": token_ids,
                    "split": "diagnostic_train_only",
                    "foundation_corpus_member": False,
                }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/foundation_v11/documents/train.jsonl.gz")
    parser.add_argument("--tokenizer", default="tokenizer/foundation-v11-base-4096.json")
    parser.add_argument("--output-dir", default="data/foundation_v15_diagnostic")
    parser.add_argument("--segments", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=15012026)
    args = parser.parse_args()

    source = ROOT / args.source
    tokenizer_path = ROOT / args.tokenizer
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = FoundationTokenizer.load(tokenizer_path)
    selected = list(candidates(source, tokenizer))
    random.Random(args.seed).shuffle(selected)
    selected = selected[:args.segments]
    if len(selected) != args.segments:
        raise RuntimeError(f"only {len(selected)} eligible diagnostic segments")

    documents_path = output / "segments.jsonl.gz"
    packed_paths = {
        "train": output / "train.bin",
        "validation": output / "validation.bin",
    }
    packed: dict[str, list[int]] = {"train": [], "validation": []}
    category_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    with gzip.open(documents_path, "wt", encoding="utf-8", newline="\n") as sink:
        for index, row in enumerate(selected):
            token_ids = row.pop("token_ids")
            diagnostic_split = "train" if index < int(args.segments * 0.9) else "validation"
            row["split"] = f"diagnostic_{diagnostic_split}"
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            packed[diagnostic_split].extend([tokenizer.bos_id, *token_ids, tokenizer.eos_id])
            category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1
            source_counts[row["source_type"]] = source_counts.get(row["source_type"], 0) + 1
    for split, path in packed_paths.items():
        np.asarray(packed[split], dtype=np.uint16).tofile(path)

    manifest = {
        "schema_version": "foundation-v15-controlled-diagnostic-corpus-v1",
        "purpose": "isolated short-sentence language-model diagnostic only",
        "source_split": "Foundation v1.1 clean train only",
        "source_documents": args.source,
        "source_documents_sha256": sha256(source),
        "tokenizer": args.tokenizer,
        "tokenizer_sha256": sha256(tokenizer_path),
        "selection": {
            "seed": args.seed,
            "segments": len(selected),
            "minimum_tokens": min(row["tokens"] for row in selected),
            "maximum_tokens": max(row["tokens"] for row in selected),
            "mean_tokens": sum(row["tokens"] for row in selected) / len(selected),
            "sentence_boundaries_required": True,
            "markup_rejected": True,
        },
        "category_counts": dict(sorted(category_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "artifacts": {
            "documents": {
                "path": documents_path.relative_to(ROOT).as_posix(),
                "bytes": documents_path.stat().st_size,
                "sha256": sha256(documents_path),
            },
            "packed": {
                split: {
                    "path": path.relative_to(ROOT).as_posix(),
                    "dtype": "uint16-native-little-endian",
                    "tokens": len(packed[split]),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "document_boundary": "BOS ... EOS",
                }
                for split, path in packed_paths.items()
            },
        },
        "source_metadata_preserved": True,
        "added_to_foundation_corpus": False,
        "external_ai_api": "OFF",
        "final_blind_used": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
