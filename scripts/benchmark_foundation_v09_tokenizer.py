from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import statistics
import time

import psutil
import torch

from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from tokenizer.tokenizer import BPETokenizer


ROOT = Path(__file__).resolve().parents[1]
TERMS = ("GPA", "履修登録", "教授メール", "卒業論文", "単位認定", "生成AI", "情報リテラシー",
         "インターンシップ", "奨学金", "標準偏差", "参考文献", "研究室配属")


def read_jsonl(path: Path, limit: int) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
        if len(rows) == limit:
            break
    return rows


def training_texts() -> list[str]:
    texts = [row["text"] for row in read_jsonl(ROOT / "data/foundation_v09/base/train.jsonl", 160)]
    texts.extend(row["text"] for row in read_jsonl(ROOT / "data/foundation_v09/campus/train.jsonl", 100))
    for row in read_jsonl(ROOT / "data/foundation_v09/instruction/train.jsonl", 140):
        texts.extend((row["user"], row["assistant"]))
    return texts


def continue_bpe(source: BPETokenizer, texts: list[str], target: int) -> BPETokenizer:
    sequences = [source.encode(text) for text in texts if text]
    next_id = source.vocab_size
    while next_id < target:
        pairs = Counter((sequence[index], sequence[index + 1])
                        for sequence in sequences for index in range(len(sequence) - 1))
        if not pairs:
            break
        pair, frequency = pairs.most_common(1)[0]
        if frequency < 2:
            break
        updated = []
        for sequence in sequences:
            merged, index = [], 0
            while index < len(sequence):
                if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                    merged.append(next_id)
                    index += 2
                else:
                    merged.append(sequence[index])
                    index += 1
            updated.append(merged)
        sequences = updated
        source.token_bytes[next_id] = source.token_bytes[pair[0]] + source.token_bytes[pair[1]]
        source.merges.append((pair[0], pair[1], next_id))
        next_id += 1
    source._refresh()
    return source


@torch.inference_mode()
def speed(vocab_size: int) -> dict:
    torch.manual_seed(509)
    config = ModelConfig(model_name="foundation-v09-tokenizer-probe", vocab_size=vocab_size,
                         context_length=512, embedding_dim=512, n_layers=14, n_heads=8,
                         ffn_dim=2048, dropout=0.0)
    model = UniPilotTransformer(config).eval()
    ids = torch.randint(8, vocab_size, (1, 96))
    model(ids)
    past = None
    started = time.perf_counter()
    first = None
    for index in range(16):
        current = ids if index == 0 else ids[:, -1:]
        logits, _, past = model(current, past_key_values=past, use_cache=True)
        if first is None:
            first = time.perf_counter() - started
        ids = torch.cat((ids, logits[:, -1:].argmax(-1)), dim=1)
    elapsed = time.perf_counter() - started
    result = {"first_token_seconds": first, "generation_tokens_per_second": 16 / elapsed,
              "model_rss_mb": psutil.Process().memory_info().rss / 1024**2}
    del model, ids, past, logits
    gc.collect()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evaluation/foundation-v09-tokenizer-benchmark.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    validation = json.loads((ROOT / "data/foundation_v09/evaluation/validation-200.json").read_text(
        encoding="utf-8"))["items"]
    heldout = [row["question"] for row in validation]
    tokenizers = {
        1024: BPETokenizer.load(ROOT / "tokenizer/vocab-standard-v08-1024.json"),
        2048: BPETokenizer.load(ROOT / "tokenizer/vocab-standard-v08-2048.json"),
    }
    output_4096 = ROOT / "tokenizer/vocab-standard-v09-4096.json"
    started = time.perf_counter()
    tokenizers[4096] = continue_bpe(
        BPETokenizer.load(ROOT / "tokenizer/vocab-standard-v08-2048.json"), training_texts(), 4096)
    training_seconds = time.perf_counter() - started
    tokenizers[4096].save(output_4096)
    rows = []
    for size, tokenizer in tokenizers.items():
        counts = [len(tokenizer.encode(text)) for text in heldout]
        characters = [len(text) for text in heldout]
        probe = speed(tokenizer.vocab_size)
        rows.append({
            "requested_vocab": size, "actual_vocab": tokenizer.vocab_size,
            "tokens_per_japanese_character": sum(counts) / sum(characters),
            "mean_validation_question_tokens": statistics.fmean(counts),
            "p95_validation_question_tokens": sorted(counts)[int(.95 * (len(counts) - 1))],
            "mean_university_term_tokens": statistics.fmean(len(tokenizer.encode(term)) for term in TERMS),
            "exact_roundtrip_rate": sum(tokenizer.decode(tokenizer.encode(text)) == text for text in heldout) / len(heldout),
            "standard_parameters_context512": 44_396_544 + tokenizer.vocab_size * 512,
            "inference_checkpoint_mb_fp32": (44_396_544 + tokenizer.vocab_size * 512) * 4 / 1024**2,
            **probe,
        })
    by_size = {row["actual_vocab"]: row for row in rows}
    gain_2048 = 1 - by_size[2048]["tokens_per_japanese_character"] / by_size[1024]["tokens_per_japanese_character"]
    gain_4096 = 1 - by_size[4096]["tokens_per_japanese_character"] / by_size[2048]["tokens_per_japanese_character"]
    selected = 2048
    if (gain_4096 >= .12
            and by_size[4096]["generation_tokens_per_second"] >= .85 * by_size[2048]["generation_tokens_per_second"]):
        selected = 4096
    report = {
        "schema_version": "foundation-v09-tokenizer-benchmark-v1", "results": rows,
        "compression_gain_2048_over_1024": gain_2048,
        "compression_gain_4096_over_2048": gain_4096,
        "selected_vocab": selected,
        "selection_rule": "Start at 2048; select 4096 only for >=12% extra compression and >=85% generation speed.",
        "tokenizer_4096_training_seconds": training_seconds,
        "tokenizer_4096_training_texts": len(training_texts()),
        "development_validation_only": True, "final_blind_opened": False,
        "external_pretrained_tokenizer": False, "external_ai_api": "OFF",
    }
    path = ROOT / args.output
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
