from __future__ import annotations

import argparse
from collections import Counter
import gc
import gzip
import json
from pathlib import Path
import statistics
import sys
import time

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation.base_tokenizer import FoundationTokenizer, train_tokenizer
from model.config import ModelConfig
from model.transformer import UniPilotTransformer


TOKENIZERS = {
    1024: "tokenizer/foundation-v10-base-1024.json",
    2048: "tokenizer/foundation-v10-base-2048.json",
    4096: "tokenizer/foundation-v10-base-4096.json",
}


def documents(split: str):
    path = ROOT / f"data/foundation_v10/documents/{split}.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def training_texts(characters_per_source: int):
    counts = Counter()
    for row in documents("train"):
        source = "wikipedia" if "wikipedia" in row["source_type"] else "wikibooks"
        if counts[source] >= characters_per_source:
            continue
        counts[source] += len(row["text"])
        yield row["text"]


@torch.inference_mode()
def speed(vocab: int) -> dict:
    torch.manual_seed(110)
    config = ModelConfig(model_name="foundation-v10-tokenizer-probe", vocab_size=vocab,
                         context_length=512, embedding_dim=384, n_layers=10,
                         n_heads=6, ffn_dim=1536, dropout=0.0)
    model = UniPilotTransformer(config).eval()
    ids = torch.randint(7, vocab, (1, 96))
    model(ids)
    started = time.perf_counter()
    first = None
    past = None
    for index in range(24):
        current = ids if index == 0 else ids[:, -1:]
        logits, _, past = model(current, past_key_values=past, use_cache=True)
        if first is None:
            first = time.perf_counter() - started
        ids = torch.cat((ids, logits[:, -1:].argmax(-1)), dim=1)
    elapsed = time.perf_counter() - started
    result = {
        "first_token_seconds": first,
        "generation_tokens_per_second": 24 / elapsed,
        "peak_process_rss_mb": psutil.Process().memory_info().rss / 1024**2,
        "probe_parameters": model.parameter_count(),
    }
    del model, ids, logits, past
    gc.collect()
    return result


def campus_questions() -> list[str]:
    payload = json.loads((ROOT / "data/foundation_v09/evaluation/validation-200.json").read_text(
        encoding="utf-8"))
    return [row["question"] for row in payload["items"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evaluation/foundation-v10-tokenizer-benchmark.json")
    parser.add_argument("--training-characters-per-source", type=int, default=500_000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    questions = campus_questions()
    audit = json.loads((ROOT / "evaluation/foundation-v10-data-audit.json").read_text(
        encoding="utf-8"))
    full_characters = int(audit["total_characters"])
    results = []
    for requested, relative in TOKENIZERS.items():
        tokenizer_started = time.perf_counter()
        tokenizer = train_tokenizer(
            training_texts(args.training_characters_per_source), requested
        )
        tokenizer_training_seconds = time.perf_counter() - tokenizer_started
        tokenizer.save(ROOT / relative)
        started = time.perf_counter()
        total_tokens = total_characters = document_count = 0
        by_source = Counter()
        by_category = Counter()
        frequencies = Counter()
        sample_texts: list[str] = []
        wikipedia_counts: list[tuple[int, int]] = []
        wikibooks_counts: list[tuple[int, int]] = []
        for split in ("validation", "test"):
            for row in documents(split):
                source = "wikipedia" if "wikipedia" in row["source_type"] else "wikibooks"
                ids = tokenizer.encode(row["text"], add_bos=True, add_eos=True)
                count = len(ids)
                characters = len(row["text"])
                total_tokens += count
                total_characters += characters
                document_count += 1
                frequencies.update(ids)
                by_source[row["source_type"]] += count
                by_category[row["category"]] += count
                target = wikipedia_counts if source == "wikipedia" else wikibooks_counts
                target.append((count, characters))
                if len(sample_texts) < 100:
                    sample_texts.append(row["text"])
        question_tokens = [len(tokenizer.encode(question)) for question in questions]
        learned_ids = range(263, tokenizer.vocab_size)
        used_learned = sum(frequencies[token_id] > 0 for token_id in learned_ids)
        scale = full_characters / total_characters
        low_frequency = sum(frequencies[token_id] * scale < 100 for token_id in learned_ids)
        results.append({
            "requested_vocab": requested, "actual_vocab": tokenizer.vocab_size,
            "sample_documents": document_count, "sample_characters": total_characters,
            "sample_tokens": total_tokens,
            "tokens_per_character": total_tokens / total_characters,
            "wikipedia_tokens_per_character": sum(x for x, _ in wikipedia_counts) /
                                                  sum(x for _, x in wikipedia_counts),
            "wikibooks_tokens_per_character": sum(x for x, _ in wikibooks_counts) /
                                                 sum(x for _, x in wikibooks_counts),
            "campus_question_mean_tokens": statistics.fmean(question_tokens),
            "campus_question_p95_tokens": sorted(question_tokens)[int(.95 * (len(question_tokens) - 1))],
            "sample_tokens_by_source": dict(by_source),
            "sample_tokens_by_category": dict(by_category),
            "estimated_full_corpus_tokens": round(total_tokens * scale),
            "learned_token_usage_rate": used_learned / max(1, tokenizer.vocab_size - 263),
            "learned_tokens_projected_below_100_full_corpus": low_frequency,
            "learned_tokens_projected_below_100_rate":
                low_frequency / max(1, tokenizer.vocab_size - 263),
            "exact_roundtrip_rate": sum(
                tokenizer.decode(tokenizer.encode(text)) == text for text in sample_texts
            ) / len(sample_texts),
            "tokenizer_training_characters": args.training_characters_per_source * 2,
            "tokenizer_training_seconds": tokenizer_training_seconds,
            "counting_seconds": time.perf_counter() - started,
            **speed(tokenizer.vocab_size),
        })
    by_vocab = {row["actual_vocab"]: row for row in results}
    gain_2048 = 1 - by_vocab[2048]["tokens_per_character"] / by_vocab[1024]["tokens_per_character"]
    gain_4096 = 1 - by_vocab[4096]["tokens_per_character"] / by_vocab[2048]["tokens_per_character"]
    selected = 2048
    high_compression_support = (
        gain_4096 >= .20
        and by_vocab[4096]["estimated_full_corpus_tokens"] / 4096 >= 5_000
    )
    if (gain_4096 >= .10
            and by_vocab[4096]["generation_tokens_per_second"] >=
                .85 * by_vocab[2048]["generation_tokens_per_second"]
            and (by_vocab[4096]["learned_tokens_projected_below_100_rate"] <= .15
                 or high_compression_support)):
        selected = 4096
    report = {
        "schema_version": "foundation-v10-tokenizer-benchmark-v1",
        "results": results, "compression_gain_2048_over_1024": gain_2048,
        "compression_gain_4096_over_2048": gain_4096,
        "selected_vocab": selected,
        "comparison_is_heldout_validation_and_test": True,
        "tokenizer_training_characters_per_source": args.training_characters_per_source,
        "selected_tokenizer_full_corpus_count_is_written_by": "scripts/pack_foundation_v10.py",
        "selection_rule": (
            "Default 2048; use 4096 only with >=10% extra compression, >=85% CPU speed, "
            "and either <=15% projected rare learned tokens or >=20% compression plus "
            ">=5,000 estimated corpus tokens per vocabulary entry."
        ),
        "tokenizers_trained_from_scratch_on_base_train_only": True,
        "external_pretrained_tokenizer": False, "external_ai_api": "OFF",
    }
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
