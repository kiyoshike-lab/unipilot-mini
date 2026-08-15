from __future__ import annotations

from collections import Counter
import unicodedata

from evaluation.metrics_v03 import aggregate as aggregate_v03


def ngram_repetition(text: str, n: int) -> float:
    compact = "".join(text.split())
    grams = [compact[index:index + n] for index in range(max(0, len(compact) - n + 1))]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def broken_generation_metrics(text: str) -> dict:
    chars = [char for char in text if not char.isspace()]
    invalid = 0
    try: text.encode("utf-8", errors="strict")
    except UnicodeError: invalid = 1
    byte_artifacts = sum(char == "�" or 0x80 <= ord(char) <= 0x9F for char in chars)
    symbols = sum(unicodedata.category(char).startswith("S") for char in chars)
    repeated_chars = sum(count >= 5 for count in Counter(chars).values())
    return {"broken_byte_rate": byte_artifacts / max(1, len(chars)), "symbol_noise_rate": symbols / max(1, len(chars)),
            "invalid_sequence_rate": float(invalid), "excessive_character_repetition": repeated_chars > 0}


def aggregate(rows: list[dict], max_new_tokens: int) -> dict:
    result = aggregate_v03(rows, max_new_tokens)
    result.update({
        "too_short_response_rate": sum(len(row["answer"].strip()) < 5 for row in rows) / len(rows),
        "average_response_characters": sum(len(row["answer"].strip()) for row in rows) / len(rows),
        "bigram_repetition_rate": sum(row["ngram_repetition"]["2"] for row in rows) / len(rows),
        "trigram_repetition_rate": sum(row["ngram_repetition"]["3"] for row in rows) / len(rows),
        "fourgram_repetition_rate": sum(row["ngram_repetition"]["4"] for row in rows) / len(rows),
        "broken_byte_rate": sum(row["broken_generation"]["broken_byte_rate"] for row in rows) / len(rows),
        "symbol_noise_rate": sum(row["broken_generation"]["symbol_noise_rate"] for row in rows) / len(rows),
        "invalid_sequence_rate": sum(row["broken_generation"]["invalid_sequence_rate"] for row in rows) / len(rows),
    })
    return result
