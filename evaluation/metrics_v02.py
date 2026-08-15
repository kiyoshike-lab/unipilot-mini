from __future__ import annotations

import re


def repetition_rate(text: str, n: int = 3) -> float:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < n: return 0.0
    grams = [compact[index:index + n] for index in range(len(compact) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def japanese_character_ratio(text: str) -> float:
    visible = [character for character in text if not character.isspace()]
    if not visible: return 0.0
    japanese = sum(bool(re.match(r"[\u3040-\u30ff\u3400-\u9fff、。！？「」『』]", character)) for character in visible)
    return japanese / len(visible)


def keyword_relevance(text: str, keywords: list[str]) -> float:
    if not keywords: return 0.0
    return sum(keyword in text for keyword in keywords) / len(keywords)


def aggregate_generation_metrics(rows: list[dict]) -> dict:
    if not rows: return {}
    return {
        "response_not_empty": sum(bool(row["answer"].strip()) for row in rows) / len(rows),
        "eos_reached": sum(bool(row.get("eos_reached")) for row in rows) / len(rows),
        "repetition_rate": sum(row["repetition_rate"] for row in rows) / len(rows),
        "average_length": sum(len(row["answer"]) for row in rows) / len(rows),
        "keyword_relevance": sum(row["keyword_relevance"] for row in rows) / len(rows),
        "japanese_character_ratio": sum(row["japanese_character_ratio"] for row in rows) / len(rows),
        "repetition_failures": sum(row["repetition_rate"] >= 0.35 for row in rows),
    }
