from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re


def tokens(text: str) -> list[str]:
    """Dependency-free Japanese-friendly character bigrams plus ASCII words."""
    normalized = re.sub(r"\s+", "", text.lower())
    japanese = re.sub(r"[^ぁ-んァ-ヶー一-龥々]", "", normalized)
    grams = [japanese[index:index + 2] for index in range(max(0, len(japanese) - 1))]
    return grams + re.findall(r"[a-z0-9]+", normalized)


class LocalBM25:
    def __init__(self, rows: list[dict], k1: float = 1.5, b: float = 0.75):
        self.rows = rows
        self.k1, self.b = k1, b
        self.term_counts = [Counter(tokens(row.get("title", "") + " " + row.get("text", ""))) for row in rows]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        document_frequency = Counter()
        for counts in self.term_counts:
            document_frequency.update(counts.keys())
        total = max(1, len(rows))
        self.idf = {term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
                    for term, frequency in document_frequency.items()}

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "LocalBM25":
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        return cls(rows)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_terms = Counter(tokens(query))
        scored = []
        for index, counts in enumerate(self.term_counts):
            score = 0.0
            length_norm = 1 - self.b + self.b * self.lengths[index] / max(1.0, self.average_length)
            for term, query_frequency in query_terms.items():
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / (frequency + self.k1 * length_norm)
                score *= 1 + math.log(query_frequency)
            if score > 0:
                row = self.rows[index]
                scored.append({"score": score, "id": row.get("id"), "title": row.get("title"),
                               "text": row.get("text", ""), "source_url": row.get("source_url"),
                               "license": row.get("license"), "usage_guard": row.get("usage_guard")})
        scored.sort(key=lambda item: (-item["score"], str(item["id"])))
        return scored[:top_k]
