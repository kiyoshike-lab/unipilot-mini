from __future__ import annotations

import json
from pathlib import Path
import re

from retrieval.bm25 import LocalBM25


def load_jsonl(path: str | Path) -> list[dict]:
    target = Path(path)
    if not target.exists():
        return []
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized(text: str) -> str:
    return re.sub(r"\s+|[。、！？!?]", "", text.lower())


class CampusFAQRetriever:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.by_id = {row["id"]: row for row in rows}
        indexed = [{"id": row["id"], "title": row["question"],
                    "text": row["question"] + " " + " ".join(row.get("keywords", []))} for row in rows]
        self.index = LocalBM25(indexed)

    @classmethod
    def from_jsonl(cls, path: str | Path = "data/campus_v1/faq/faq.jsonl") -> "CampusFAQRetriever":
        return cls(load_jsonl(path))

    def search(self, question: str, category: str, top_k: int = 3) -> list[dict]:
        candidates = self.index.search(question, top_k=max(30, top_k * 10))
        reranked = []
        for result in candidates:
            row = self.by_id[result["id"]]
            category_match = row["category"] == category
            exact = normalized(row["question"]) == normalized(question)
            score = float(result["score"]) * (1.65 if category_match else 0.55) + (25 if exact else 0)
            reranked.append({**row, "retrieval_score": score, "category_match": category_match, "exact": exact})
        reranked.sort(key=lambda row: (-row["retrieval_score"], not row["category_match"], row["id"]))
        selected = reranked[:top_k]
        if selected:
            first, second = selected[0]["retrieval_score"], selected[1]["retrieval_score"] if len(selected) > 1 else 0.0
            confidence = min(1.0, 0.28 + first / 24 + max(0.0, first - second) / 30 +
                             (0.18 if selected[0]["category_match"] else 0) + (0.35 if selected[0]["exact"] else 0))
            selected[0]["confidence"] = confidence
        return selected


class CampusUniversityKnowledge:
    """Loads only explicitly enabled, source-attributed university records."""

    def __init__(self, rows: list[dict]):
        self.rows = [row for row in rows if row.get("enabled") is True and row.get("source_url") and row.get("retrieved_at")]
        indexed = [{"id": row["id"], "title": row["title"], "text": row["text"]} for row in self.rows]
        self.index = LocalBM25(indexed) if indexed else None
        self.by_id = {row["id"]: row for row in self.rows}

    @classmethod
    def from_root(cls, root: str | Path = "knowledge/universities") -> "CampusUniversityKnowledge":
        rows = []
        for path in Path(root).glob("*/*.jsonl") if Path(root).exists() else []:
            rows.extend(load_jsonl(path))
        return cls(rows)

    def search(self, question: str, university: str | None, top_k: int = 3) -> list[dict]:
        if not university or self.index is None:
            return []
        results = []
        for result in self.index.search(question, top_k=max(20, top_k * 5)):
            row = self.by_id[result["id"]]
            if row.get("university") == university:
                results.append({**row, "retrieval_score": result["score"]})
        return results[:top_k]


class CampusPublicKnowledge:
    def __init__(self, rows: list[dict]):
        self.rows = [row for row in rows if row.get("source_url") and row.get("license")]
        indexed = [{"id": row["id"], "title": row["title"], "text": row["text"]} for row in self.rows]
        self.index = LocalBM25(indexed) if indexed else None
        self.by_id = {row["id"]: row for row in self.rows}

    @classmethod
    def from_jsonl(cls, path: str | Path = "data/campus_v1/knowledge/public.jsonl") -> "CampusPublicKnowledge":
        return cls(load_jsonl(path))

    def search(self, question: str, category: str, top_k: int = 3) -> list[dict]:
        if self.index is None:
            return []
        rows = []
        for result in self.index.search(question, top_k=max(12, top_k * 4)):
            row = self.by_id[result["id"]]
            score = result["score"] * (1.4 if row.get("category") == category else .8)
            rows.append({**row, "retrieval_score": score})
        rows.sort(key=lambda row: (-row["retrieval_score"], row["id"]))
        return rows[:top_k]
