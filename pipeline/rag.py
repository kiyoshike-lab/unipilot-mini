from __future__ import annotations

import json
from pathlib import Path

from retrieval.bm25 import LocalBM25


class KnowledgeRetriever:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.by_id = {row["id"]: row for row in documents}
        self.index = LocalBM25(documents)

    @classmethod
    def from_jsonl(cls, path: str | Path = "data/v07/knowledge/documents.jsonl") -> "KnowledgeRetriever":
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        return cls(rows)

    def retrieve(self, query: str, category: str, top_k: int = 3) -> list[dict]:
        candidates = self.index.search(query, top_k=max(20, top_k * 5))
        reranked = []
        for result in candidates:
            document = self.by_id[result["id"]]
            category_match = document.get("category") == category
            score = float(result["score"]) * (1.5 if category_match else 0.75)
            reranked.append({**document, "retrieval_score": score, "category_match": category_match})
        reranked.sort(key=lambda row: (-row["retrieval_score"], not row["category_match"], row["id"]))
        return reranked[:top_k]

    @staticmethod
    def grounded_answer(results: list[dict], category: str) -> str | None:
        for result in results:
            if result.get("category") == category and result.get("source_type") == "project_authored_faq":
                return result.get("answer") or result.get("text")
        return None
