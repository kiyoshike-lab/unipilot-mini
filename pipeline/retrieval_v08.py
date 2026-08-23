from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re

from pipeline.classifier import BM25CategoryClassifier
from retrieval.bm25 import LocalBM25, tokens as bm25_tokens


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def char_ngrams(text: str, minimum: int = 2, maximum: int = 4) -> Counter:
    value = normalized(text)
    return Counter(value[index:index + size] for size in range(minimum, maximum + 1)
                   for index in range(max(0, len(value) - size + 1)))


class StandardHybridRetriever:
    """Dependency-free local lexical retrieval; no embedding model or external API."""

    def __init__(self, documents: list[dict], classifier_examples: list[dict]):
        self.documents = documents
        self.by_id = {row["id"]: row for row in documents}
        self.bm25 = LocalBM25(documents)
        self.classifier = BM25CategoryClassifier(classifier_examples)
        counts = [char_ngrams(row.get("title", "") + row.get("text", "")) for row in documents]
        frequency = Counter()
        for row_counts in counts:
            frequency.update(row_counts.keys())
        total = len(documents)
        self.idf = {term: math.log((1 + total) / (1 + count)) + 1 for term, count in frequency.items()}
        self.tfidf = []
        for row_counts in counts:
            vector = {term: count * self.idf[term] for term, count in row_counts.items()}
            norm = math.sqrt(sum(value * value for value in vector.values()))
            self.tfidf.append((vector, norm))
        self.keyword_sets = [set(bm25_tokens(row.get("title", "") + row.get("text", ""))) for row in documents]

    @classmethod
    def from_files(cls, knowledge: str | Path = "data/v08/knowledge/documents.jsonl",
                   classifier: str | Path = "data/v08/classifier/train.jsonl") -> "StandardHybridRetriever":
        def rows(path):
            return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
        return cls(rows(knowledge), rows(classifier))

    def predict_category(self, query: str) -> tuple[str, float]:
        category, confidence, _ = self.classifier.predict(query)
        return category, confidence

    def _bm25_scores(self, query: str) -> dict[str, float]:
        return {row["id"]: row["score"] for row in self.bm25.search(query, top_k=len(self.documents))}

    def _tfidf_scores(self, query: str) -> dict[str, float]:
        counts = char_ngrams(query)
        vector = {term: count * self.idf.get(term, 0.0) for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(value * value for value in vector.values()))
        scores = {}
        for document, (candidate, candidate_norm) in zip(self.documents, self.tfidf):
            dot = sum(value * candidate.get(term, 0.0) for term, value in vector.items())
            score = dot / max(norm * candidate_norm, 1e-9)
            if score > 0:
                scores[document["id"]] = score
        return scores

    def _keyword_scores(self, query: str) -> dict[str, float]:
        query_terms = set(bm25_tokens(query))
        if not query_terms:
            return {}
        return {document["id"]: len(query_terms & terms) / len(query_terms)
                for document, terms in zip(self.documents, self.keyword_sets) if query_terms & terms}

    @staticmethod
    def _scale(scores: dict[str, float]) -> dict[str, float]:
        maximum = max(scores.values(), default=0.0)
        return {key: value / maximum for key, value in scores.items()} if maximum else scores

    def search(self, query: str, top_k: int = 3, method: str = "hybrid") -> list[dict]:
        if method not in {"bm25", "tfidf", "keyword", "hybrid"}:
            raise ValueError(f"unsupported retrieval method: {method}")
        category, category_confidence = self.predict_category(query)
        bm25 = self._scale(self._bm25_scores(query))
        tfidf = self._scale(self._tfidf_scores(query)) if method in {"tfidf", "hybrid"} else {}
        keyword = self._scale(self._keyword_scores(query)) if method in {"keyword", "hybrid"} else {}
        ranked = []
        for document in self.documents:
            doc_id = document["id"]
            if method == "bm25":
                score = bm25.get(doc_id, 0.0)
            elif method == "tfidf":
                score = tfidf.get(doc_id, 0.0)
            elif method == "keyword":
                score = keyword.get(doc_id, 0.0)
            else:
                score = 0.55 * bm25.get(doc_id, 0.0) + 0.30 * tfidf.get(doc_id, 0.0) + 0.15 * keyword.get(doc_id, 0.0)
                if document.get("category") == category:
                    score += 0.20 + 0.10 * category_confidence
            if score > 0:
                ranked.append({**document, "retrieval_score": score, "predicted_category": category,
                               "category_confidence": category_confidence})
        ranked.sort(key=lambda row: (-row["retrieval_score"], row["id"]))
        return ranked[:top_k]
