from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import re
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from pipeline.campus_categories_v2 import CATEGORY_TO_LEVEL1, LEVEL1_GROUPS
from pipeline.campus_retrieval import load_jsonl
from pipeline.campus_retrieval_v21 import expand_query
from retrieval.bm25 import LocalBM25, tokens


KNOWLEDGE_FILES = (
    "data/campus_v22/knowledge/government.jsonl",
    "data/campus_v22/knowledge/university.jsonl",
    "data/campus_v22/knowledge/wikipedia.jsonl",
)

DYNAMIC_CATEGORIES = {
    "scholarship", "tuition", "part_time_job", "career_schedule", "internship",
    "registration", "university_policy", "gpa", "credit",
}


def _verification_age(value: str | None) -> int:
    if not value:
        return 10_000
    try:
        verified = datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            verified = date.fromisoformat(value[:10])
        except ValueError:
            return 10_000
    return max(0, (date.today() - verified).days)


def _sentence_units(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[。！？])\s*|\n+", text) if len(part.strip()) >= 18]


def _overlap_score(query: str, sentence: str) -> float:
    query_tokens = set(tokens(query))
    sentence_tokens = set(tokens(sentence))
    if not query_tokens or not sentence_tokens:
        return 0.0
    return len(query_tokens & sentence_tokens) / max(1, len(query_tokens))


class CampusKnowledgeRetrieverV22:
    """Local multi-stage retrieval with license, university and freshness guards."""

    def __init__(self, rows: list[dict]):
        self.rows = [row for row in rows if row.get("source_url") and row.get("license")]
        self.positions = {row["id"]: index for index, row in enumerate(self.rows)}
        documents = [f"{row['title']} {row['title']} {row['text']}" for row in self.rows]
        self.index = LocalBM25([
            {"id": row["id"], "title": row["title"], "text": row["text"]} for row in self.rows
        ]) if self.rows else None
        self.char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, sublinear_tf=True)
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 3), min_df=1, token_pattern=r"(?u)\b\w+\b", sublinear_tf=True,
        )
        self.char_matrix = self.char_vectorizer.fit_transform(documents) if documents else None
        self.word_matrix = self.word_vectorizer.fit_transform(documents) if documents else None

    @classmethod
    def from_files(cls, paths: tuple[str, ...] = KNOWLEDGE_FILES) -> "CampusKnowledgeRetrieverV22":
        rows: list[dict] = []
        for path in paths:
            rows.extend(load_jsonl(Path(path)))
        return cls(rows)

    def _bm25(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.rows), dtype=np.float64)
        if self.index is None:
            return scores
        results = self.index.search(query, top_k=min(150, len(self.rows)))
        maximum = max((float(item["score"]) for item in results), default=0.0)
        if maximum:
            for item in results:
                scores[self.positions[item["id"]]] = float(item["score"]) / maximum
        return scores

    def _allowed(self, university: str | None) -> np.ndarray:
        return np.array([
            not row.get("university_specific") or (
                bool(university) and row.get("university_name") == university
            )
            for row in self.rows
        ], dtype=bool)

    @staticmethod
    def _category_bonus(row: dict, category: str) -> float:
        if row.get("category") == category:
            return 0.14
        level1 = CATEGORY_TO_LEVEL1.get(category)
        if level1 and row.get("category") in LEVEL1_GROUPS[level1]:
            return 0.045
        return 0.0

    @staticmethod
    def _stale(row: dict) -> bool:
        maximum_age = 180 if row.get("category") in DYNAMIC_CATEGORIES else (
            365 if row.get("source_type", "").startswith("official_") else 730
        )
        return _verification_age(row.get("last_verified_at")) > maximum_age

    def search(
        self,
        question: str,
        category: str,
        *,
        university: str | None = None,
        top_k: int = 5,
        response_mode: str = "normal",
        threshold: float = 0.20,
    ) -> tuple[list[dict], dict]:
        started = time.perf_counter()
        if not self.rows or self.char_matrix is None or self.word_matrix is None:
            return [], {"latency_ms": 0.0, "candidates": 0, "method": "empty"}
        expanded = expand_query(question, category)
        char_original = linear_kernel(self.char_vectorizer.transform([question]), self.char_matrix).ravel()
        char_expanded = linear_kernel(self.char_vectorizer.transform([expanded]), self.char_matrix).ravel()
        word = linear_kernel(self.word_vectorizer.transform([question]), self.word_matrix).ravel()
        bm25 = self._bm25(expanded)
        priority = np.array([min(1.0, float(row.get("source_priority", 0)) / 50.0) for row in self.rows])
        freshness = np.array([0.0 if self._stale(row) else 1.0 for row in self.rows])
        category_bonus = np.array([self._category_bonus(row, category) for row in self.rows])
        compact_question = re.sub(r"\s+", "", question).lower()
        title_bonus = np.array([
            .45 if len(re.sub(r"\s+", "", row["title"])) >= 2
            and re.sub(r"\s+", "", row["title"]).lower() in compact_question else 0.0
            for row in self.rows
        ])
        scores = (
            .38 * char_original + .12 * char_expanded + .14 * word + .19 * bm25
            + .08 * priority + .04 * freshness + category_bonus + title_bonus
        )
        allowed = self._allowed(university)
        scores = np.where(allowed, scores, -1.0)
        ordered = np.argsort(-scores)[: min(50, len(scores))]
        selected: list[dict] = []
        seen_sources: set[str] = set()
        for position in ordered:
            score = float(scores[position])
            if score < threshold:
                continue
            row = self.rows[int(position)]
            # Keep evidence diverse once one page already contributes two chunks.
            source_count = sum(item["source_url"] == row["source_url"] for item in selected)
            if source_count >= 2:
                continue
            selected_text = self.select_context(question, row["text"], response_mode)
            if not selected_text:
                continue
            selected.append({
                **row,
                "retrieval_score": score,
                "selected_text": selected_text,
                "stale": self._stale(row),
                "score_components": {
                    "character": float(char_original[position]),
                    "word": float(word[position]),
                    "bm25": float(bm25[position]),
                    "source_priority": float(priority[position]),
                    "freshness": float(freshness[position]),
                    "category_bonus": float(category_bonus[position]),
                    "title_bonus": float(title_bonus[position]),
                },
            })
            seen_sources.add(row["source_url"])
            if len(selected) >= top_k:
                break
        return selected, {
            "latency_ms": (time.perf_counter() - started) * 1000,
            "candidates": int(sum(scores >= threshold)),
            "method": "bm25+word_tfidf+character_tfidf+router+priority+freshness",
            "query_expanded": expanded != question,
            "unique_sources": len(seen_sources),
        }

    @staticmethod
    def select_context(question: str, text: str, response_mode: str) -> str:
        budget = {"short": 260, "normal": 720, "detailed": 1250}.get(response_mode, 720)
        sentences = _sentence_units(text)
        if not sentences:
            return text[:budget]
        ranked = sorted(enumerate(sentences), key=lambda item: (-_overlap_score(question, item[1]), item[0]))
        chosen: list[tuple[int, str]] = []
        length = 0
        for position, sentence in ranked:
            if length and length + len(sentence) > budget:
                continue
            chosen.append((position, sentence))
            length += len(sentence)
            if length >= budget * .72 or len(chosen) >= 6:
                break
        if not chosen:
            chosen = [(0, sentences[0][:budget])]
        chosen.sort()
        return "".join(sentence for _, sentence in chosen)[:budget]


def detect_numeric_conflict(documents: list[dict]) -> bool:
    """Flag conflicting dynamic claims; a caller can then ask for official confirmation."""
    if len({row.get("source_url") for row in documents}) < 2:
        return False
    numbers_by_source = {
        row["source_url"]: set(re.findall(r"\d+(?:\.\d+)?(?:%|％|円|日|回|単位)", row.get("selected_text", "")))
        for row in documents
    }
    nonempty = [values for values in numbers_by_source.values() if values]
    return len(nonempty) >= 2 and not set.intersection(*nonempty)
