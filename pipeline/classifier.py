from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
import time

from pipeline.categories import CATEGORIES, CATEGORY_KEYWORDS
from retrieval.bm25 import LocalBM25


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def char_ngrams(text: str, minimum: int = 2, maximum: int = 4) -> Counter:
    text = normalize_text(text)
    return Counter(text[index:index + size] for size in range(minimum, maximum + 1)
                   for index in range(max(0, len(text) - size + 1)))


class RuleCategoryClassifier:
    name = "rules"

    def predict(self, text: str) -> tuple[str, float, dict]:
        normalized = normalize_text(text)
        scores = {category: 0.0 for category in CATEGORIES}
        for category, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in normalized:
                    scores[category] += 1 + min(3, len(keyword) / 2)
        # Disambiguation rules have higher precision than generic keyword counts.
        if any(word in normalized for word in ("メール", "文面", "件名", "連絡文")):
            scores["professor_email"] += 10
        if "欠席" in normalized and "メール" not in normalized and "連絡文" not in normalized:
            scores["attendance"] += 4
        if "遅刻" in normalized:
            scores["lateness"] += 8
        if "gpa" in normalized:
            scores["gpa"] += 12
        if "単位" in normalized and "認定" not in normalized:
            scores["credit"] += 6
        best = max(scores, key=scores.get)
        maximum = scores[best]
        if maximum <= 0:
            return "general", 0.0, scores
        ordered = sorted(scores.values(), reverse=True)
        margin = maximum - ordered[1]
        confidence = min(1.0, 0.45 + maximum / 20 + margin / 20)
        return best, confidence, scores


class BM25CategoryClassifier:
    name = "bm25"

    def __init__(self, examples: list[dict]):
        rows = [{"id": row["id"], "title": row["question"], "text": row["question"],
                 "category": row["category"]} for row in examples]
        self.categories = {row["id"]: row["category"] for row in rows}
        self.index = LocalBM25(rows)

    def predict(self, text: str) -> tuple[str, float, dict]:
        results = self.index.search(text, top_k=7)
        scores = defaultdict(float)
        for rank, result in enumerate(results):
            scores[self.categories[result["id"]]] += result["score"] / (rank + 1)
        if not scores:
            return "general", 0.0, {}
        best = max(scores, key=scores.get)
        total = sum(scores.values())
        return best, scores[best] / max(total, 1e-9), dict(scores)


class TfidfCategoryClassifier:
    name = "tfidf"

    def __init__(self, examples: list[dict]):
        documents = [(row["category"], char_ngrams(row["question"])) for row in examples]
        frequency = Counter()
        for _, counts in documents:
            frequency.update(counts.keys())
        self.idf = {term: math.log((1 + len(documents)) / (1 + count)) + 1 for term, count in frequency.items()}
        vectors = defaultdict(Counter)
        category_counts = Counter()
        for category, counts in documents:
            category_counts[category] += 1
            for term, count in counts.items():
                vectors[category][term] += count * self.idf[term]
        self.centroids = {}
        for category, vector in vectors.items():
            scaled = {term: value / category_counts[category] for term, value in vector.items()}
            norm = math.sqrt(sum(value * value for value in scaled.values()))
            self.centroids[category] = (scaled, norm)

    def predict(self, text: str) -> tuple[str, float, dict]:
        counts = char_ngrams(text)
        query = {term: count * self.idf.get(term, 0.0) for term, count in counts.items() if term in self.idf}
        query_norm = math.sqrt(sum(value * value for value in query.values()))
        scores = {}
        for category, (centroid, norm) in self.centroids.items():
            dot = sum(value * centroid.get(term, 0.0) for term, value in query.items())
            scores[category] = dot / max(query_norm * norm, 1e-9)
        if not scores or max(scores.values()) <= 0:
            return "general", 0.0, scores
        best = max(scores, key=scores.get)
        return best, scores[best], scores


class HybridCategoryClassifier:
    name = "hybrid"

    def __init__(self, examples: list[dict]):
        self.rules = RuleCategoryClassifier()
        self.tfidf = TfidfCategoryClassifier(examples)

    def predict(self, text: str) -> tuple[str, float, dict]:
        rule_category, rule_confidence, rule_scores = self.rules.predict(text)
        tfidf_category, tfidf_confidence, tfidf_scores = self.tfidf.predict(text)
        if rule_confidence >= 0.72:
            return rule_category, rule_confidence, {"source": "rules", "rules": rule_scores, "tfidf": tfidf_scores}
        if rule_category == tfidf_category and rule_confidence > 0:
            return rule_category, min(1.0, 0.15 + rule_confidence + tfidf_confidence / 2), {
                "source": "agreement", "rules": rule_scores, "tfidf": tfidf_scores}
        return tfidf_category, tfidf_confidence, {"source": "tfidf", "rules": rule_scores, "tfidf": tfidf_scores}


def benchmark_classifier(classifier, items: list[dict], repeats: int = 1) -> dict:
    started = time.perf_counter()
    predictions = []
    for _ in range(repeats):
        predictions = [classifier.predict(item["prompt"])[0] for item in items]
    elapsed = time.perf_counter() - started
    correct = sum(prediction == item["category"] for prediction, item in zip(predictions, items))
    return {"method": classifier.name, "questions": len(items), "accuracy": correct / len(items),
            "mean_latency_ms": elapsed * 1000 / (len(items) * repeats), "predictions": predictions}
