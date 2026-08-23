from __future__ import annotations

from collections import Counter, defaultdict
import math
import re

from pipeline.campus_categories import CAMPUS_CATEGORIES, CAMPUS_KEYWORDS
from retrieval.bm25 import LocalBM25


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def char_ngrams(text: str) -> Counter:
    value = normalize(text)
    return Counter(value[index:index + size] for size in (2, 3, 4)
                   for index in range(max(0, len(value) - size + 1)))


class CampusRuleRouter:
    name = "rules"

    def predict(self, text: str) -> tuple[str, float, dict]:
        value = normalize(text)
        scores = {category: 0.0 for category in CAMPUS_CATEGORIES}
        for category, keywords in CAMPUS_KEYWORDS.items():
            scores[category] += sum(2.0 + min(4.0, len(keyword) / 2) for keyword in keywords if keyword in value)

        # High-precision tool routes are ordered before broad university topics.
        if "gpa" in value and any(word in value for word in ("計算", "出して", "だす", "算出")):
            scores["gpa"] += 20
        if any(word in value for word in ("何点必要", "必要点", "合格するには", "残り評価")):
            scores["grade_simulator"] += 24
        if "欠席" in value and any(word in value for word in ("メール", "連絡", "文面", "送る")):
            scores["absence_email"] += 24
        if "遅刻" in value and any(word in value for word in ("メール", "連絡", "文面", "送る")):
            scores["lateness_email"] += 24
        if any(word in value for word in ("提出遅延", "提出が遅れ", "締切に遅れ", "課題遅延")):
            scores["late_submission_email"] += 24
        if any(word in value for word in ("教授", "先生", "教員")) and any(word in value for word in ("メール", "文面", "件名", "送る")):
            scores["professor_email"] += 16
        if "課題" in value and any(word in value for word in ("優先", "どれから", "どっちから", "複数")):
            scores["assignment_priority"] += 24
        if any(word in value for word in ("締切整理", "期限整理", "締切一覧", "締切をまとめ")):
            scores["deadline_organizer"] += 24
        if any(word in value for word in ("レポート", "卒論")) and any(word in value for word in ("構成", "章立て", "アウトライン")):
            scores["report_outline"] += 24
        if any(word in value for word in ("プレゼン", "発表", "スライド")) and any(word in value for word in ("構成", "流れ", "アウトライン")):
            scores["presentation_outline"] += 24
        if "toeic" in value and any(word in value for word in ("計画", "勉強", "目標", "日")):
            scores["toeic_plan"] += 24
        if any(word in value for word in ("試験", "テスト")) and any(word in value for word in ("計画", "まで", "時間", "明日", "日後")):
            scores["study_plan"] += 18
        if any(word in value for word in ("うちの大学", "この大学", "学則")) and any(
                word in value for word in ("ある", "何回", "必要", "でき", "ルール")):
            scores["university_policy"] += 20

        best = max(scores, key=scores.get)
        maximum = scores[best]
        if maximum <= 0:
            return "general", 0.0, scores
        second = sorted(scores.values(), reverse=True)[1]
        confidence = min(1.0, 0.48 + maximum / 35 + (maximum - second) / 45)
        return best, confidence, scores


class CampusBM25Router:
    name = "bm25"

    def __init__(self, examples: list[dict]):
        rows = [{"id": row["id"], "title": row["question"], "text": row["question"]} for row in examples]
        self.categories = {row["id"]: example["category"] for row, example in zip(rows, examples)}
        self.index = LocalBM25(rows)

    def predict(self, text: str) -> tuple[str, float, dict]:
        scores = defaultdict(float)
        for rank, result in enumerate(self.index.search(text, top_k=9)):
            scores[self.categories[result["id"]]] += result["score"] / (rank + 1)
        if not scores:
            return "general", 0.0, {}
        best = max(scores, key=scores.get)
        return best, scores[best] / max(sum(scores.values()), 1e-9), dict(scores)


class CampusTfidfRouter:
    name = "tfidf"

    def __init__(self, examples: list[dict]):
        documents = [(row["category"], char_ngrams(row["question"])) for row in examples]
        frequency = Counter(term for _, counts in documents for term in counts)
        self.idf = {term: math.log((1 + len(documents)) / (1 + count)) + 1 for term, count in frequency.items()}
        totals, category_counts = defaultdict(Counter), Counter()
        for category, counts in documents:
            category_counts[category] += 1
            for term, count in counts.items():
                totals[category][term] += count * self.idf[term]
        self.centroids = {}
        for category, vector in totals.items():
            scaled = {term: value / category_counts[category] for term, value in vector.items()}
            self.centroids[category] = (scaled, math.sqrt(sum(value * value for value in scaled.values())))

    def predict(self, text: str) -> tuple[str, float, dict]:
        counts = char_ngrams(text)
        query = {term: count * self.idf.get(term, 0.0) for term, count in counts.items() if term in self.idf}
        query_norm = math.sqrt(sum(value * value for value in query.values()))
        scores = {}
        for category, (centroid, norm) in self.centroids.items():
            scores[category] = sum(value * centroid.get(term, 0.0) for term, value in query.items()) / max(query_norm * norm, 1e-9)
        if not scores or max(scores.values()) <= 0:
            return "general", 0.0, scores
        best = max(scores, key=scores.get)
        return best, scores[best], scores


class CampusHybridRouter:
    name = "hybrid"

    def __init__(self, examples: list[dict]):
        self.rules = CampusRuleRouter()
        self.bm25 = CampusBM25Router(examples)
        self.tfidf = CampusTfidfRouter(examples)

    def predict(self, text: str) -> tuple[str, float, dict]:
        rule_category, rule_confidence, rule_scores = self.rules.predict(text)
        bm_category, bm_confidence, bm_scores = self.bm25.predict(text)
        tfidf_category, tfidf_confidence, tfidf_scores = self.tfidf.predict(text)
        votes = Counter((rule_category, bm_category, tfidf_category))
        majority, count = votes.most_common(1)[0]
        if rule_confidence >= 0.78:
            selected, confidence, source = rule_category, rule_confidence, "rules"
        elif count >= 2:
            selected = majority
            supporting = [value for category, value in ((rule_category, rule_confidence), (bm_category, bm_confidence),
                                                          (tfidf_category, tfidf_confidence)) if category == selected]
            confidence, source = min(1.0, 0.45 + sum(supporting) / len(supporting) / 2), "majority"
        else:
            selected, confidence, source = tfidf_category, tfidf_confidence, "tfidf"
        return selected, confidence, {"source": source, "rules": rule_scores, "bm25": bm_scores,
                                      "tfidf": tfidf_scores, "votes": dict(votes)}
