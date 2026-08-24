from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from time import perf_counter
from typing import Any

from pipeline.campus_categories import CAMPUS_CATEGORIES
from pipeline.campus_categories_v2 import CATEGORY_TO_LEVEL1, LEVEL1_GROUPS, TOOL_AVAILABLE
from pipeline.campus_router import CampusBM25Router, CampusRuleRouter, CampusTfidfRouter, normalize


AMBIGUOUS_ONLY = {
    "やばい", "どうしよう", "詰んだ", "たすけて", "助けて", "これ大丈夫", "間に合うかな",
    "大学つらい", "相談したい", "困ってます", "何からすればいい", "もう無理",
}

ACTION_WORDS = (
    "計算", "算出", "作って", "作成", "作り", "文面", "メール", "整理", "優先", "構成", "配分",
    "計画", "シミュレーション", "チェック", "割り振", "あと何日", "目標gpa",
)

DEFINITION_WORDS = ("とは", "って何", "意味", "仕組み", "違い", "どんなもの")

COMPOUND_SPLIT = re.compile(r"(?:それと|あと、|ついでに|加えて|および|ならびに|\s+and\s+)", re.I)
NEGATION_SPLIT = re.compile(r"(?:ではなく|じゃなくて|じゃなく|ではない。?)(.+)$")


def _clean_for_intent(text: str) -> str:
    value = text.lower().strip()
    match = NEGATION_SPLIT.search(value)
    if match and match.group(1).strip():
        value = match.group(1).strip()
    return value


def _softmax_confidence(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    ordered = sorted(values, reverse=True)
    top = ordered[0]
    second = ordered[1] if len(ordered) > 1 else top - 1
    gap = top - second
    confidence = 1 / (1 + math.exp(-gap))
    return confidence, gap


def _decision_scores(classifier: Any, matrix: Any) -> dict[str, float]:
    raw = classifier.decision_function(matrix)
    row = raw[0]
    if getattr(row, "ndim", 0) == 0:
        row = [-float(row), float(row)]
    return {str(label): float(score) for label, score in zip(classifier.classes_, row)}


@dataclass(frozen=True)
class CampusRouteDecision:
    primary: str
    intents: tuple[str, ...]
    level1: str
    top2: tuple[str, ...]
    confidence: float
    confidence_band: str
    action: str
    clarify_question: str | None
    latency_ms: float
    evidence: dict

    def to_dict(self) -> dict:
        return asdict(self)


class CampusSklearnRouter:
    """Evaluation router. sklearn stays in the opt-in Campus v2 dependency file."""

    def __init__(self, examples: list[dict], method: str = "char_svm", hierarchical: bool = False):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.pipeline import Pipeline
            from sklearn.svm import LinearSVC
        except ImportError as error:
            raise RuntimeError("Campus v2 requires: pip install -r requirements-campus-v2.txt") from error

        self.name = ("hierarchical_" if hierarchical else "flat_") + method
        self.method = method
        self.hierarchical = hierarchical
        self.examples = examples

        def estimator():
            if method == "logistic":
                return LogisticRegression(max_iter=1200, class_weight="balanced", C=4.0)
            if method == "naive_bayes":
                return MultinomialNB(alpha=0.08)
            return LinearSVC(C=2.2, class_weight="balanced")

        def vectorizer():
            if method in ("word_svm", "logistic", "naive_bayes"):
                return TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=1,
                                       sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
            return TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1,
                                   sublinear_tf=True, max_features=70000)

        def pipeline():
            return Pipeline((("tfidf", vectorizer()), ("classifier", estimator())))

        questions = [row["question"] for row in examples]
        categories = [row["category"] for row in examples]
        if not hierarchical:
            self.flat = pipeline().fit(questions, categories)
            return
        level1 = [CATEGORY_TO_LEVEL1[category] for category in categories]
        self.level1_model = pipeline().fit(questions, level1)
        self.level2_models = {}
        for group, allowed in LEVEL1_GROUPS.items():
            subset = [(question, category) for question, category in zip(questions, categories) if category in allowed]
            self.level2_models[group] = pipeline().fit([row[0] for row in subset], [row[1] for row in subset])

    @staticmethod
    def _scores(model: Any, text: str) -> dict[str, float]:
        matrix = model.named_steps["tfidf"].transform([text])
        classifier = model.named_steps["classifier"]
        if hasattr(classifier, "decision_function"):
            return _decision_scores(classifier, matrix)
        probabilities = classifier.predict_proba(matrix)[0]
        return {str(label): float(math.log(max(value, 1e-9)))
                for label, value in zip(classifier.classes_, probabilities)}

    def predict_scores(self, text: str) -> dict[str, float]:
        value = _clean_for_intent(text)
        if not self.hierarchical:
            return self._scores(self.flat, value)
        group_scores = self._scores(self.level1_model, value)
        candidate_groups = sorted(group_scores, key=group_scores.get, reverse=True)[:2]
        result = {category: -20.0 for category in CAMPUS_CATEGORIES}
        for group in candidate_groups:
            for category, score in self._scores(self.level2_models[group], value).items():
                result[category] = score + 0.35 * group_scores[group]
        return result

    def predict(self, text: str) -> tuple[str, float, dict]:
        scores = self.predict_scores(text)
        ordered = sorted(scores, key=scores.get, reverse=True)
        confidence, _ = _softmax_confidence([scores[category] for category in ordered])
        return ordered[0], confidence, scores


class CampusRouterV2:
    name = "hierarchical-hybrid-v2"

    def __init__(self, examples: list[dict]):
        self.rules = CampusRuleRouter()
        self.flat = CampusSklearnRouter(examples, "char_svm", hierarchical=False)
        self.hierarchical = CampusSklearnRouter(examples, "char_svm", hierarchical=True)
        self.tool_available = set(TOOL_AVAILABLE)

    @staticmethod
    def action_for(question: str, primary: str, intents: tuple[str, ...], confidence_band: str) -> str:
        value = normalize(question)
        if confidence_band == "low":
            return "CLARIFY"
        if len(intents) > 1:
            return "TOOL+MODEL" if any(intent in TOOL_AVAILABLE for intent in intents) else "RAG+MODEL"
        if primary == "university_policy":
            return "RAG"
        if primary == "general":
            return "MODEL"
        if primary in ("programming", "math", "statistics", "ai_usage"):
            return "RAG+MODEL"
        if primary in TOOL_AVAILABLE and any(word in value for word in ACTION_WORDS):
            return "TOOL"
        if primary == "grade_simulator":
            return "TOOL"
        if any(word in value for word in DEFINITION_WORDS):
            return "FAQ"
        return "FAQ"

    @staticmethod
    def _is_ambiguous(text: str) -> bool:
        value = normalize(text)
        return len(value) <= 16 and any(phrase in value for phrase in AMBIGUOUS_ONLY)

    def _combined_scores(self, text: str) -> tuple[dict[str, float], dict]:
        value = _clean_for_intent(text)
        flat = self.flat.predict_scores(value)
        hierarchical = self.hierarchical.predict_scores(value)
        rule_category, rule_confidence, rule_scores = self.rules.predict(value)
        scores = {}
        for category in CAMPUS_CATEGORIES:
            scores[category] = 0.52 * flat.get(category, -20) + 0.48 * hierarchical.get(category, -20)
        if rule_confidence >= 0.78 and rule_scores.get(rule_category, 0) >= 16:
            scores[rule_category] += 2.2
        return scores, {"flat": flat, "hierarchical": hierarchical, "rule_category": rule_category,
                        "rule_confidence": rule_confidence}

    def decide(self, text: str) -> CampusRouteDecision:
        started = perf_counter()
        scores, evidence = self._combined_scores(text)
        ordered = sorted(scores, key=scores.get, reverse=True)
        confidence, gap = _softmax_confidence([scores[category] for category in ordered])
        primary = ordered[0]

        segments = [segment.strip() for segment in COMPOUND_SPLIT.split(text) if len(segment.strip()) >= 2]
        intents: list[str] = [primary]
        if len(segments) >= 2:
            segment_predictions = []
            for segment in segments[:4]:
                segment_scores, _ = self._combined_scores(segment)
                category = max(segment_scores, key=segment_scores.get)
                segment_predictions.append(category)
            intents = list(dict.fromkeys(segment_predictions))
            if intents:
                primary = intents[0]

        if self._is_ambiguous(text):
            primary, intents, confidence, gap = "general", ["general"], 0.24, 0.0
        confidence = min(0.99, max(0.05, confidence))
        if confidence >= 0.78 and gap >= 0.8:
            band = "high"
        elif confidence >= 0.58:
            band = "medium"
        else:
            band = "low"
        if self._is_ambiguous(text):
            band = "low"
        action = self.action_for(text, primary, tuple(intents), band)
        clarify = None
        if action == "CLARIFY":
            clarify = "何について困っていますか？ 例：試験、課題、単位、履修、教授への連絡"
        top2 = tuple(dict.fromkeys([primary] + ordered))[:2]
        return CampusRouteDecision(
            primary=primary, intents=tuple(intents), level1=CATEGORY_TO_LEVEL1[primary], top2=top2,
            confidence=round(confidence, 6), confidence_band=band, action=action,
            clarify_question=clarify, latency_ms=(perf_counter() - started) * 1000,
            evidence={**evidence, "score_gap": gap, "segments": segments},
        )

    def predict(self, text: str) -> tuple[str, float, dict]:
        decision = self.decide(text)
        return decision.primary, decision.confidence, decision.to_dict()


def build_comparison_routers(examples: list[dict]) -> dict[str, Any]:
    return {
        "rules": CampusRuleRouter(),
        "bm25": CampusBM25Router(examples),
        "tfidf_char_centroid": CampusTfidfRouter(examples),
        "char_ngram_svm": CampusSklearnRouter(examples, "char_svm"),
        "word_ngram_svm": CampusSklearnRouter(examples, "word_svm"),
        "logistic_regression": CampusSklearnRouter(examples, "logistic"),
        "linear_svm": CampusSklearnRouter(examples, "char_svm"),
        "naive_bayes": CampusSklearnRouter(examples, "naive_bayes"),
        "hierarchical_svm": CampusSklearnRouter(examples, "char_svm", hierarchical=True),
        "hierarchical_hybrid": CampusRouterV2(examples),
    }
