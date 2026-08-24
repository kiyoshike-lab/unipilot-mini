from __future__ import annotations

import json
from pathlib import Path
import re
from time import perf_counter

from pipeline.campus_categories import CAMPUS_KEYWORDS
from pipeline.campus_categories_v2 import CATEGORY_TO_LEVEL1, TOOL_AVAILABLE
from pipeline.campus_router import normalize
from pipeline.campus_router_v2 import (COMPOUND_SPLIT, CampusRouteDecision, CampusRouterV2,
                                       _softmax_confidence)


VAGUE_MARKERS = (
    "やばい", "どうしよう", "間に合わない", "大学むり", "大学無理", "これ落とす", "先生どうしよう",
    "もうだめ", "詰んだ", "何か忘れた", "これで平気", "助けて", "不安すぎる", "今日むり",
    "怒られそう", "終わった", "どう連絡する", "これ必要", "無理そう", "判断できない", "いま何する",
    "大学のことで困", "どうしたらいい", "間に合うかな", "相談したい", "何からすればいい",
    "もう無理", "これ大丈夫", "助けてほしい",
)

EXTRA_SIGNALS = {
    "email": ("メール", "メル", "文面", "件名", "なんて送", "連絡文"),
    "exam": ("テスト", "試験", "追試", "過去問", "受験"),
    "credit": ("単位", "落単", "卒業要件"),
    "gpa": ("gpa", "gp", "成績平均"),
    "assignment": ("課題", "宿題", "提出"),
    "attendance": ("欠席", "出席", "公欠"),
    "lateness": ("遅刻", "遅れる", "遅延"),
    "registration": ("履修", "時間割", "必修"),
    "study": ("勉強", "学習", "復習"),
    "money": ("学費", "授業料", "奨学金"),
}

NEGATION_CONTRAST = re.compile(
    r"^(?P<negative>.+?)(?P<connector>ではなく|じゃなくて|じゃなく|じゃない[。、]?(?:聞きたいのは)?|"
    r"の相談ではない[。、]?代わりに|は不要で|より[、]?知りたいのは|の話は不要で|"
    r"と比べたいのではなく)(?P<positive>.+)$"
)
CORRECTION = re.compile(r"^(?P<negative>.+?)(?P<connector>[。、](?:違う|訂正すると|正しくは)[、。]?)(?P<positive>.+)$")
FOCUS_SHIFT = re.compile(
    r"^(?P<negative>.+?)(?P<connector>ではありません[。]?対象は|より優先したいのは|"
    r"分類対象は|求める処理は|は違う[。]?)(?P<positive>.+)$"
)


def parse_negation_contrast(text: str) -> dict | None:
    value = text.strip()
    match = NEGATION_CONTRAST.match(value) or CORRECTION.match(value) or FOCUS_SHIFT.match(value)
    if not match:
        return None
    negative = match.group("negative").strip(" 、。")
    positive = match.group("positive").strip(" 、。")
    if len(negative) < 2 or len(positive) < 2:
        return None
    connector = match.group("connector")
    kind = "CORRECTION" if any(token in connector for token in ("違う", "訂正", "正しく")) else (
        "CONTRAST" if any(token in connector for token in ("より", "比べ", "不要")) else "NEGATION")
    return {"negative_text": negative, "positive_text": positive, "connector": connector, "kind": kind}


def signal_count(text: str) -> int:
    value = normalize(text)
    for marker in VAGUE_MARKERS:
        value = value.replace(normalize(marker), "")
    signals = set()
    for category, keywords in CAMPUS_KEYWORDS.items():
        # ``予定`` alone is often a temporal modifier (for example ``予定外で``),
        # not a concrete schedule object. It must not disable clarification.
        if category in ("general", "schedule"):
            continue
        if any(keyword in value for keyword in keywords):
            signals.add(category)
    for name, aliases in EXTRA_SIGNALS.items():
        if any(alias in value for alias in aliases):
            signals.add(name)
    return len(signals)


def should_clarify(features: dict, config: dict) -> bool:
    if features["has_negation"] or features["multi_intent"] or features["tool_signal"]:
        return False
    if features["signal_count"] >= config["minimum_signals_to_route"]:
        return False
    vague_path = features["has_vague_marker"] and (
        features["query_length"] <= config["vague_max_length"] or
        features["top1_confidence"] < config["vague_confidence_floor"] or
        features["score_margin"] < config["vague_margin_floor"]
    )
    uncertain_path = (
        features["query_length"] <= config["uncertain_max_length"] and
        features["top1_confidence"] < config["uncertain_confidence_floor"] and
        features["score_margin"] < config["uncertain_margin_floor"]
    )
    return vague_path or uncertain_path


DEFAULT_CLARIFICATION_CONFIG = {
    "minimum_signals_to_route": 1,
    "vague_max_length": 36,
    "vague_confidence_floor": .82,
    "vague_margin_floor": 1.2,
    "uncertain_max_length": 12,
    "uncertain_confidence_floor": .62,
    "uncertain_margin_floor": .5,
    "selection_source": "default-before-validation-search",
}


class CampusRouterV21(CampusRouterV2):
    name = "hierarchical-hybrid-v2.1"

    def __init__(self, examples: list[dict], config_path: str | Path = "data/campus_v21/router/clarification-config.json"):
        super().__init__(examples)
        path = Path(config_path)
        self.clarification_config = (json.loads(path.read_text(encoding="utf-8"))["selected_config"]
                                     if path.exists() else dict(DEFAULT_CLARIFICATION_CONFIG))

    @staticmethod
    def _clarification_question(top2: tuple[str, ...]) -> str:
        labels = {
            "exam": "試験", "assignment": "課題", "credit": "単位", "attendance": "出席・欠席",
            "registration": "履修", "professor_email": "教授への連絡", "schedule": "予定",
            "study_plan": "勉強計画", "general": "その他の相談", "faq_search": "確認先の検索",
        }
        candidates = [labels.get(category, category.replace("_", " ")) for category in top2]
        examples = "・".join(dict.fromkeys([*candidates, "試験", "課題", "単位", "教授への連絡"]))
        return f"何について困っていますか？ {examples}などから、近いものを教えてください。"

    def _features(self, text: str, scores: dict[str, float], base: CampusRouteDecision,
                  parsed: dict | None) -> dict:
        ordered = sorted(scores, key=scores.get, reverse=True)
        confidence, margin = _softmax_confidence([scores[item] for item in ordered])
        compact = normalize(text)
        return {
            "top1": ordered[0], "top2": ordered[1], "top1_score": scores[ordered[0]],
            "top2_score": scores[ordered[1]], "top1_confidence": confidence,
            "top2_confidence": 1.0 - confidence, "score_margin": margin,
            "query_length": len(compact), "signal_count": signal_count(text),
            "has_vague_marker": any(marker in compact for marker in VAGUE_MARKERS),
            "has_negation": parsed is not None, "multi_intent": len(base.intents) > 1,
            "tool_signal": any(intent in TOOL_AVAILABLE for intent in base.intents) and any(
                token in compact for token in ("計算", "作って", "文面", "整理", "配分", "何点", "何単位")),
        }

    def analyze(self, text: str) -> tuple[CampusRouteDecision, dict]:
        started = perf_counter()
        parsed = parse_negation_contrast(text)
        scores, score_evidence = self._combined_scores(text)
        if parsed:
            negative_scores, _ = self._combined_scores(parsed["negative_text"])
            positive_scores, _ = self._combined_scores(parsed["positive_text"])
            negative_intent = max(negative_scores, key=negative_scores.get)
            positive_intent = max(positive_scores, key=positive_scores.get)
            scores[negative_intent] -= 5.0
            scores[positive_intent] += 6.0
            ordered = sorted(scores, key=scores.get, reverse=True)
            confidence, margin = _softmax_confidence([scores[item] for item in ordered])
            band = "high" if confidence >= .78 and margin >= .8 else "medium"
            primary = positive_intent
            top2 = tuple(dict.fromkeys([primary, *ordered]))[:2]
            action = self.action_for(text, primary, (primary,), band)
            base = CampusRouteDecision(primary, (primary,), CATEGORY_TO_LEVEL1[primary], top2,
                                       min(.99, confidence), band, action, None, 0.0,
                                       {**score_evidence, "score_gap": margin})
            parsed.update(negative_intent=negative_intent, positive_intent=positive_intent)
        else:
            # Rebuild the v2 decision from its scores so the legacy ``AMBIGUOUS_ONLY``
            # shortcut cannot mark short-but-clear queries before the v2.1 gate runs.
            ordered = sorted(scores, key=scores.get, reverse=True)
            confidence, margin = _softmax_confidence([scores[item] for item in ordered])
            primary = ordered[0]
            segments = [segment.strip() for segment in COMPOUND_SPLIT.split(text)
                        if len(segment.strip()) >= 2]
            intents = [primary]
            if len(segments) >= 2:
                segment_predictions = []
                for segment in segments[:4]:
                    segment_scores, _ = self._combined_scores(segment)
                    segment_predictions.append(max(segment_scores, key=segment_scores.get))
                intents = list(dict.fromkeys(segment_predictions)) or intents
                primary = intents[0]
            band = "high" if confidence >= .78 and margin >= .8 else (
                "medium" if confidence >= .58 else "low")
            action = self.action_for(text, primary, tuple(intents), band)
            top2 = tuple(dict.fromkeys([primary, *ordered]))[:2]
            base = CampusRouteDecision(
                primary, tuple(intents), CATEGORY_TO_LEVEL1[primary], top2,
                min(.99, max(.05, confidence)), band, action, None, 0.0,
                {**score_evidence, "score_gap": margin, "segments": segments},
            )

        features = self._features(text, scores, base, parsed)
        clarify = should_clarify(features, self.clarification_config)
        primary, intents, action, band = base.primary, base.intents, base.action, base.confidence_band
        if clarify:
            primary, intents, action, band = "general", ("general",), "CLARIFY", "low"
        elif len(intents) > 1:
            # Two separately recognizable intents are not ambiguous even when the full-query margin is small.
            action = "TOOL+MODEL" if any(intent in TOOL_AVAILABLE for intent in intents) else "RAG+MODEL"
            band = "high" if features["signal_count"] else "medium"
        elif action == "CLARIFY":
            # Low classifier confidence alone is insufficient once a concrete campus signal exists.
            band = "medium"
            action = self.action_for(text, primary, intents, band)
        top2 = base.top2
        question = self._clarification_question(top2) if clarify else None
        evidence = {**base.evidence, "scores": {category: scores[category] for category in top2},
                    "ambiguity_features": features, "clarification_config": self.clarification_config,
                    "ambiguous_flag": clarify, "negation_contrast": parsed}
        decision = CampusRouteDecision(
            primary=primary, intents=intents, level1=CATEGORY_TO_LEVEL1[primary], top2=top2,
            confidence=.2 if clarify else base.confidence, confidence_band=band, action=action,
            clarify_question=question, latency_ms=(perf_counter() - started) * 1000, evidence=evidence,
        )
        return decision, features

    def decide(self, text: str) -> CampusRouteDecision:
        return self.analyze(text)[0]
