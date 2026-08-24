from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


AXES = ("correctness", "relevance", "actionable", "naturalness", "completeness",
        "specificity", "grounding", "conciseness", "helpfulness")
CAUSES = ("TOO_SHORT", "TOO_GENERIC", "NOT_ACTIONABLE", "MISSING_DETAIL", "PARTIAL_ANSWER",
          "WRONG_PRIORITY", "UNNATURAL", "WEAK_GROUNDING", "UNCLEAR", "ROUTER_ISSUE",
          "RETRIEVAL_ISSUE", "TOOL_ISSUE", "MODEL_ISSUE", "OTHER")

CATEGORY_CUES: dict[str, tuple[str, ...]] = {
    "gpa": ("GPA", "成績", "単位", "GP"),
    "grade_simulator": ("点", "得点", "合格", "評価"),
    "professor_email": ("先生", "教授", "メール", "面談", "件名"),
    "absence_email": ("欠席", "休", "メール", "先生"),
    "lateness_email": ("遅刻", "到着", "メール", "先生"),
    "study_plan": ("勉強", "学習", "科目", "試験", "時間"),
    "assignment_priority": ("課題", "締切", "優先", "所要時間"),
    "deadline_organizer": ("締切", "期限", "提出", "一覧"),
    "university_policy": ("公式", "学生便覧", "履修要項", "教務", "制度", "公欠"),
    "exam": ("試験", "過去問", "範囲", "問題", "持込"),
    "assignment": ("課題", "提出", "締切", "評価基準"),
    "credit": ("単位", "必修", "卒業", "認定", "教務"),
    "attendance": ("出席", "欠席", "回数", "シラバス", "担当教員"),
    "lateness": ("遅刻", "到着", "授業"),
    "registration": ("履修", "登録", "科目", "時間割", "上限"),
    "report_outline": ("レポート", "序論", "本論", "結論", "構成"),
    "citation_check": ("引用", "出典", "一次資料", "参考文献"),
    "presentation_outline": ("発表", "プレゼン", "スライド", "構成"),
    "toeic_plan": ("TOEIC", "単語", "読解", "リスニング", "学習"),
    "career_schedule": ("就活", "面接", "企業", "ES", "応募"),
    "internship": ("インターン", "募集", "応募", "面接"),
    "scholarship": ("奨学金", "給付", "貸与", "募集", "返還"),
    "part_time_job": ("バイト", "アルバイト", "勤務", "シフト"),
    "ai_usage": ("AI", "生成AI", "個人情報", "課題", "引用"),
    "campus_life": ("大学生活", "研究室", "学生", "相談", "学内"),
    "relationship": ("同級生", "人間関係", "相談"),
    "programming": ("コード", "エラー", "デバッグ", "再現"),
    "statistics": ("統計", "確率", "分布", "平均"),
    "general": ("確認", "困", "相談", "期限", "何について"),
}

CRITIQUE_TEXT = {
    "TOO_SHORT": "必要な説明量に届かず、理由や次の行動が不足しています。",
    "TOO_GENERIC": "質問固有の情報が少なく、他の質問にも当てはまる回答です。",
    "NOT_ACTIONABLE": "読後に何をすればよいかが十分明確ではありません。",
    "MISSING_DETAIL": "判断や実行に必要な確認項目・具体例が不足しています。",
    "PARTIAL_ANSWER": "複数の依頼の一部しか扱えていません。",
    "WRONG_PRIORITY": "否定された話題または優先度の低い話題に回答しています。",
    "UNNATURAL": "日本語または構成に機械的で読みにくい箇所があります。",
    "WEAK_GROUNDING": "数字・制度・事実を支える根拠または確認先が弱いです。",
    "UNCLEAR": "確認質問や説明の対象が曖昧です。",
    "ROUTER_ISSUE": "質問意図とrouteの組み合わせに不整合があります。",
    "RETRIEVAL_ISSUE": "必要なFAQ/RAG情報を取得できていない可能性があります。",
    "TOOL_ISSUE": "Tool選択またはTool結果の使い方に問題があります。",
    "MODEL_ISSUE": "Model由来部分の具体性または自然さが不足しています。",
    "OTHER": "他の分類に収まらない品質上の問題があります。",
}


def _clamp(value: float) -> float:
    return round(max(0.0, min(5.0, value)), 2)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[。！？])\s*|\n+", text) if part.strip()]


def _normalise(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.lower())


def _char_ngrams(text: str, size: int = 2) -> set[str]:
    value = _normalise(text)
    return {value[index:index + size] for index in range(max(0, len(value) - size + 1))}


@dataclass(frozen=True)
class RedundancyResult:
    rate: float
    duplicate_pairs: list[dict[str, Any]]


class RedundancyDetector:
    def analyse(self, text: str) -> RedundancyResult:
        sentences = [sentence for sentence in _sentences(text) if len(_normalise(sentence)) >= 8]
        pairs: list[dict[str, Any]] = []
        for left_index, left in enumerate(sentences):
            left_grams = _char_ngrams(left)
            for right_index in range(left_index + 1, len(sentences)):
                right = sentences[right_index]
                right_grams = _char_ngrams(right)
                union = left_grams | right_grams
                similarity = len(left_grams & right_grams) / len(union) if union else 0.0
                shorter, longer = sorted((_normalise(left), _normalise(right)), key=len)
                containment = len(shorter) / len(longer) if shorter and shorter in longer else 0.0
                if similarity >= .72 or containment >= .85:
                    pairs.append({"left": left_index, "right": right_index,
                                  "similarity": round(max(similarity, containment), 3)})
        denominator = max(len(sentences) - 1, 1)
        return RedundancyResult(round(min(1.0, len(pairs) / denominator), 3), pairs)


class UnsupportedClaimDetector:
    SAFE_BOUNDARIES = ("断定しません", "断定できません", "とは限りません", "確認してください", "確認します",
                       "断定せず", "異なる", "可能性", "推測で", "分からない", "例：", "目安")
    POLICY_TERMS = ("必ず", "一律", "認められます", "不可です", "免除されます", "公欠になります",
                    "取得できます", "申請できます")

    def analyse(self, question: str, answer: str, sources: list[str]) -> dict[str, Any]:
        support = "\n".join([question, *sources]).replace("パーセント", "%")
        candidates: list[str] = []
        unsupported: list[str] = []
        for sentence in _sentences(answer):
            if any(boundary in sentence for boundary in self.SAFE_BOUNDARIES):
                continue
            numbers = re.findall(r"\d+(?:\.\d+)?\s*(?:%|点|回|単位|年)", sentence)
            policy_claim = any(term in sentence for term in self.POLICY_TERMS)
            if not numbers and not policy_claim:
                continue
            candidates.append(sentence)
            unsupported_number = any(number.strip() not in support for number in numbers)
            if unsupported_number or (policy_claim and not sources):
                unsupported.append(sentence)
        rate = len(unsupported) / len(candidates) if candidates else 0.0
        return {"claim_candidates": candidates, "unsupported_claims": unsupported,
                "unsupported_claim_rate": round(rate, 4)}


class CampusAIJudge:
    """A deterministic local judge. It never calls a generative model or external API."""

    def __init__(self, rubric_path: str | Path = "quality/campus_answer_rubric.json"):
        self.rubric = json.loads(Path(rubric_path).read_text(encoding="utf-8"))
        calibration_path = Path("quality/campus_ai_judge_calibration.json")
        self.calibration = (json.loads(calibration_path.read_text(encoding="utf-8"))
                            if calibration_path.exists() else {})
        self.redundancy = RedundancyDetector()
        self.unsupported = UnsupportedClaimDetector()

    def calibrated_label(self, question: str, answer: str, raw_label: str,
                         issues: list[str], checks: dict[str, Any]) -> str:
        """Calibrate coarse labels from generic features; never use question IDs or answer keys."""
        config = self.calibration
        issue_set = set(issues)
        compound = bool(checks.get("compound_question"))
        if raw_label == "good" and compound and config.get("compound_good_to_close", True):
            return "close"
        severe_bad = (
            {"TOO_GENERIC", "PARTIAL_ANSWER"} <= issue_set
            or {"TOO_GENERIC", "UNCLEAR"} <= issue_set
            or bool(checks.get("unsupported_claims"))
        )
        if raw_label == "bad" and not severe_bad and config.get("recover_non_severe_bad", True):
            return "close"
        only_short = issue_set <= {"TOO_SHORT", "MISSING_DETAIL"}
        grounded = bool(checks.get("source_ids"))
        minimum = int(config.get("grounded_short_good_min_chars", 160))
        if raw_label == "close" and only_short and grounded and len(answer) >= minimum:
            return "good"
        return raw_label

    @staticmethod
    def response_mode(question: str, metadata: dict[str, Any]) -> str:
        planned_depth = metadata.get("answer_depth")
        if planned_depth in ("simple", "normal", "complex"):
            return {"simple": "short", "normal": "normal", "complex": "detailed"}[planned_depth]
        action = str(metadata.get("action", ""))
        cards = metadata.get("cards") or []
        if any(token in question for token in ("詳しく", "詳細", "徹底")):
            return "detailed"
        if re.search(r"(?:って|とは)何[？?]?", question):
            return "simple_definition"
        if any(token in question for token in ("短め", "ざっくり", "簡潔", "一言")):
            return "short"
        if "それと" in question or "両方" in question:
            return "normal"
        if action == "CLARIFY" or ("TOOL" in action and any(card.get("fields") for card in cards)):
            return "short"
        return "normal"

    @staticmethod
    def source_ids(metadata: dict[str, Any]) -> list[str]:
        result = []
        for card in metadata.get("cards") or []:
            data = card.get("data") or {}
            for key in ("faq_id", "source_id", "source"):
                value = data.get(key)
                if isinstance(value, str) and value:
                    result.append(value)
        for key in ("source", "sources"):
            value = metadata.get(key)
            if isinstance(value, str):
                result.append(value)
            elif isinstance(value, list):
                result.extend(str(item) for item in value if item)
        return list(dict.fromkeys(result))

    @staticmethod
    def _category_aligned(category: str, answer: str) -> bool:
        cues = CATEGORY_CUES.get(category, ())
        return not cues or any(cue.lower() in answer.lower() for cue in cues)

    def evaluate(self, question: str, answer: str, metadata: dict[str, Any] | None = None,
                 source_texts: list[str] | None = None) -> dict[str, Any]:
        metadata = metadata or {}
        sources = source_texts or []
        category = str(metadata.get("category") or "general")
        predicted_category = str(metadata.get("predicted_category") or category)
        action = str(metadata.get("action") or "")
        route = str(metadata.get("route") or "")
        cards = metadata.get("cards") or []
        scores = {"correctness": 4.65, "relevance": 4.70, "actionable": 4.55,
                  "naturalness": 4.65, "completeness": 4.55, "specificity": 4.45,
                  "grounding": 4.55, "conciseness": 4.65, "helpfulness": 4.55}
        issues: list[str] = []
        checks: dict[str, Any] = {}
        length = len(answer.strip())
        mode = self.response_mode(question, metadata)
        guidance = self.rubric["length_guidance"][mode]
        min_length, max_length = guidance["min_chars"], guidance["max_chars"]
        tool_waiting_for_input = "TOOL" in action and any(card.get("fields") for card in cards) and (
            "教えてください" in answer or "入力してください" in answer or "必要です" in answer)
        if action == "CLARIFY":
            scores["completeness"] -= .10

        direct = bool(re.match(r"^(?:結論[:：]|そのまま|まず|大学名|何について|目標|必要|具体的)", answer.strip()))
        checks["direct_answer_or_conclusion"] = direct
        if not direct:
            scores["relevance"] -= .20
            scores["naturalness"] -= .10

        too_short = length < min_length and not tool_waiting_for_input and action != "CLARIFY"
        too_long = length > max_length
        checks.update({"response_mode": mode, "character_count": length,
                       "recommended_min_chars": min_length, "recommended_max_chars": max_length,
                       "too_short": too_short, "too_long": too_long})
        if too_short:
            issues.extend(("TOO_SHORT", "MISSING_DETAIL"))
            scores["completeness"] -= .65
            scores["specificity"] -= .35
            scores["helpfulness"] -= .20
            if length < min_length * .4:
                scores["completeness"] -= .35
                scores["specificity"] -= .25
                scores["actionable"] -= .15
        if too_long:
            scores["conciseness"] -= min(1.5, (length - max_length) / max_length * 2)

        step_count = len(re.findall(r"(?:^|[\s　])\d+[.．]", answer)) + len(re.findall(r"(?:^|\n)[・●-]", answer))
        has_action = any(token in answer for token in ("今やること", "次に", "まず", "確認してください", "入力してください",
                                                          "教えてください", "送信", "問い合わせ"))
        checks.update({"action_present": has_action, "step_count": step_count})
        if not has_action:
            issues.append("NOT_ACTIONABLE")
            scores["actionable"] -= 1.20
            scores["helpfulness"] -= .45
        elif step_count >= 3:
            scores["actionable"] += .20
            scores["specificity"] += .20
            scores["helpfulness"] += .15

        generic_fallback = "一致度の高いFAQを確認できなかった" in answer or "まず質問の対象を入力" in answer
        checks["generic_fallback"] = generic_fallback
        if generic_fallback:
            issues.extend(("TOO_GENERIC", "UNCLEAR", "RETRIEVAL_ISSUE"))
            scores["relevance"] -= .80
            scores["actionable"] -= .40
            scores["completeness"] -= .60
            scores["specificity"] -= 1.00
            scores["helpfulness"] -= .60

        category_aligned = self._category_aligned(category, answer)
        checks["category_aligned"] = category_aligned
        if not category_aligned:
            scores["relevance"] -= .45
            scores["specificity"] -= .25

        count_intent_negated = any(token in question for token in ("じゃなくて", "前者は不要", "前者不要"))
        missing_count_answer = (not count_intent_negated and
            any(token in question for token in ("欠席回数", "何回")) and not any(
            token in answer for token in ("欠席", "出席", "回数", "シラバス", "担当教員")))
        checks["missing_count_answer"] = missing_count_answer
        if missing_count_answer:
            issues.extend(("PARTIAL_ANSWER", "MISSING_DETAIL"))
            scores["correctness"] -= 1.00
            scores["relevance"] -= 1.00
            scores["actionable"] -= .55
            scores["completeness"] -= .80
            scores["specificity"] -= .70
            scores["helpfulness"] -= .80

        policy_boilerplate = "大学名・入学年度が分からないため" in answer and category not in (
            "university_policy", "attendance", "registration", "credit")
        wrong_priority = policy_boilerplate or ("じゃなくて" in question and not category_aligned and not generic_fallback)
        checks["wrong_priority"] = wrong_priority
        if wrong_priority:
            issues.extend(("WRONG_PRIORITY", "ROUTER_ISSUE"))
            scores["correctness"] -= 1.50
            scores["relevance"] -= 2.00
            scores["actionable"] -= 1.00
            scores["completeness"] -= 1.50
            scores["specificity"] -= 1.20
            scores["helpfulness"] -= 1.30
            scores["naturalness"] -= .50

        router_mismatch = bool(predicted_category and category and predicted_category != category)
        checks["predicted_category"] = predicted_category
        checks["router_mismatch"] = router_mismatch
        if router_mismatch:
            issues.append("ROUTER_ISSUE")

        compound = "それと" in question or "両方" in question
        covers_second = any(token in answer for token in ("もう一つ", "それと", "次に、もう一方", "2つ目"))
        partial = compound and not covers_second
        checks.update({"compound_question": compound, "covers_second_intent": covers_second})
        if partial:
            issues.append("PARTIAL_ANSWER")
            scores["relevance"] -= .55
            scores["completeness"] -= 1.15
            scores["helpfulness"] -= .55

        unnatural = "part time job" in answer or answer.count("結論：") >= 3 or "具体的結果：注意：" in answer
        checks["natural_japanese"] = not unnatural
        if unnatural:
            issues.append("UNNATURAL")
            scores["naturalness"] -= .75
            scores["helpfulness"] -= .15

        redundancy = self.redundancy.analyse(answer)
        checks["redundancy_rate"] = redundancy.rate
        if redundancy.rate:
            scores["conciseness"] -= min(1.6, redundancy.rate * 2)
            if redundancy.rate >= .34:
                issues.append("UNNATURAL")

        source_ids = self.source_ids(metadata)
        claim_result = self.unsupported.analyse(question, answer, sources)
        checks.update({"source_ids": source_ids, **claim_result})
        if claim_result["unsupported_claims"]:
            issues.append("WEAK_GROUNDING")
            penalty = min(2.0, .80 + claim_result["unsupported_claim_rate"])
            scores["correctness"] -= penalty * .60
            scores["grounding"] -= penalty
        elif source_ids or sources:
            scores["grounding"] += .25
        elif category in ("university_policy", "attendance", "registration", "credit") and not any(
                token in answer for token in ("公式", "シラバス", "履修要項", "教務", "担当教員")):
            issues.append("WEAK_GROUNDING")
            scores["grounding"] -= .90

        if "TOOL" in action and route != "tool" and not tool_waiting_for_input:
            issues.append("TOOL_ISSUE")
            scores["actionable"] -= .50
        if "MODEL" in action and (generic_fallback or unnatural):
            issues.append("MODEL_ISSUE")
        if generic_fallback and route in ("faq", "safe"):
            scores["grounding"] -= .20

        unique_issues = [issue for issue in CAUSES if issue in issues]
        final_scores = {axis: _clamp(scores[axis]) for axis in AXES}
        final_scores["helpfulness"] = _clamp((final_scores["relevance"] + final_scores["actionable"] +
                                               final_scores["completeness"] + scores["helpfulness"]) / 4)
        overall = round(sum(final_scores.values()) / (len(AXES) * 5) * 100, 2)
        label = "good" if overall >= 90 else "close" if overall >= 70 else "bad"
        calibrated = self.calibrated_label(question, answer, label, unique_issues, checks)
        return {"scores_0_to_5": final_scores, "overall_score": overall, "quality_label": label,
                "calibrated_quality_label": calibrated,
                "calibration_version": self.calibration.get("version", "none"),
                "issues": unique_issues, "primary_issue": unique_issues[0] if unique_issues else None,
                "critique": [CRITIQUE_TEXT[issue] for issue in unique_issues], "checks": checks,
                "redundancy": {"rate": redundancy.rate, "duplicate_pairs": redundancy.duplicate_pairs},
                "unsupported_claim_rate": claim_result["unsupported_claim_rate"],
                "unsupported_claims": claim_result["unsupported_claims"],
                "hallucination_suspected": bool(claim_result["unsupported_claims"]),
                "judge_type": "deterministic_local", "external_ai_api": "OFF"}
