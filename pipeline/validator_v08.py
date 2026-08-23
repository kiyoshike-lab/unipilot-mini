from __future__ import annotations

from dataclasses import asdict, dataclass
import re


BROKEN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f�]")
POLICY = re.compile(r"どの大学でも|全国の大学で|必ず(?:認め|合格|無料|返還不要|単位)|全員(?:が)?対象")
UNSUPPORTED_NUMBER = re.compile(r"(?:20\d{2}年\d{1,2}月\d{1,2}日|(?:料金|費用|受験料)は?\d{3,}円)")
SUBJECTS = ("法学", "経済学", "心理学", "日本史", "物理学", "化学", "英文学")


@dataclass
class StandardValidation:
    valid: bool
    score: float
    issues: list[str]
    context_overlap: float

    def to_dict(self) -> dict:
        return asdict(self)


class StandardAnswerValidator:
    def validate(self, question: str, answer: str, context: str = "") -> StandardValidation:
        issues = []
        value = answer.strip()
        if not value:
            issues.append("empty")
        if BROKEN.search(answer):
            issues.append("broken_text")
        extras = [subject for subject in SUBJECTS if subject in answer and subject not in question and subject not in context]
        if extras:
            issues.append("invented_subject:" + ",".join(extras))
        if POLICY.search(answer):
            issues.append("university_specific_hallucination")
        if UNSUPPORTED_NUMBER.search(answer) and not any(match in context for match in UNSUPPORTED_NUMBER.findall(answer)):
            issues.append("unsupported_date_or_fee")
        if len(value) < 12:
            issues.append("incomplete")
        answer_terms = set(re.findall(r"[ぁ-んァ-ヶー一-龥々]{2,}|[A-Za-z0-9]+", answer.lower()))
        context_terms = set(re.findall(r"[ぁ-んァ-ヶー一-龥々]{2,}|[A-Za-z0-9]+", context.lower()))
        overlap = len(answer_terms & context_terms) / max(1, len(answer_terms)) if context else 0.0
        score = max(0.0, min(1.0, 1.0 - 0.22 * len(issues) + min(0.15, overlap)))
        return StandardValidation(not issues, score, issues, overlap)

    @staticmethod
    def fallback(category: str) -> str:
        return "確認できる情報だけでは断定できません。現在年度の大学公式案内を確認し、必要なら担当教員または該当窓口へ相談してください。"
