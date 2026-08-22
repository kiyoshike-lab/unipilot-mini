from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from pipeline.categories import CATEGORY_KEYWORDS, SAFE_FALLBACKS


SUBJECTS = ("経済学", "法学", "心理学", "情報科学", "日本史", "統計学", "英語", "数学", "物理学")
POLICY_ASSERTION = re.compile(r"全国(?:の)?大学で|どの大学でも|必ず(?:認め|合格|留年|入室|変更|取得)|全員(?:が)?(?:対象|無料|返還不要)")
BROKEN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f�]")


@dataclass
class ValidationResult:
    valid: bool
    score: float
    issues: list[str]
    category_keyword_hits: int
    grounded: bool

    def to_dict(self) -> dict:
        return asdict(self)


class AnswerValidator:
    def validate(self, question: str, answer: str, category: str, grounded_answer: str | None = None) -> ValidationResult:
        issues = []
        stripped = answer.strip()
        grounded = bool(grounded_answer and stripped == grounded_answer.strip())
        if not stripped:
            issues.append("empty")
        if BROKEN.search(answer):
            issues.append("broken_text")
        extra_subjects = [subject for subject in SUBJECTS if subject in answer and subject not in question]
        if extra_subjects:
            issues.append("invented_subject:" + ",".join(extra_subjects))
        if POLICY_ASSERTION.search(answer):
            issues.append("university_specific_hallucination")
        keywords = CATEGORY_KEYWORDS.get(category, ())
        hits = sum(keyword in answer.lower() for keyword in keywords)
        if not grounded and category != "general" and hits == 0:
            issues.append("category_mismatch")
        if len(stripped) < 12:
            issues.append("incomplete")
        if any(phrase in answer for phrase in ("友達を増やしてサークル", "残り1日なら")) and not any(
                phrase in question for phrase in ("友達", "サークル", "残り1日")):
            issues.append("template_mixing")
        score = 1.0 - 0.22 * len(issues) + min(0.2, hits * 0.05) + (0.2 if grounded else 0.0)
        return ValidationResult(valid=not issues, score=max(0.0, min(1.0, score)), issues=issues,
                                category_keyword_hits=hits, grounded=grounded)

    @staticmethod
    def fallback(category: str, grounded_answer: str | None) -> str:
        return grounded_answer or SAFE_FALLBACKS[category]
