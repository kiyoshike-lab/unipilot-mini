from __future__ import annotations

from dataclasses import asdict, dataclass
import re


UNSAFE_POLICY = re.compile(
    r"(?:どの|すべての|全国の)大学(?:でも|で)|必ず(?:追試|合格|進級|卒業|再提出|免除)|"
    r"欠席(?:は)?\d+回で(?:必ず)?単位|GPA(?:は)?\d(?:\.\d+)?以上なら必ず"
)
UNSUPPORTED_MONEY_DATE = re.compile(r"(?:\d{1,4}円|\d{1,3}万円|20\d{2}年\d{1,2}月\d{1,2}日)")
BROKEN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f�]")


@dataclass
class CampusValidation:
    valid: bool
    score: float
    issues: list[str]
    actionable_score: float
    grounded: bool

    def to_dict(self) -> dict:
        return asdict(self)


class CampusValidator:
    def validate(self, question: str, answer: str, *, grounded: bool = False, tool_result: bool = False,
                 source_urls: list[str] | None = None, university_known: bool = False) -> CampusValidation:
        issues = []
        value = answer.strip()
        if not value:
            issues.append("empty")
        if BROKEN.search(answer):
            issues.append("broken_text")
        if UNSAFE_POLICY.search(answer):
            issues.append("university_specific_assertion")
        if UNSUPPORTED_MONEY_DATE.search(answer) and not (grounded and source_urls):
            issues.append("unsupported_money_or_date")
        if any(phrase in question.lower() for phrase in ("うちの大学", "この大学")) and not university_known:
            if not any(phrase in answer for phrase in ("大学名", "公式", "学生便覧", "履修要項", "シラバス")):
                issues.append("missing_confirmation_path")
        if len(value) < 16:
            issues.append("incomplete")
        action_markers = sum(marker in answer for marker in ("まず", "確認", "入力", "件名", "1.", "①", "次に", "計算結果"))
        if tool_result:
            actionable = 5.0 if "入力してください" not in answer and "教えてください" not in answer else 3.0
        elif grounded:
            actionable = min(4.0, 2.5 + action_markers * 0.5)
        else:
            actionable = min(3.5, 1.5 + action_markers * 0.5)
        score = max(0.0, min(1.0, 1 - 0.25 * len(issues) + (0.1 if grounded else 0) + (0.15 if tool_result else 0)))
        return CampusValidation(not issues, score, issues, actionable, grounded)

    @staticmethod
    def safe_fallback() -> str:
        return "大学ごとに条件が異なる可能性があります。現在年度の学生便覧・履修要項・シラバスを確認し、不明点は担当教員または教務窓口へ確認してください。"
