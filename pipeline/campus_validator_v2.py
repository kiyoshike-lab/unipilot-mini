from __future__ import annotations

import re

from pipeline.campus_validator import CampusValidation, CampusValidator


POLICY_CLAIM = re.compile(r"(?:この|あなたの|その)大学では.{0,18}(?:必ず|一律|確実に)|欠席\d+回で単位を落とす")
MIX_MARKERS = {
    "email": ("件名：", "先生"), "calculation": ("計算結果", "式："),
    "writing": ("序論", "本論", "結論"), "career": ("ES", "面接", "企業"),
}


class CampusValidatorV2(CampusValidator):
    def validate(self, question: str, answer: str, *, grounded: bool = False, tool_result: bool = False,
                 source_urls: list[str] | None = None, university_known: bool = False,
                 category: str | None = None, action: str | None = None, cards: list[dict] | None = None) -> CampusValidation:
        base = super().validate(question, answer, grounded=grounded, tool_result=tool_result,
                                source_urls=source_urls, university_known=university_known)
        issues = list(base.issues)
        if tool_result and cards and any(item.get("fields") or item.get("action_label") for item in cards):
            issues = [issue for issue in issues if issue != "incomplete"]
        if POLICY_CLAIM.search(answer) and not (grounded and source_urls):
            issues.append("invented_university_policy")
        if action == "CLARIFY" and "？" not in answer and "?" not in answer:
            issues.append("clarification_without_question")
        if not tool_result and re.search(r"(?:計算結果|残り)[:：]?\s*-?\d+(?:\.\d+)?", answer) and not grounded:
            issues.append("unverified_calculation")
        active_templates = [name for name, markers in MIX_MARKERS.items() if all(marker in answer for marker in markers)]
        if len(active_templates) >= 3:
            issues.append("mixed_templates")
        if "［" in answer and "］" in answer and not tool_result:
            issues.append("unfinished_template")
        issues = list(dict.fromkeys(issues))
        action_markers = sum(marker in answer for marker in ("結論：", "今やること：", "まず", "確認", "入力", "1.", "計算結果", "？"))
        if tool_result and cards:
            actionable = 5.0 if any(card.get("action_label") for card in cards) else 4.5
        elif action == "CLARIFY":
            actionable = 4.5 if ("？" in answer or "?" in answer) else 2.5
        elif grounded:
            actionable = min(5.0, 3.5 + action_markers * .35)
        else:
            actionable = min(4.5, 2.5 + action_markers * .3)
        score = max(0.0, min(1.0, 1 - .24 * len(issues) + (.1 if grounded else 0) + (.12 if tool_result else 0)))
        return CampusValidation(not issues, score, issues, actionable, grounded)
