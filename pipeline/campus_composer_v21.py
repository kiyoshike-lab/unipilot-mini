from __future__ import annotations

import re

from pipeline.campus_tools import ToolResult


EMAIL_INTENTS = {"professor_email", "absence_email", "lateness_email", "late_submission_email"}


class CampusAnswerComposerV21:
    @staticmethod
    def response_mode(question: str, *, tool: ToolResult | None = None, multi_intent: bool = False) -> str:
        if tool and tool.intent in EMAIL_INTENTS:
            return "email"
        if tool and tool.intent in ("study_plan", "toeic_plan"):
            return "study_plan"
        if tool or any(token in question for token in ("計算", "整理", "作って", "手順", "計画")):
            return "action"
        if multi_intent or len(question) >= 55:
            return "complex"
        return "simple"

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [item.strip() for item in re.split(r"(?<=[。！？])\s*|\n+", text) if item.strip()]

    def compose_faq(self, question: str, answer: str, *, multi_intent: bool = False) -> str:
        mode = self.response_mode(question, multi_intent=multi_intent)
        if mode != "simple":
            return answer
        sentences = self._sentences(answer)
        conclusion = next((item for item in sentences if item.startswith("結論：")), sentences[0] if sentences else answer)
        action = next((item for item in sentences if item.startswith("今やること：")), None)
        caution = next((item for item in sentences if item.startswith("注意：")), None)
        selected = [conclusion]
        if action:
            # Keep one immediately executable step for short questions.
            first_step = re.split(r"\s+2\.| 2\.", action)[0]
            selected.append(first_step)
        elif len(sentences) > 1:
            selected.append(sentences[1])
        if caution and len("".join(selected)) < 150:
            selected.append(caution)
        return "\n".join(selected[:3])

    def compose_tool(self, question: str, result: ToolResult) -> ToolResult:
        mode = self.response_mode(question, tool=result)
        if mode in ("email", "study_plan") or not result.completed:
            return result
        text = result.text.strip()
        if text.startswith("結論："):
            return result
        sentences = self._sentences(text)
        if not sentences:
            return result
        conclusion = sentences[0]
        details = " ".join(sentences[1:-1]) if len(sentences) > 2 else ""
        last = sentences[-1] if len(sentences) > 1 else "結果を確認し、必要な入力を保存してください。"
        composed = f"結論：{conclusion}\n"
        if details:
            composed += f"具体的結果：{details}\n"
        composed += f"次にやること：{last}"
        return ToolResult(result.intent, composed, result.cards, result.completed,
                          result.missing_fields, result.calculation)

    @staticmethod
    def safe_no_match() -> str:
        return ("結論：一致度の高いFAQを確認できなかったため、似たFAQを推測で返しません。\n"
                "今やること：1. まず質問の対象を入力する 2. 期限を確認する 3. 必要な結果を一つ書く。\n"
                "確認質問：対象と期限は何ですか？ 大学固有の制度は公式案内で確認します。")
