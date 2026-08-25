from __future__ import annotations

import re

from pipeline.campus_tools_v2 import CampusToolEngineV2
from pipeline.campus_tools import ToolResult, card, number


class CampusToolEngineV23(CampusToolEngineV2):
    """Campus v2.3-only tool safety fixes; the frozen v2.1 tool remains unchanged."""

    @staticmethod
    def _prompt_number(prompt: str, pattern: str) -> float | None:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        return number(match.group(1)) if match else None

    def _toeic_plan(self, prompt: str, state: dict, inputs: dict) -> ToolResult:
        days = number(inputs.get("days"))
        hours = number(inputs.get("hours_per_day"))
        if days is None:
            days = self._prompt_number(prompt, r"(\d+(?:\.\d+)?)\s*日(?:後|間|で)")
        if hours is None:
            hours = self._prompt_number(prompt, r"(?:毎日|1日)に?\s*(\d+(?:\.\d+)?)\s*時間")

        current_score = number(inputs.get("current_score"))
        target_score = number(inputs.get("target_score"))
        focus = str(inputs.get("focus") or "").strip()
        if not focus:
            focus = next((label for cue, label in (
                ("リスニング", "リスニング"), ("読解", "読解"),
                ("文法", "文法"), ("語彙", "語彙"), ("単語", "語彙"),
            ) if cue in prompt), "")

        missing = [name for name, value in (
            ("days", days), ("hours_per_day", hours),
        ) if value is None or value <= 0]
        if missing:
            text = (
                "TOEIC計画の日数・学習時間・固定配分は、入力がない状態では作りません。\n"
                "まず試験日または残り日数と1日に使える時間を確認してください。"
                "現在スコア、目標スコア、伸ばしたい技能もあれば、実績に合わせて調整できます。\n"
                "入力待ちの間は、直近の模試または練習問題の誤答を、語彙・聞き取り・読解・時間不足に分けて記録してください。"
            )
            fields = [
                {"name": "days", "label": "試験までの残り日数"},
                {"name": "hours_per_day", "label": "1日に使える学習時間"},
                {"name": "current_score", "label": "現在スコア（任意）"},
                {"name": "target_score", "label": "目標スコア（任意）"},
                {"name": "focus", "label": "優先したい技能（任意）"},
            ]
            return ToolResult(
                "toeic_plan", text,
                [card("toeic_plan_inputs", "TOEIC計画の条件確認", "未入力の条件を補ってください。",
                      action_label="条件を入力", fields=fields)],
                False, missing,
            )

        focus_text = f"「{focus}」を優先し" if focus else "直近の誤答が多い技能を優先し"
        score_text = (
            f"現在スコア{current_score:g}から目標{target_score:g}までの差も記録します。"
            if current_score is not None and target_score is not None else
            "現在スコアと目標スコアが分かれば、週ごとの進捗と比較できます。"
        )
        text = (
            f"入力された目安は残り{days:g}日、1日{hours:g}時間です。"
            f"{focus_text}、残りの技能も短い練習で維持します。\n"
            "進め方：最初に時間を測って練習し、誤答を原因別に記録します。"
            "次の学習では誤答の多い原因から直し、週ごとに練習結果を見て時間配分を更新します。"
            f"{score_text}\n"
            "注意：試験日や結果発表日は公式サイトで確認し、確認できない数値は補完しません。"
        )
        data = {
            "days": days, "hours_per_day": hours, "current_score": current_score,
            "target_score": target_score, "focus": focus or None, "fixed_ratio_used": False,
        }
        return ToolResult(
            "toeic_plan", text,
            [card("toeic_plan", "TOEIC学習計画", "入力条件と誤答記録に基づく計画です。",
                  action_label="計画をコピー", copy_text=text, data=data)],
            True, [], data,
        )
