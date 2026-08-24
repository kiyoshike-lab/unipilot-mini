from __future__ import annotations

from datetime import date
import re

from pipeline.campus_tools import CampusToolEngine, ToolResult, card, number


class CampusToolEngineV2(CampusToolEngine):
    """Campus v1's 16 tools plus deterministic, university-neutral calculators."""

    added_tools = ("gpa_target", "credit_progress", "exam_countdown", "report_allocation",
                   "presentation_allocation", "time_allocation")

    def execute(self, intent: str, prompt: str, state: dict, inputs: dict | None = None) -> ToolResult:
        values = inputs or {}
        compact = prompt.lower().replace(" ", "")
        if intent == "gpa" and ("target_gpa" in values or "目標gpa" in compact or "必要なgpa" in compact):
            return self._gpa_target(prompt, state, values)
        if intent == "credit" and (values or any(token in compact for token in ("あと何単位", "進捗", "取得単位"))):
            return self._credit_progress(prompt, state, values)
        if intent == "exam" and ("exam_date" in values or "あと何日" in compact):
            return self._exam_countdown(prompt, state, values)
        if intent == "report_outline" and ("target_characters" in values or "文字配分" in compact or "文字数" in compact):
            return self._report_allocation(prompt, state, values)
        if intent == "presentation_outline" and ("total_minutes" in values or "時間配分" in compact):
            return self._presentation_allocation(prompt, state, values)
        if intent == "schedule" and (values or "時間配分" in compact):
            return self._time_allocation(prompt, state, values)
        return super().execute(intent, prompt, state, values)

    def can_handle(self, intent: str) -> bool:
        return intent in {"credit", "exam", "schedule"} or super().can_handle(intent)

    @staticmethod
    def _gpa_target(prompt: str, state: dict, inputs: dict) -> ToolResult:
        current = number(inputs.get("current_gpa") or state.get("gpa"))
        current_credits = number(inputs.get("current_credits") or state.get("earned_credits"))
        target = number(inputs.get("target_gpa") or state.get("target_gpa"))
        future_credits = number(inputs.get("future_credits"))
        missing = [key for key, value in (("current_gpa", current), ("current_credits", current_credits),
                                          ("target_gpa", target), ("future_credits", future_credits)) if value is None]
        if missing:
            fields = [{"name": key, "label": label} for key, label in (
                ("current_gpa", "現在の累積GPA"), ("current_credits", "現在の取得単位"),
                ("target_gpa", "目標GPA"), ("future_credits", "今後取得する単位"))]
            text = "目標GPAの計算には、現在GPA、現在の取得単位、目標GPA、今後の単位数が必要です。大学のGP上限も確認してください。"
            return ToolResult("gpa_target", text, [card("gpa_target", "目標GPAを計算", "4項目を入力してください。",
                                                                action_label="必要GPAを計算", fields=fields)], False, missing)
        required = (target * (current_credits + future_credits) - current * current_credits) / future_credits if future_credits > 0 else float("inf")
        feasible = 0 <= required <= 4
        conclusion = f"今後{future_credits:g}単位で平均GPA {required:.2f} が必要です。" if feasible else f"必要平均GPAは {required:.2f} で、0〜4の仮定では到達困難です。"
        text = f"計算結果：{conclusion}\n式：目標GPA×合計単位から現在のGP総量を引き、今後の単位数で割りました。大学のGPA規程が異なる場合はその上限で再計算してください。"
        data = {"required_future_gpa": round(required, 4), "feasible_on_4_scale": feasible,
                "current_gpa": current, "current_credits": current_credits, "target_gpa": target,
                "future_credits": future_credits}
        return ToolResult("gpa_target", text, [card("gpa_target_result", "目標GPA計算", conclusion,
                                                            action_label="結果をコピー", copy_text=text, data=data)], True, [], data)

    @staticmethod
    def _credit_progress(prompt: str, state: dict, inputs: dict) -> ToolResult:
        earned = number(inputs.get("earned_credits") or state.get("earned_credits"))
        required = number(inputs.get("required_credits") or state.get("required_credits"))
        missing = [key for key, value in (("earned_credits", earned), ("required_credits", required)) if value is None]
        if missing:
            text = "単位進捗を計算するには、取得済み単位と、入学年度・学部の履修要項に書かれた必要単位を入力してください。必要単位を大学名だけから推測しません。"
            fields = [{"name": "earned_credits", "label": "取得済み単位"},
                      {"name": "required_credits", "label": "公式要項の必要単位"}]
            return ToolResult("credit_progress", text, [card("credit_progress", "単位進捗", "公式の必要単位を入力してください。",
                                                                      action_label="進捗を計算", fields=fields)], False, missing)
        remaining = max(0.0, required - earned)
        rate = earned / required * 100 if required > 0 else 0.0
        text = f"計算結果：{earned:g}/{required:g}単位（{rate:.1f}%）、残り{remaining:g}単位です。\n必修・選択区分は合計とは別に履修要項で照合してください。"
        data = {"earned_credits": earned, "required_credits": required, "remaining_credits": remaining,
                "progress_percent": round(rate, 2)}
        return ToolResult("credit_progress", text, [card("credit_progress_result", "単位進捗", f"残り{remaining:g}単位",
                                                                 action_label="結果をコピー", copy_text=text, data=data)], True, [], data)

    @staticmethod
    def _exam_countdown(prompt: str, state: dict, inputs: dict) -> ToolResult:
        exam_date = str(inputs.get("exam_date") or state.get("exam_date") or "")
        current_date = str(inputs.get("current_date") or date.today().isoformat())
        if not exam_date:
            return ToolResult("exam_countdown", "試験日をYYYY-MM-DDで入力してください。大学の試験案内に記載された日付を使います。",
                              [card("exam_countdown", "試験までの日数", "試験日を入力してください。", action_label="日数を計算",
                                    fields=[{"name": "exam_date", "label": "試験日", "example": "2026-09-10"}])], False, ["exam_date"])
        try:
            remaining = (date.fromisoformat(exam_date) - date.fromisoformat(current_date)).days
        except ValueError:
            return ToolResult("exam_countdown", "日付をYYYY-MM-DD形式で入力してください。", [], False, ["exam_date"])
        text = f"計算結果：試験まで{remaining}日です。\n試験時刻・教室・持込条件は試験案内で別に確認してください。"
        data = {"exam_date": exam_date, "current_date": current_date, "days_remaining": remaining}
        return ToolResult("exam_countdown", text, [card("exam_countdown_result", "試験カウントダウン", f"残り{remaining}日",
                                                                action_label="結果をコピー", copy_text=text, data=data)], True, [], data)

    @staticmethod
    def _report_allocation(prompt: str, state: dict, inputs: dict) -> ToolResult:
        total = number(inputs.get("target_characters"))
        if total is None:
            match = re.search(r"(\d{3,6})\s*字", prompt)
            total = number(match.group(1)) if match else None
        if total is None:
            return ToolResult("report_allocation", "目標文字数を入力してください。", [card("report_allocation", "レポート文字配分",
                              "目標文字数を入力してください。", action_label="文字数を配分",
                              fields=[{"name": "target_characters", "label": "目標文字数"}])], False, ["target_characters"])
        ratios = (("序論", .15), ("本論・根拠", .50), ("考察", .20), ("結論", .15))
        allocation = {name: round(total * ratio) for name, ratio in ratios}
        text = "文字数配分案：" + "、".join(f"{name}{value}字" for name, value in allocation.items()) + "。課題の指定見出しがあれば、指定を優先してください。"
        return ToolResult("report_allocation", text, [card("report_allocation_result", "レポート文字配分", f"合計{total:g}字",
                                                                 action_label="配分をコピー", copy_text=text, data=allocation)], True, [], allocation)

    @staticmethod
    def _presentation_allocation(prompt: str, state: dict, inputs: dict) -> ToolResult:
        total = number(inputs.get("total_minutes"))
        if total is None:
            match = re.search(r"(\d+(?:\.\d+)?)\s*分", prompt)
            total = number(match.group(1)) if match else None
        if total is None:
            return ToolResult("presentation_allocation", "発表の合計時間を分で入力してください。", [card("presentation_allocation",
                              "発表時間配分", "合計時間を入力してください。", action_label="時間を配分",
                              fields=[{"name": "total_minutes", "label": "合計時間（分）"}])], False, ["total_minutes"])
        ratios = (("導入", .15), ("本論", .55), ("考察", .15), ("まとめ", .10), ("余白", .05))
        allocation = {name: round(total * ratio, 1) for name, ratio in ratios}
        text = "時間配分案：" + "、".join(f"{name}{value:g}分" for name, value in allocation.items()) + "。質疑時間が別枠かを確認してください。"
        return ToolResult("presentation_allocation", text, [card("presentation_allocation_result", "発表時間配分", f"合計{total:g}分",
                                                                       action_label="配分をコピー", copy_text=text, data=allocation)], True, [], allocation)

    @staticmethod
    def _time_allocation(prompt: str, state: dict, inputs: dict) -> ToolResult:
        total = number(inputs.get("available_hours") or state.get("available_hours"))
        fixed = number(inputs.get("fixed_hours")) or 0.0
        if total is None:
            return ToolResult("time_allocation", "使える合計時間と、授業・バイトなど固定時間を入力してください。",
                              [card("time_allocation", "時間配分", "合計時間と固定時間を入力してください。", action_label="時間を配分",
                                    fields=[{"name": "available_hours", "label": "合計時間"}, {"name": "fixed_hours", "label": "固定時間"}])],
                              False, ["available_hours"])
        flexible = max(0.0, total - fixed)
        allocation = {"fixed_hours": fixed, "study_hours": round(flexible * .6, 2),
                      "tasks_hours": round(flexible * .25, 2), "buffer_hours": round(flexible * .15, 2)}
        text = f"時間配分案：固定{fixed:g}時間、学習{allocation['study_hours']:g}時間、課題{allocation['tasks_hours']:g}時間、予備{allocation['buffer_hours']:g}時間です。締切が近い日は課題枠を増やしてください。"
        return ToolResult("time_allocation", text, [card("time_allocation_result", "時間配分", f"使える時間{total:g}時間",
                                                               action_label="配分をコピー", copy_text=text, data=allocation)], True, [], allocation)

