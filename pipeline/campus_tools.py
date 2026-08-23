from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from pipeline.campus_categories import CAMPUS_LABELS, TOOL_INTENTS


GRADE_POINTS = {"S": 4.0, "A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0, "F": 0.0}


@dataclass
class ToolResult:
    intent: str
    text: str
    cards: list[dict]
    completed: bool
    missing_fields: list[str]
    calculation: dict | None = None


def card(kind: str, title: str, summary: str, *, action_label: str | None = None,
         copy_text: str | None = None, fields: list[dict] | None = None, data: dict | None = None) -> dict:
    return {"kind": kind, "title": title, "summary": summary, "action_label": action_label,
            "copy_text": copy_text, "fields": fields or [], "data": data or {}}


def number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class CampusToolEngine:
    def can_handle(self, intent: str) -> bool:
        return intent in TOOL_INTENTS

    def execute(self, intent: str, prompt: str, state: dict, inputs: dict | None = None) -> ToolResult:
        method = getattr(self, f"_{intent}", None)
        if method is None:
            return ToolResult(intent, "この機能は準備中です。大学生活FAQから確認方法を案内します。", [], False, [])
        return method(prompt, state, inputs or {})

    def _gpa(self, prompt: str, state: dict, inputs: dict) -> ToolResult:
        courses = inputs.get("courses") if isinstance(inputs.get("courses"), list) else []
        parsed = []
        for course in courses:
            credits = number(course.get("credits")) if isinstance(course, dict) else None
            grade = str(course.get("grade", "")).upper() if isinstance(course, dict) else ""
            gp = number(course.get("gp")) if isinstance(course, dict) else None
            if gp is None:
                gp = GRADE_POINTS.get(grade)
            if credits and gp is not None:
                parsed.append({"name": course.get("name", "科目"), "credits": credits, "gp": gp})
        if not parsed:
            for grade, credits in re.findall(r"\b([SABCDF])\s*[:：()（）]?\s*(\d+(?:\.\d+)?)\s*単位", prompt.upper()):
                parsed.append({"name": grade, "credits": float(credits), "gp": GRADE_POINTS[grade]})
        if not parsed:
            text = (
                "GPAを正確に計算するには、各科目の成績またはGPと単位数を入力してください。\n"
                "例：A 2単位、B 2単位、S 1単位\n"
                "大学によってS/A/B/CのGPや不合格科目の扱いが異なるため、計算規程も確認してください。"
            )
            fields = [{"name": "courses", "label": "科目・成績・単位数", "example": "A 2単位、B 2単位"}]
            return ToolResult("gpa", text, [card("gpa_calculator", "GPAを計算する", "成績と単位数を入力してください。",
                                                        action_label="GPAを計算する", fields=fields)], False, ["courses"])
        total_credits = sum(row["credits"] for row in parsed)
        weighted = sum(row["credits"] * row["gp"] for row in parsed)
        gpa = weighted / total_credits
        text = (
            f"計算結果：GPAは {gpa:.2f} です。\n"
            f"計算：GP×単位数の合計 {weighted:.2f} ÷ 対象単位 {total_credits:g}\n"
            "これはS=4、A=3、B=2、C=1、D/F=0の仮定です。所属大学の規程と異なる場合はGPを直接指定してください。"
        )
        data = {"gpa": round(gpa, 4), "weighted_points": weighted, "credits": total_credits, "courses": parsed}
        return ToolResult("gpa", text, [card("gpa_result", "GPA計算結果", f"GPA {gpa:.2f}", copy_text=text, data=data)], True, [], data)

    def _grade_simulator(self, prompt: str, state: dict, inputs: dict) -> ToolResult:
        earned = number(inputs.get("earned_points"))
        target = number(inputs.get("target_points"))
        remaining = number(inputs.get("remaining_weight"))
        if earned is None:
            match = re.search(r"(?:現在|今)\s*(\d+(?:\.\d+)?)\s*点", prompt)
            earned = number(match.group(1)) if match else None
        if target is None:
            match = re.search(r"(?:合格|目標)(?:点|は|ライン)?\s*(\d+(?:\.\d+)?)\s*点?", prompt)
            target = number(match.group(1)) if match else None
        if remaining is None:
            match = re.search(r"残り(?:評価)?\s*(\d+(?:\.\d+)?)\s*%", prompt)
            remaining = number(match.group(1)) if match else None
        missing = [name for name, value in (("earned_points", earned), ("target_points", target), ("remaining_weight", remaining)) if value is None]
        if missing:
            text = "必要点数を計算するため、現在までの獲得点、目標点、残り評価の割合を入力してください。例：現在40点、合格60点、残り評価30%。"
            fields = [
                {"name": "earned_points", "label": "現在までの獲得点"},
                {"name": "target_points", "label": "目標・合格点"},
                {"name": "remaining_weight", "label": "残り評価の割合（%）"},
            ]
            return ToolResult("grade_simulator", text, [card("grade_simulator", "必要点数を計算", "3つの数値を入力してください。",
                                                                     action_label="必要点数を計算", fields=fields)], False, missing)
        if remaining <= 0:
            required = math.inf
        else:
            required = (target - earned) / remaining * 100
        if required <= 0:
            conclusion = "すでに目標点へ到達しています。"
        elif required > 100:
            conclusion = f"残り評価で平均 {required:.1f}% が必要なため、現在の条件だけでは到達できません。配点や救済措置を公式情報で確認してください。"
        else:
            conclusion = f"残り評価で平均 {required:.1f}% 以上が必要です。"
        text = f"計算結果：{conclusion}\n式：（目標 {target:g} − 獲得済み {earned:g}）÷ 残り比率 {remaining:g}% × 100。配点の解釈が正しいかシラバスで確認してください。"
        data = {"earned_points": earned, "target_points": target, "remaining_weight": remaining,
                "required_average_percent": None if math.isinf(required) else round(required, 4)}
        return ToolResult("grade_simulator", text, [card("grade_result", "必要点数", conclusion, copy_text=text, data=data)], True, [], data)

    def _email(self, intent: str, state: dict, reason: str, request: str) -> ToolResult:
        subject = state.get("subject", "［授業名］")
        university = state.get("university", "［大学名・学部］")
        titles = {"absence_email": "欠席のご連絡", "lateness_email": "遅刻のご連絡",
                  "late_submission_email": "課題提出遅延についてのご相談", "professor_email": "ご相談"}
        title = titles[intent]
        body = (
            f"件名：{subject}／{title}／［学籍番号・氏名］\n\n"
            "［先生名］先生\n\n"
            f"お世話になっております。{university}の［学籍番号・氏名］です。\n"
            f"{reason}\n{request}\n"
            "ご迷惑をおかけし申し訳ありません。よろしくお願いいたします。"
        )
        text = f"そのまま編集できるメール案です。事実と依頼内容を確認し、角括弧を置き換えてください。\n\n{body}"
        return ToolResult(intent, text, [card("email", CAMPUS_LABELS[intent], "角括弧を置き換えて送信前に確認してください。",
                                                   action_label="メールをコピー", copy_text=body)], True, [])

    def _professor_email(self, prompt: str, state: dict, inputs: dict) -> ToolResult:
        return self._email("professor_email", state, "［相談したい事実を簡潔に記載］", "［確認・お願いしたいことを一つ記載］")

    def _absence_email(self, prompt: str, state: dict, inputs: dict) -> ToolResult:
        return self._email("absence_email", state, "［日付］の授業を［理由］のため欠席いたします。", "必要な資料や対応がありましたら、ご教示いただけますと幸いです。")

    def _lateness_email(self, prompt: str, state: dict, inputs: dict) -> ToolResult:
        return self._email("lateness_email", state, "［理由］により、［到着予定時刻］ごろ到着する見込みです。", "到着後の対応について、指示がありましたらお願いいたします。")

    def _late_submission_email(self, prompt: str, state: dict, inputs: dict) -> ToolResult:
        return self._email("late_submission_email", state, "［課題名］について、［事実］により期限内の提出が難しい状況です。", "［提出可能日時］までの提出が可能か、ご相談させてください。")

    def _registration(self, prompt: str, state: dict, inputs: dict) -> ToolResult:
        text = (
            "履修相談では、次の順で確認してください。\n"
            "1. 入学年度の履修要項で必修・前提科目・上限単位を確認\n"
            "2. 時間割重複、卒業要件、再履修科目を一覧化\n"
            "3. 登録期限や例外対応は学生ポータルと教務窓口で確認\n"
            "学部・学年・候補科目が分かれば、確認表に整理できます。"
        )
        return ToolResult("registration", text, [card("registration_checklist", "履修確認リスト", "必修・時間割・期限を順に確認します。",
                                                              action_label="確認リストをコピー", copy_text=text)], True, [])

    def _study_plan(self, prompt: str, state: dict, inputs: dict) -> ToolResult:
        days = int(number(inputs.get("days")) or state.get("days_remaining") or 0)
        hours = number(inputs.get("hours_per_day")) or state.get("available_hours")
        subject = str(inputs.get("subject") or state.get("subject") or "")
        missing = [name for name, value in (("subject", subject), ("days", days), ("hours_per_day", hours)) if not value]
        if missing:
            text = "具体的な勉強計画を作るため、科目、試験までの日数、1日に使える時間を教えてください。例：数学、5日、1日2時間。"
            fields = [{"name": "subject", "label": "科目"}, {"name": "days", "label": "残り日数"},
                      {"name": "hours_per_day", "label": "1日に使える時間"}]
            return ToolResult("study_plan", text, [card("study_planner", "勉強計画を作成", "3項目を入力してください。",
                                                            action_label="勉強計画を作成", fields=fields)], False, missing)
        lines = []
        shown = min(days, 7)
        for day in range(1, shown + 1):
            if day == 1:
                task = "試験範囲を確認し、弱点を診断する"
            elif day == days:
                task = "間違えた問題だけを再確認し、持ち物と時間を確認する"
            elif day <= max(2, days * 2 // 3):
                task = "重要範囲の理解と標準問題を進める"
            else:
                task = "時間を測って問題演習し、誤答を復習する"
            lines.append(f"{day}日目（{hours:g}時間）：{task}")
        if days > 7:
            lines.append(f"8〜{days}日目：弱点演習と通し演習を交互に行う")
        plan = "\n".join(lines)
        text = f"{subject}の{days}日間計画です。\n{plan}\n各日の最後に10分で誤答を記録し、翌日の最初に解き直してください。"
        data = {"subject": subject, "days": days, "hours_per_day": hours, "plan": lines}
        return ToolResult("study_plan", text, [card("study_plan", f"{subject}の勉強計画", f"{days}日×{hours:g}時間",
                                                        action_label="計画をコピー", copy_text=text, data=data)], True, [], data)

    def _assignment_priority(self, prompt: str, state: dict, inputs: dict) -> ToolResult:
        assignments = inputs.get("assignments") if isinstance(inputs.get("assignments"), list) else []
        clean = []
        for index, item in enumerate(assignments):
            if not isinstance(item, dict):
                continue
            days = number(item.get("days_remaining")); hours = number(item.get("estimated_hours")); impact = number(item.get("impact", 3))
            if days is not None and hours is not None:
                urgency = (impact or 3) * 2 + hours / max(days + 0.25, 0.25)
                clean.append({"name": item.get("name", f"課題{index + 1}"), "days_remaining": days,
                              "estimated_hours": hours, "impact": impact, "priority_score": urgency})
        if not clean:
            text = "課題名、締切までの日数、予想作業時間、成績への影響を入力してください。先に『提出不能になる期限』、次に『短時間で提出可能にできる課題』を優先します。"
            fields = [{"name": "assignments", "label": "課題一覧", "example": "統計レポート・2日・3時間"}]
            return ToolResult("assignment_priority", text, [card("assignment_priority", "課題の優先順位", "課題ごとの期限と作業量を入力してください。",
                                                                       action_label="優先順位を作成", fields=fields)], False, ["assignments"])
        clean.sort(key=lambda row: (-row["priority_score"], row["days_remaining"], row["name"]))
        lines = [f"{index + 1}. {row['name']}（残り{row['days_remaining']:g}日・約{row['estimated_hours']:g}時間）" for index, row in enumerate(clean)]
        text = "優先順位：\n" + "\n".join(lines) + "\n締切と配点が授業案内どおりか確認してから着手してください。"
        return ToolResult("assignment_priority", text, [card("priority_result", "課題優先順位", "上から順に着手します。",
                                                                       action_label="一覧をコピー", copy_text=text, data={"assignments": clean})], True, [], {"assignments": clean})

    def _deadline_organizer(self, prompt: str, state: dict, inputs: dict) -> ToolResult:
        deadlines = inputs.get("deadlines") if isinstance(inputs.get("deadlines"), list) else []
        if not deadlines:
            text = "締切を整理するため、課題名・期限・所要時間を入力してください。期限が確定していないものは、確認先も一緒に記録します。"
            return ToolResult("deadline_organizer", text, [card("deadline_organizer", "締切を整理", "締切一覧を入力してください。",
                                                                      action_label="締切を整理", fields=[{"name": "deadlines", "label": "課題名・期限・所要時間"}])], False, ["deadlines"])
        rows = sorted((item for item in deadlines if isinstance(item, dict)), key=lambda item: str(item.get("deadline", "9999")))
        lines = [f"{index + 1}. {item.get('name', '課題')} — {item.get('deadline', '期限未確認')}（{item.get('estimated_hours', '時間未確認')}）" for index, item in enumerate(rows)]
        text = "締切順に整理しました。\n" + "\n".join(lines) + "\n各締切を学生ポータルで再確認し、前日に通知を設定してください。"
        return ToolResult("deadline_organizer", text, [card("deadline_list", "締切一覧", "期限順です。", action_label="一覧をコピー", copy_text=text, data={"deadlines": rows})], True, [], {"deadlines": rows})

    def _report_outline(self, prompt: str, state: dict, inputs: dict) -> ToolResult:
        topic = str(inputs.get("topic") or "［レポートの問い］")
        outline = (
            f"テーマ：{topic}\n1. 序論：問い・背景・結論の見通し\n2. 本論1：主要な根拠と資料\n"
            "3. 本論2：別の根拠または反対意見の検討\n4. 考察：根拠から言えること・限界\n"
            "5. 結論：問いへの回答と残る課題\n6. 参考文献：指定形式で統一"
        )
        text = "レポート構成案です。課題文の評価基準に合わせて見出しを調整してください。\n\n" + outline
        return ToolResult("report_outline", text, [card("report_outline", "レポート構成", "問いと根拠を対応させます。", action_label="構成をコピー", copy_text=outline)], True, [])

    def _citation_check(self, prompt: str, state: dict, inputs: dict) -> ToolResult:
        fields = {key: str(inputs.get(key, "")).strip() for key in ("author", "title", "year", "publisher", "url", "accessed")}
        missing = [key for key, value in fields.items() if not value and key in ("author", "title", "year", "publisher")]
        if all(not value for value in fields.values()):
            text = "引用情報を確認するため、著者、資料名、発行年、掲載元を入力してください。Web資料ならURLと閲覧日も記録します。"
            return ToolResult("citation_check", text, [card("citation_check", "引用を確認", "書誌情報を入力してください。", action_label="引用を確認",
                                                               fields=[{"name": key, "label": label} for key, label in (("author", "著者"), ("title", "資料名"), ("year", "発行年"), ("publisher", "掲載元"), ("url", "URL"), ("accessed", "閲覧日"))])], False, ["citation"])
        text = "引用チェック結果：" + ("不足項目は「" + "、".join(missing) + "」です。" if missing else "主要な書誌情報がそろっています。") + " 授業指定の形式と、引用範囲・利用条件も確認してください。"
        return ToolResult("citation_check", text, [card("citation_result", "引用チェック結果", text, copy_text=text, data={"fields": fields, "missing": missing})], not missing, missing, {"missing": missing})

    def _presentation_outline(self, prompt: str, state: dict, inputs: dict) -> ToolResult:
        topic = str(inputs.get("topic") or "［発表テーマ］")
        outline = f"1. 結論：{topic}で最も伝えたいこと\n2. 背景：なぜ重要か\n3. 根拠：データ・事例\n4. 検討：限界や反対意見\n5. まとめ：結論と次の行動\n6. 質疑用：出典と補足"
        text = "プレゼン構成案です。1枚1メッセージにし、発表時間の8割で終わるよう練習してください。\n\n" + outline
        return ToolResult("presentation_outline", text, [card("presentation_outline", "プレゼン構成", "結論から始める6部構成です。", action_label="構成をコピー", copy_text=outline)], True, [])

    def _career_schedule(self, prompt: str, state: dict, inputs: dict) -> ToolResult:
        text = "就活スケジュール案：\n1. 今週：締切一覧と経験素材を整理\n2. 2週間以内：企業研究とES初稿\n3. 応募前：大学キャリア窓口で確認\n4. 面接前：想定質問と逆質問を練習\n5. 毎週：授業・課題時間を先に確保\n企業ごとの公式締切を入力すれば日付順に整理できます。"
        return ToolResult("career_schedule", text, [card("career_schedule", "就活スケジュール", "学業時間を先に確保します。", action_label="計画をコピー", copy_text=text)], True, [])

    def _es_outline(self, prompt: str, state: dict, inputs: dict) -> ToolResult:
        text = "ES構成：\n1. 結論（何を伝えるか）\n2. 状況と課題\n3. 自分が取った具体的行動\n4. 工夫した理由\n5. 結果を数字・事実で示す\n6. 応募先でどう生かすか\n事実を誇張せず、企業ごとの設問と文字数に合わせてください。"
        return ToolResult("es_outline", text, [card("es_outline", "ES構成", "行動と結果を具体化します。", action_label="構成をコピー", copy_text=text)], True, [])

    def _toeic_plan(self, prompt: str, state: dict, inputs: dict) -> ToolResult:
        days = int(number(inputs.get("days")) or state.get("days_remaining") or 30)
        hours = number(inputs.get("hours_per_day")) or state.get("available_hours") or 1.0
        text = f"TOEIC {days}日計画：毎日{hours:g}時間を、語彙20%、リスニング30%、文法・読解30%、復習20%に分けます。週1回は時間を測って模試を行い、誤答を原因別に記録してください。提出期限がある場合は公式の試験日・結果発送日も確認してください。"
        return ToolResult("toeic_plan", text, [card("toeic_plan", "TOEIC勉強計画", f"{days}日・1日{hours:g}時間", action_label="計画をコピー", copy_text=text,
                                                         data={"days": days, "hours_per_day": hours})], True, [], {"days": days, "hours_per_day": hours})
