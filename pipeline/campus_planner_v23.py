from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from pipeline.campus_planner_v22 import (
    AnswerPlan,
    CampusAnswerPlannerV22,
    INTENT_SIGNALS,
    _normalise,
    split_sub_intents,
)


V23_INTENT_SIGNALS: dict[str, tuple[str, ...]] = {
    **INTENT_SIGNALS,
    "gpa": (*INTENT_SIGNALS["gpa"], "目標まで", "どの程度上げ"),
    "professor_email": (*INTENT_SIGNALS["professor_email"], "丁寧に確認", "丁寧に聞"),
    "citation_check": (*INTENT_SIGNALS["citation_check"], "翻訳書", "原著", "引用元"),
    "toeic_plan": (*INTENT_SIGNALS["toeic_plan"], "模試", "時間切れ"),
    "scholarship": (*INTENT_SIGNALS["scholarship"], "継続手続き", "採用後の手続き"),
    "part_time_job": (*INTENT_SIGNALS["part_time_job"], "辞める時期", "退職時期"),
    "ai_usage": (*INTENT_SIGNALS["ai_usage"], "個人情報を消", "資料を入力", "匿名化"),
    "assignment_priority": ("優先順位", "どっちから", "先にやる", "順番", "複数の課題"),
    "deadline_organizer": ("締切管理", "期限一覧", "予定が重なる", "締切が散らば"),
    "absence_email": ("欠席メール", "欠席連絡", "休む連絡", "交通障害", "運休"),
    "lateness_email": ("遅刻メール", "遅刻連絡", "到着見込み"),
    "late_submission_email": ("提出遅れ", "遅延連絡", "期限に遅れる"),
    "internship": ("インターン", "長期と短期", "短期と長期", "実習先", "就業体験"),
    "relationship": ("人間関係", "友達", "グループ", "サークル", "距離を置"),
    "general": ("相談", "困りごと", "どうしよう", "やばい"),
}

PRIORITY_CUES = ("どっち", "どちら", "優先", "先に", "順番", "同時", "重な")
IMMEDIATE_CUES = ("今日", "今すぐ", "まず", "最初", "明日", "締切", "間に合")
TIME_CUES = ("時間配分", "残り時間", "何時間", "何分", "空き時間", "今日まで", "明日")
VERIFY_CUES = ("どこ", "何を確認", "公式", "根拠", "問い合わせ", "LMS", "ポータル", "資料")
MESSAGE_CUES = ("メール", "連絡", "文面", "伝え", "件名", "問い合わせ")
REASON_CUES = ("なぜ", "理由", "根拠", "どうして")


def detect_intents_v23(question: str) -> list[str]:
    compact = _normalise(question)
    scored: list[tuple[int, int, str]] = []
    for order, (intent, signals) in enumerate(V23_INTENT_SIGNALS.items()):
        matched = [signal for signal in signals if _normalise(signal) in compact]
        if matched:
            strength = sum(max(1, min(5, len(_normalise(signal)) // 2)) for signal in matched)
            scored.append((strength, -order, intent))
    scored.sort(reverse=True)
    return [intent for _, _, intent in scored]


@dataclass(frozen=True)
class AtomicRequirement:
    id: str
    label: str
    signals: tuple[str, ...]
    intent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "signals": list(self.signals), "intent": self.intent}


@dataclass(frozen=True)
class AnswerPlanV23(AnswerPlan):
    primary_category: str
    secondary_category: str | None
    atomic_requirements: tuple[AtomicRequirement, ...]
    action_timeline: bool


class CampusCompletenessPlannerV23(CampusAnswerPlannerV22):
    """Planner v2: deterministic atomic requirements, without exposing chain-of-thought."""

    @staticmethod
    def _segment_intents(question: str) -> list[str]:
        parts = split_sub_intents(question)
        result = []
        for part in parts:
            intents = detect_intents_v23(part)
            if intents:
                result.append(intents[0])
        return list(dict.fromkeys(result))

    @staticmethod
    def _requirements(question: str, intents: tuple[str, ...]) -> tuple[AtomicRequirement, ...]:
        requirements: list[AtomicRequirement] = []
        for intent in intents:
            label_signals = V23_INTENT_SIGNALS.get(intent, (intent,))
            requirements.append(AtomicRequirement(
                f"address_{intent}", f"{intent}への回答", tuple(label_signals[:5]), intent,
            ))
        if any(cue in question for cue in PRIORITY_CUES):
            requirements.append(AtomicRequirement("priority", "優先順位", ("優先順位", "先に", "順番", "最優先")))
        if any(cue in question for cue in IMMEDIATE_CUES):
            requirements.append(AtomicRequirement("immediate_action", "直近の行動", ("今すぐ", "今日", "今やること", "まず")))
        if any(cue in question for cue in TIME_CUES) and len(intents) >= 2:
            requirements.append(AtomicRequirement("time_allocation", "時間配分", ("時間配分", "使える時間", "残り時間", "時間を分け")))
        if any(cue in question for cue in VERIFY_CUES):
            requirements.append(AtomicRequirement("verification", "確認対象と確認先", ("確認", "公式", "LMS", "窓口", "問い合わせ")))
        if ("公式" in question and any(token in question for token in ("LMS", "ポータル", "表示"))
                and any(token in question for token in ("違", "優先", "どちら"))):
            requirements.append(AtomicRequirement(
                "source_conflict", "情報源の優先と差分確認", ("公式ページ", "LMS", "差分", "照合"),
            ))
        if any(cue in question for cue in MESSAGE_CUES):
            requirements.append(AtomicRequirement("communication", "伝える内容", ("件名", "本文", "伝える内容", "問い合わせ内容")))
        if any(cue in question for cue in REASON_CUES):
            requirements.append(AtomicRequirement("reason", "理由", ("理由", "ためです", "根拠")))
        if not requirements:
            requirements.append(AtomicRequirement("direct_answer", "質問への直接回答", ("結論", "まず", "今やること")))
        unique = {requirement.id: requirement for requirement in requirements}
        return tuple(unique.values())

    def plan(self, question: str, *, previous_question: str | None = None,
             response_mode: str = "auto", tool_inputs: dict | None = None) -> AnswerPlanV23:
        contextual = self.contextualize(question, previous_question)
        base = super().plan(question, previous_question=previous_question,
                            response_mode=response_mode, tool_inputs=tool_inputs)
        segment_intents = self._segment_intents(question)
        full_intents = detect_intents_v23(contextual)
        ordered = list(dict.fromkeys(segment_intents or full_intents[:1] or list(base.sub_intents)))
        if len(split_sub_intents(question)) >= 2:
            for intent in full_intents:
                if intent not in ordered and len(ordered) < 4:
                    ordered.append(intent)
        primary = ordered[0] if ordered else base.intent
        secondary = ordered[1] if len(ordered) >= 2 else next(
            (intent for intent in full_intents if intent != primary), None,
        )
        compact = _normalise(question)
        explicit_tool = bool(tool_inputs) or any(cue in compact for cue in (
            "計算", "何点必要", "何単位", "作って", "文面を書", "メールを書", "配分して", "テンプレ",
        ))
        informational_conflict = any(cue.lower() in question.lower() for cue in (
            "公式ページ", "LMSの表示", "どちらを優先", "違う", "根拠", "出典", "制度を教えて",
        ))
        need_tool = explicit_tool and not informational_conflict
        need_retrieval = informational_conflict or base.need_retrieval or any(cue in question for cue in VERIFY_CUES)
        vague = primary == "general" and len(compact) <= 12
        requirements = self._requirements(question, tuple(ordered or (primary,)))
        complex_question = len(compact) >= 70 or len(ordered) >= 2
        simple_question = len(compact) <= 24 and len(requirements) <= 2 and not any(
            cue in question for cue in ("詳しく", "具体的", "手順", "全部", "理由")
        )
        depth = "complex" if response_mode == "detailed" or complex_question else (
            "simple" if response_mode == "short" or simple_question else "normal"
        )
        return AnswerPlanV23(
            intent=primary,
            sub_intents=tuple(ordered or (primary,)),
            known_facts=base.known_facts,
            unknown_facts=base.unknown_facts,
            need_clarification=vague and not need_tool,
            need_tool=need_tool,
            need_retrieval=need_retrieval,
            answer_depth=depth,
            required_sections=base.required_sections,
            contextual_question=contextual,
            primary_category=primary,
            secondary_category=secondary,
            atomic_requirements=requirements,
            action_timeline=any(cue in question for cue in (*IMMEDIATE_CUES, *PRIORITY_CUES)),
        )


def requirement_coverage(answer: str, requirements: tuple[AtomicRequirement, ...]) -> dict[str, Any]:
    covered: dict[str, bool] = {}
    normalised = answer.lower()
    for requirement in requirements:
        if requirement.id == "verification":
            covered[requirement.id] = "確認先" in answer or (
                "公式" in answer and any(token in answer for token in ("LMS", "シラバス", "窓口", "学生便覧"))
            )
        elif requirement.id == "communication":
            covered[requirement.id] = any(token in answer for token in ("伝える内容", "問い合わせ内容", "件名：", "本文："))
        elif requirement.id == "time_allocation":
            covered[requirement.id] = "時間配分" in answer or (
                "使える時間" in answer and any(token in answer for token in ("分け", "配分", "枠"))
            )
        elif requirement.id == "source_conflict":
            covered[requirement.id] = (
                "公式ページ" in answer and "LMS" in answer
                and any(token in answer for token in ("差分", "照合", "優先"))
            )
        else:
            covered[requirement.id] = any(signal.lower() in normalised for signal in requirement.signals)
    total = max(1, len(requirements))
    answered = sum(covered.values())
    return {
        "answered": answered,
        "total": len(requirements),
        "score": round(answered / total, 4),
        "covered": covered,
        "missing": [requirement_id for requirement_id, value in covered.items() if not value],
    }
