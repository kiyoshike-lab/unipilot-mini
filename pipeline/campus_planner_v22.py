from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import re
from threading import RLock
from typing import Any


INTENT_SIGNALS: dict[str, tuple[str, ...]] = {
    "exam": ("試験", "テスト", "過去問", "持込", "持ち込み"),
    "assignment": ("課題", "提出", "締切", "要件"),
    "credit": ("単位", "卒業要件", "必修"),
    "gpa": ("gpa", "gp", "成績平均"),
    "grade_simulator": ("必要点", "合格点", "残り", "配点"),
    "attendance": ("出席", "欠席", "公欠"),
    "lateness": ("遅刻", "遅延", "寝坊"),
    "registration": ("履修", "時間割", "科目登録"),
    "professor_email": ("教授", "先生", "教員", "面談", "メール"),
    "report_outline": ("レポート", "論文", "章立て"),
    "citation_check": ("引用", "参考文献", "出典", "孫引き"),
    "presentation_outline": ("発表", "プレゼン", "スライド"),
    "study_plan": ("勉強", "学習計画", "復習", "暗記"),
    "toeic_plan": ("toeic", "英語", "単語", "リスニング"),
    "career_schedule": ("就活", "面接", "企業", "es", "選考"),
    "internship": ("インターン", "実習"),
    "scholarship": ("奨学金", "給付", "貸与", "jasso"),
    "tuition": ("学費", "授業料", "延納", "分納"),
    "part_time_job": ("バイト", "アルバイト", "シフト", "労働"),
    "relationship": ("友達", "人間関係", "サークル", "グループ"),
    "campus_life": ("大学生活", "研究室", "ゼミ", "新歓"),
    "programming": ("コード", "プログラム", "エラー", "デバッグ"),
    "statistics": ("統計", "標準偏差", "回帰", "検定", "確率"),
    "ai_usage": ("生成ai", "chatgpt", "ai利用", "プロンプト"),
    "university_policy": ("学則", "規程", "制度", "大学による", "うちの大学"),
}

TOOL_INTENTS = {
    "gpa", "grade_simulator", "credit", "exam", "study_plan", "assignment",
    "professor_email", "attendance", "lateness", "registration", "report_outline",
    "presentation_outline", "career_schedule",
}

TOOL_CUES = ("計算", "何点", "何単位", "作って", "文面", "書いて", "配分", "整理", "テンプレ")
RETRIEVAL_CUES = ("とは", "意味", "制度", "根拠", "出典", "公式", "最新", "条件", "対象")
VAGUE_ONLY = ("どうしよう", "やばい", "困った", "助けて", "わからん", "無理", "相談")
COMPOUND_SPLIT = re.compile(r"(?:。|！|？|\n|それと|あと、?|しかも|ついでに|両方|どっちから|一方で)")


def _normalise(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.lower())


def detect_intents(question: str) -> list[str]:
    compact = _normalise(question)
    scored: list[tuple[int, str]] = []
    for intent, signals in INTENT_SIGNALS.items():
        score = sum(2 if len(_normalise(signal)) >= 3 else 1 for signal in signals if _normalise(signal) in compact)
        if score:
            scored.append((score, intent))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [intent for _, intent in scored]


def split_sub_intents(question: str) -> list[str]:
    parts = [part.strip(" 、。！？") for part in COMPOUND_SPLIT.split(question) if len(_normalise(part)) >= 2]
    return list(dict.fromkeys(parts))[:4] or [question.strip()]


@dataclass(frozen=True)
class AnswerPlan:
    intent: str
    sub_intents: tuple[str, ...]
    known_facts: tuple[str, ...]
    unknown_facts: tuple[str, ...]
    need_clarification: bool
    need_tool: bool
    need_retrieval: bool
    answer_depth: str
    required_sections: tuple[str, ...]
    contextual_question: str

    def to_internal_dict(self) -> dict[str, Any]:
        return asdict(self)


class CampusAnswerPlannerV22:
    """Deterministic planner used before routing; its full plan is never returned to the user."""

    @staticmethod
    def contextualize(question: str, previous_question: str | None) -> str:
        compact = _normalise(question)
        if not previous_question or len(compact) > 18 or detect_intents(question):
            return question
        if re.fullmatch(r"(?:はい|いいえ|ありがとう|了解|ok|おけ|なるほど)", compact):
            return question
        return f"{previous_question}。その条件の補足：{question}"

    def plan(self, question: str, *, previous_question: str | None = None,
             response_mode: str = "auto", tool_inputs: dict | None = None) -> AnswerPlan:
        contextual = self.contextualize(question, previous_question)
        intents = detect_intents(contextual)
        parts = split_sub_intents(question)
        segment_intents = [intent for part in parts for intent in detect_intents(part)[:1]]
        # Multiple lexical labels in one clause often describe one request (for example
        # attendance + university policy). Only independently split clauses form multi-intent.
        ordered_intents = list(dict.fromkeys(segment_intents if len(parts) >= 2 else intents[:1]))
        primary = ordered_intents[0] if ordered_intents else "general"
        compact = _normalise(question)

        known: list[str] = []
        for value in re.findall(r"\d+(?:\.\d+)?\s*(?:%|％|点|単位|時間|分|日|回|円)?", question):
            known.append(value.strip())
        for value in re.findall(r"[ぁ-んァ-ヶー一-龥々A-Za-z0-9・]{2,24}大学", question):
            known.append(value)
        if tool_inputs:
            known.extend(str(key) for key, value in tool_inputs.items() if value not in (None, "", []))

        need_tool = primary in TOOL_INTENTS and (bool(tool_inputs) or any(cue in compact for cue in TOOL_CUES))
        need_retrieval = any(cue in compact for cue in RETRIEVAL_CUES) or primary in {
            "scholarship", "tuition", "university_policy", "ai_usage", "statistics", "programming",
        }
        no_signal = not ordered_intents
        vague = no_signal and (len(compact) <= 12 or any(marker in compact for marker in VAGUE_ONLY))
        need_clarification = vague and not tool_inputs

        unknown: list[str] = []
        if primary in ("university_policy", "registration", "attendance", "credit") and not any("大学" in fact for fact in known):
            unknown.append("対象大学・年度または授業")
        if primary in ("exam", "assignment", "study_plan") and not re.search(r"(?:今日|明日|今週|来週|\d+日|締切|試験日)", question):
            unknown.append("期限")
        if need_tool and not tool_inputs:
            unknown.append("計算・作成に必要な入力値")

        complex_question = len(_normalise(question)) >= 70 or len(ordered_intents) >= 2 or len(parts) >= 3
        simple_question = len(_normalise(question)) <= 24 and len(ordered_intents) <= 1 and not any(
            cue in question for cue in ("詳しく", "具体的", "手順", "理由も", "全部"))
        if response_mode == "detailed" or complex_question:
            depth = "complex"
        elif response_mode == "short" or simple_question:
            depth = "simple"
        else:
            depth = "normal"

        sections = ["direct_answer"]
        if depth != "simple":
            sections.append("reason")
        sections.append("next_actions")
        if unknown or primary in ("university_policy", "registration", "attendance", "credit", "scholarship"):
            sections.append("conditions_or_caution")
        if need_clarification or unknown:
            sections.append("targeted_question")
        if len(ordered_intents) >= 2:
            sections.append("all_sub_intents")
        return AnswerPlan(
            intent=primary,
            sub_intents=tuple(ordered_intents or ("general",)),
            known_facts=tuple(dict.fromkeys(known)),
            unknown_facts=tuple(dict.fromkeys(unknown)),
            need_clarification=need_clarification,
            need_tool=need_tool,
            need_retrieval=need_retrieval,
            answer_depth=depth,
            required_sections=tuple(sections),
            contextual_question=contextual,
        )


class CampusConversationMemoryV22:
    """Small process-memory cache. Nothing is written to disk or used for training."""

    def __init__(self, maximum_sessions: int = 1024, turns_per_session: int = 4):
        self.maximum_sessions = maximum_sessions
        self.turns_per_session = turns_per_session
        self._sessions: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        self._lock = RLock()

    def latest_question(self, session_id: str | None) -> str | None:
        if not session_id:
            return None
        with self._lock:
            turns = self._sessions.get(session_id, [])
            return turns[-1]["question"] if turns else None

    def remember(self, session_id: str | None, question: str, category: str) -> None:
        if not session_id:
            return
        with self._lock:
            turns = self._sessions.setdefault(session_id, [])
            turns.append({"question": question, "category": category})
            del turns[:-self.turns_per_session]
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self.maximum_sessions:
                self._sessions.popitem(last=False)

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None
