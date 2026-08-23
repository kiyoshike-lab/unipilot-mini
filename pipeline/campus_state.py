from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import re
from threading import Lock


SUBJECTS = ("数学", "英語", "統計", "物理", "化学", "生物", "情報", "プログラミング", "経済", "法学", "心理")


class CampusSessionStore:
    def __init__(self, maximum_sessions: int = 256):
        self.maximum_sessions = maximum_sessions
        self._sessions: OrderedDict[str, dict] = OrderedDict()
        self._lock = Lock()

    def get(self, session_id: str | None) -> dict:
        if not session_id:
            return {}
        with self._lock:
            value = self._sessions.get(session_id, {})
            if session_id in self._sessions:
                self._sessions.move_to_end(session_id)
            return deepcopy(value)

    def update(self, session_id: str | None, text: str, pending_intent: str | None = None) -> dict:
        state = self.get(session_id)
        university = re.search(r"([ぁ-んァ-ヶー一-龥々A-Za-z0-9・]{2,24}大学)", text)
        grade = re.search(r"([1-6])\s*年(?:生)?", text)
        days = re.search(r"(?:あと|まで)?\s*(\d+)\s*日", text)
        hours = re.search(r"(\d+(?:\.\d+)?)\s*時間", text)
        subject = next((item for item in SUBJECTS if item in text), None)
        if university:
            state["university"] = university.group(1)
        if grade:
            state["grade"] = int(grade.group(1))
        if days:
            state["days_remaining"] = int(days.group(1))
        elif "明日" in text:
            state["days_remaining"] = 1
        if hours:
            state["available_hours"] = float(hours.group(1))
        if subject:
            state["subject"] = subject
        if pending_intent is not None:
            state["pending_intent"] = pending_intent
        if session_id:
            with self._lock:
                self._sessions[session_id] = deepcopy(state)
                self._sessions.move_to_end(session_id)
                while len(self._sessions) > self.maximum_sessions:
                    self._sessions.popitem(last=False)
        return state

    def clear_pending(self, session_id: str | None) -> None:
        if not session_id:
            return
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].pop("pending_intent", None)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None
