from __future__ import annotations

from copy import deepcopy
import re

from pipeline.campus_state import CampusSessionStore


class CampusSessionStoreV2(CampusSessionStore):
    def update(self, session_id: str | None, text: str, pending_intent: str | None = None) -> dict:
        state = super().update(session_id, text, pending_intent)
        patterns = {
            "gpa": r"(?:現在|累積)?GPA\s*[:：は]?\s*(\d(?:\.\d+)?)",
            "target_gpa": r"目標GPA\s*[:：は]?\s*(\d(?:\.\d+)?)",
            "earned_credits": r"(?:取得済み|今まで|現在)\s*(\d+(?:\.\d+)?)\s*単位",
            "required_credits": r"(?:必要|卒業要件)\s*(\d+(?:\.\d+)?)\s*単位",
            "remaining_time_hours": r"残り\s*(\d+(?:\.\d+)?)\s*時間",
            "exam_date": r"試験日\s*[:：は]?\s*(20\d{2}-\d{2}-\d{2})",
            "deadline": r"締切\s*[:：は]?\s*(20\d{2}-\d{2}-\d{2})",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.I)
            if match:
                state[key] = match.group(1) if key in ("exam_date", "deadline") else float(match.group(1))
        tasks = re.findall(r"(?:課題|タスク)[:：]\s*([^、。]+)", text)
        if tasks:
            state["tasks"] = list(dict.fromkeys([*state.get("tasks", []), *tasks]))[-20:]
        if session_id:
            with self._lock:
                self._sessions[session_id] = deepcopy(state)
                self._sessions.move_to_end(session_id)
                while len(self._sessions) > self.maximum_sessions:
                    self._sessions.popitem(last=False)
        return state

