from __future__ import annotations

import json
from pathlib import Path
import time

from inference.generate import generate_text, iter_generate_text
from pipeline.campus_categories import CAMPUS_LABELS, TOOL_INTENTS, UNIVERSITY_SPECIFIC_PHRASES
from pipeline.campus_retrieval import CampusFAQRetriever, CampusPublicKnowledge, CampusUniversityKnowledge, load_jsonl
from pipeline.campus_router import CampusHybridRouter
from pipeline.campus_state import CampusSessionStore
from pipeline.campus_tools import CampusToolEngine, ToolResult, card
from pipeline.campus_validator import CampusValidator


SYSTEM_CAMPUS = (
    "あなたは大学生活OSのUniPilot Campusです。結論と今やることを先に示します。"
    "大学固有の制度は、公式根拠がない限り断定せず確認先を案内します。"
)


class UniPilotCampusV1:
    version = "campus-v1"

    def __init__(self, model=None, tokenizer=None,
                 router_path: str = "data/campus_v1/router/train.jsonl",
                 faq_path: str = "data/campus_v1/faq/faq.jsonl"):
        self.model = model
        self.tokenizer = tokenizer
        examples = load_jsonl(router_path)
        self.router = CampusHybridRouter(examples)
        self.faq = CampusFAQRetriever.from_jsonl(faq_path)
        self.public_knowledge = CampusPublicKnowledge.from_jsonl()
        self.universities = CampusUniversityKnowledge.from_root()
        self.tools = CampusToolEngine()
        self.validator = CampusValidator()
        self.sessions = CampusSessionStore()

    @staticmethod
    def university_specific(question: str) -> bool:
        value = question.lower()
        return any(phrase in value for phrase in UNIVERSITY_SPECIFIC_PHRASES) or "うちの大学" in value or "この大学" in value

    def _resolve(self, question: str, session_id: str | None, tool_inputs: dict | None) -> dict:
        started = time.perf_counter()
        before = self.sessions.get(session_id)
        state = self.sessions.update(session_id, question)
        category, confidence, evidence = self.router.predict(question)
        pending = before.get("pending_intent")
        if pending in TOOL_INTENTS and category in ("general", "math", "statistics", "exam", "schedule"):
            category, confidence = pending, max(confidence, 0.9)
            evidence = {**evidence, "source": "session_pending_intent"}

        if category in TOOL_INTENTS:
            result = self.tools.execute(category, question, state, tool_inputs)
            if result.completed:
                self.sessions.clear_pending(session_id)
            else:
                state = self.sessions.update(session_id, question, pending_intent=category)
            return {"kind": "tool", "category": category, "confidence": confidence, "router": evidence,
                    "state": state, "tool": result, "started": started}

        university_query = self.university_specific(question)
        if university_query:
            official = self.universities.search(question, state.get("university"), 3)
            if official:
                answer = official[0]["text"] + f"\n\n根拠：{official[0]['title']}（取得日 {official[0]['retrieved_at']}）"
                return {"kind": "official", "category": category, "confidence": confidence, "router": evidence,
                        "state": state, "documents": official, "text": answer, "started": started}
            university = state.get("university")
            lead = f"{university}について、" if university else "大学名が分からないため、"
            answer = (lead + "公式根拠を確認できず、制度の有無や回数を断定できません。\n"
                      "今やること：現在年度の学生便覧・履修要項・シラバスを確認し、担当教員または教務窓口へ問い合わせてください。")
            return {"kind": "safety", "category": "university_policy", "confidence": confidence,
                    "router": evidence, "state": state, "documents": [], "text": answer, "started": started}

        documents = self.faq.search(question, category, 3)
        if documents and documents[0].get("confidence", 0) >= 0.66 and documents[0]["category_match"]:
            row = documents[0]
            answer = row["answer"]
            if row.get("needs_confirmation"):
                answer += "\n大学・授業ごとに条件が異なるため、最新の公式案内も確認してください。"
            return {"kind": "faq", "category": category, "confidence": confidence, "router": evidence,
                    "state": state, "documents": documents, "text": answer, "started": started}
        public_documents = self.public_knowledge.search(question, category, 2)
        return {"kind": "model", "category": category, "confidence": confidence, "router": evidence,
                "state": state, "documents": public_documents or documents, "started": started}

    def _prompt(self, question: str, documents: list[dict], state: dict) -> str:
        context = "".join(f"[{row.get('question') or row.get('title')}]\n{row.get('answer') or row.get('text')}\n" for row in documents[:2])
        known = json.dumps({key: state[key] for key in ("university", "grade", "subject", "days_remaining") if key in state}, ensure_ascii=False)
        fixed = f"<BOS><SYSTEM>\n{SYSTEM_CAMPUS}\n<CONTEXT>\n{context}<USER>\n会話状態：{known}\n{question}\n<ASSISTANT>\n"
        if self.model is None or self.tokenizer is None:
            return fixed
        ids = self.tokenizer.encode(fixed)
        if len(ids) <= self.model.config.context_length - 64:
            return fixed
        context = self.tokenizer.decode(self.tokenizer.encode(context)[:64], skip_special=True)
        return f"<BOS><SYSTEM>\n{SYSTEM_CAMPUS}\n<CONTEXT>\n{context}<USER>\n{question}\n<ASSISTANT>\n"

    @staticmethod
    def _retrieval_rows(documents: list[dict]) -> list[dict]:
        return [{"id": row["id"], "category": row.get("category"), "question": row.get("question") or row.get("title"),
                 "score": row.get("retrieval_score", 0.0), "source": row.get("source"),
                 "source_url": row.get("source_url"), "retrieved_at": row.get("retrieved_at")}
                for row in documents]

    def _static_result(self, resolved: dict) -> dict:
        elapsed = time.perf_counter() - resolved["started"]
        tool: ToolResult | None = resolved.get("tool")
        text = tool.text if tool else resolved["text"]
        documents = resolved.get("documents", [])
        grounded = resolved["kind"] in ("faq", "official")
        validation = self.validator.validate(
            resolved.get("question", ""), text, grounded=grounded, tool_result=tool is not None,
            source_urls=[row.get("source_url") for row in documents if row.get("source_url")],
            university_known=bool(resolved["state"].get("university")),
        )
        cards = tool.cards if tool else []
        if resolved["kind"] == "faq" and documents:
            cards = [card("faq", CAMPUS_LABELS.get(resolved["category"], "大学生活FAQ"),
                          "確認済みFAQを基に回答しました。", data={"faq_id": documents[0]["id"]})]
        metrics = {"tokens": 0, "seconds": elapsed, "tokens_per_sec": 0.0, "eos_reached": True,
                   "kv_cache": True, "first_event_seconds": elapsed}
        return {"text": text, "raw_text": text, "category": resolved["category"],
                "category_confidence": resolved["confidence"], "route": resolved["kind"],
                "router": resolved["router"], "tool": tool.intent if tool else None, "cards": cards,
                "missing_fields": tool.missing_fields if tool else [], "calculation": tool.calculation if tool else None,
                "session_state": resolved["state"], "retrieval": self._retrieval_rows(documents),
                "validator": validation.to_dict(), "fallback_used": False, "generation_metrics": metrics,
                "timing": {"total_seconds": elapsed}, "pipeline": self.version, "external_ai_api": "OFF"}

    def answer(self, question: str, max_new_tokens: int = 100, temperature: float = 0.0, top_k: int = 40,
               top_p: float = 0.9, repetition_penalty: float = 1.1, response_mode: str = "auto",
               session_id: str | None = None, tool_inputs: dict | None = None) -> dict:
        resolved = self._resolve(question, session_id, tool_inputs)
        resolved["question"] = question
        if resolved["kind"] != "model" or self.model is None:
            if resolved["kind"] == "model":
                resolved.update(kind="safety", text=self.validator.safe_fallback())
            return self._static_result(resolved)
        prompt = self._prompt(question, resolved["documents"], resolved["state"])
        text, metrics = generate_text(self.model, self.tokenizer, prompt, min(max_new_tokens, 96),
                                      temperature, top_k, top_p, repetition_penalty)
        validation = self.validator.validate(question, text, grounded=False,
                                             university_known=bool(resolved["state"].get("university")))
        fallback = not validation.valid
        final = self.validator.safe_fallback() if fallback else text
        elapsed = time.perf_counter() - resolved["started"]
        return {"text": final, "raw_text": text, "category": resolved["category"],
                "category_confidence": resolved["confidence"], "route": "model", "router": resolved["router"],
                "tool": None, "cards": [], "missing_fields": [], "calculation": None,
                "session_state": resolved["state"], "retrieval": self._retrieval_rows(resolved["documents"]),
                "validator": validation.to_dict(), "fallback_used": fallback, "generation_metrics": metrics,
                "timing": {"total_seconds": elapsed}, "pipeline": self.version, "external_ai_api": "OFF"}

    def iter_answer(self, question: str, max_new_tokens: int = 100, temperature: float = 0.0, top_k: int = 40,
                    top_p: float = 0.9, repetition_penalty: float = 1.1, response_mode: str = "auto",
                    session_id: str | None = None, tool_inputs: dict | None = None):
        resolved = self._resolve(question, session_id, tool_inputs)
        resolved["question"] = question
        if resolved["kind"] != "model" or self.model is None:
            if resolved["kind"] == "model":
                resolved.update(kind="safety", text=self.validator.safe_fallback())
            result = self._static_result(resolved)
            yield {**result["generation_metrics"], "text": result["text"], "phase": "complete",
                   "pipeline": self.version, "route": result["route"], "category": result["category"],
                   "cards": result["cards"], "validator": result["validator"], "session_state": result["session_state"]}
            return
        prompt = self._prompt(question, resolved["documents"], resolved["state"])
        last = None
        for snapshot in iter_generate_text(self.model, self.tokenizer, prompt, min(max_new_tokens, 96),
                                           temperature, top_k, top_p, repetition_penalty):
            last = snapshot
            yield {**snapshot, "phase": "generating", "pipeline": self.version, "route": "model",
                   "category": resolved["category"], "cards": []}
        raw = last["text"] if last else ""
        validation = self.validator.validate(question, raw, university_known=bool(resolved["state"].get("university")))
        if not validation.valid:
            yield {"text": self.validator.safe_fallback(), "tokens": last["tokens"] if last else 0,
                   "seconds": last["seconds"] if last else 0.0, "tokens_per_sec": last["tokens_per_sec"] if last else 0.0,
                   "eos_reached": True, "kv_cache": True, "phase": "validated_replacement", "pipeline": self.version,
                   "route": "model", "category": resolved["category"], "cards": [], "fallback_used": True,
                   "validator": validation.to_dict(), "session_state": resolved["state"]}
