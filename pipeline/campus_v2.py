from __future__ import annotations

import time

from pipeline.campus_categories import CAMPUS_LABELS, TOOL_INTENTS
from pipeline.campus_categories_v2 import CATEGORY_TO_LEVEL1, TOOL_AVAILABLE
from pipeline.campus_retrieval import CampusFAQRetriever, load_jsonl
from pipeline.campus_router_v2 import CampusRouterV2
from pipeline.campus_state_v2 import CampusSessionStoreV2
from pipeline.campus_tools import ToolResult, card
from pipeline.campus_tools_v2 import CampusToolEngineV2
from pipeline.campus_v1 import UniPilotCampusV1
from pipeline.campus_validator_v2 import CampusValidatorV2


class UniPilotCampusV2(UniPilotCampusV1):
    version = "campus-v2"

    def __init__(self, model=None, tokenizer=None,
                 router_path: str = "data/campus_v2/router/train.jsonl",
                 faq_path: str = "data/campus_v2/faq/reviewed.jsonl"):
        super().__init__(model, tokenizer, "data/campus_v1/router/train.jsonl", "data/campus_v1/faq/faq.jsonl")
        self.router = CampusRouterV2(load_jsonl(router_path))
        self.faq = CampusFAQRetriever.from_jsonl(faq_path)
        self.tools = CampusToolEngineV2()
        self.validator = CampusValidatorV2()
        self.sessions = CampusSessionStoreV2()

    @staticmethod
    def _generic_answer(category: str) -> str:
        label = CAMPUS_LABELS.get(category, "大学生活の相談")
        return (f"結論：{label}の状況を、期限・現在分かっていること・確認先に分けると次の行動を決められます。\n"
                "今やること：1. 期限を確認する 2. 不足情報を一つ書く 3. シラバス、学生ポータル、担当窓口のうち適切な確認先を選ぶ。\n"
                "大学固有の条件は推測せず、最新の公式案内で確認してください。")

    def _resolve(self, question: str, session_id: str | None, tool_inputs: dict | None) -> dict:
        started = time.perf_counter()
        before = self.sessions.get(session_id)
        state = self.sessions.update(session_id, question)
        decision = self.router.decide(question)
        pending = before.get("pending_intent")
        if pending in TOOL_AVAILABLE and decision.action == "CLARIFY" and len(question.strip()) >= 2:
            # A follow-up containing missing tool fields continues the prior tool instead of starting a second route.
            category = pending
            decision = type(decision)(
                primary=category, intents=(category,), level1=CATEGORY_TO_LEVEL1[category], top2=(category,),
                confidence=.92, confidence_band="high", action="TOOL", clarify_question=None,
                latency_ms=decision.latency_ms, evidence={**decision.evidence, "source": "session_pending_intent"},
            )
        router_evidence = decision.to_dict()
        if decision.confidence_band == "medium" and len(decision.top2) == 2:
            # Resolve top-2 with available local capabilities. A second intent replaces the first only
            # when the first has neither an FAQ candidate nor a deterministic tool.
            local_candidates = {}
            for candidate in decision.top2:
                faq_rows = [item for item in self.faq.search(question, candidate, 1) if item.get("category_match")]
                local_candidates[candidate] = {"faq_score": faq_rows[0].get("retrieval_score", 0.0) if faq_rows else 0.0,
                                               "tool_available": self.tools.can_handle(candidate)}
            router_evidence["top2_resolution"] = local_candidates
            first, second = decision.top2
            first_available = local_candidates[first]["faq_score"] > 0 or local_candidates[first]["tool_available"]
            second_available = local_candidates[second]["faq_score"] > 0 or local_candidates[second]["tool_available"]
            if not first_available and second_available:
                action = self.router.action_for(question, second, (second,), decision.confidence_band)
                decision = type(decision)(
                    primary=second, intents=(second,), level1=CATEGORY_TO_LEVEL1[second], top2=decision.top2,
                    confidence=decision.confidence, confidence_band=decision.confidence_band, action=action,
                    clarify_question=None, latency_ms=decision.latency_ms,
                    evidence={**decision.evidence, "source": "top2_local_capability"},
                )
                router_evidence["top2_selected"] = second
        category = decision.primary
        base = {"category": category, "confidence": decision.confidence, "router": router_evidence,
                "decision": decision, "state": state, "started": started}

        if decision.action == "CLARIFY":
            self.sessions.update(session_id, question, pending_intent=category if category in TOOL_AVAILABLE else "general")
            return {**base, "kind": "clarify", "documents": [], "text": decision.clarify_question}

        university_query = category == "university_policy" or self.university_specific(question)
        if university_query:
            official = self.universities.search(question, state.get("university"), 3)
            if official:
                text = official[0]["text"] + f"\n\n根拠：{official[0]['title']}（取得日 {official[0]['retrieved_at']}）"
                return {**base, "kind": "official", "documents": official, "text": text}
            university = state.get("university")
            lead = f"{university}について、" if university else "大学名・入学年度が分からないため、"
            text = (lead + "公式根拠を確認できず、制度の有無・回数・期限を断定できません。\n"
                    "今やること：学生便覧、履修要項、シラバスを確認し、見つからなければ担当教員または教務窓口へ問い合わせてください。")
            return {**base, "kind": "safety", "category": "university_policy", "documents": [], "text": text}

        if decision.action in ("TOOL", "TOOL+MODEL"):
            executable = next((intent for intent in decision.intents if self.tools.can_handle(intent)), None)
            if executable:
                result = self.tools.execute(executable, question, state, tool_inputs)
                documents = []
                if len(decision.intents) > 1:
                    for secondary in decision.intents:
                        if secondary == executable:
                            continue
                        matches = [item for item in self.faq.search(question, secondary, 2) if item.get("category_match")]
                        if matches:
                            documents.append(matches[0])
                    if documents:
                        secondary_text = documents[0]["answer"]
                        combined = result.text + "\n\nもう一つの相談：\n" + secondary_text
                        result = ToolResult(result.intent, combined, result.cards + [card(
                            "faq", CAMPUS_LABELS.get(documents[0]["category"], "FAQ"),
                            "もう一つの意図に対応する確認済みFAQです。", data={"faq_id": documents[0]["id"]},
                        )], result.completed, result.missing_fields, result.calculation)
                if result.completed:
                    self.sessions.clear_pending(session_id)
                else:
                    self.sessions.update(session_id, question, pending_intent=executable)
                return {**base, "kind": "tool", "tool": result, "documents": documents}

        documents = [item for item in self.faq.search(question, category, 3) if item.get("category_match")]
        if documents:
            answer = documents[0]["answer"]
            if documents[0].get("needs_confirmation"):
                answer += "\n大学・授業ごとに条件が異なるため、最新の公式案内も確認してください。"
            return {**base, "kind": "faq", "documents": documents, "text": answer}
        return {**base, "kind": "faq", "documents": [], "text": self._generic_answer(category)}

    def _static_result(self, resolved: dict) -> dict:
        elapsed = time.perf_counter() - resolved["started"]
        tool: ToolResult | None = resolved.get("tool")
        text = tool.text if tool else resolved["text"]
        documents = resolved.get("documents", [])
        decision = resolved["decision"]
        grounded = resolved["kind"] in ("faq", "official") and bool(documents)
        cards = tool.cards if tool else []
        if resolved["kind"] == "faq":
            cards = [card("faq", CAMPUS_LABELS.get(resolved["category"], "大学生活FAQ"),
                          "確認済みFAQを基に、次の行動を示しました。",
                          data={"faq_id": documents[0]["id"] if documents else None})]
        if resolved["kind"] == "clarify":
            cards = [card("clarify", "相談内容を確認", "該当する項目を一つ選んで続けてください。",
                          fields=[{"name": "topic", "label": "試験・課題・単位・履修・連絡など"}])]
        validation = self.validator.validate(
            resolved.get("question", ""), text, grounded=grounded, tool_result=tool is not None,
            source_urls=[item.get("source_url") for item in documents if item.get("source_url")],
            university_known=bool(resolved["state"].get("university")), category=resolved["category"],
            action=decision.action, cards=cards,
        )
        metrics = {"tokens": 0, "seconds": elapsed, "tokens_per_sec": 0.0, "eos_reached": True,
                   "kv_cache": True, "first_event_seconds": elapsed}
        return {
            "text": text, "raw_text": text, "category": resolved["category"],
            "intents": list(decision.intents), "top2": list(decision.top2),
            "category_confidence": decision.confidence, "confidence_band": decision.confidence_band,
            "route": resolved["kind"], "route_action": decision.action,
            "executed_action": "TOOL" if tool else ("RAG" if grounded else resolved["kind"].upper()),
            "router": resolved["router"], "tool": tool.intent if tool else None, "cards": cards,
            "missing_fields": tool.missing_fields if tool else [],
            "calculation": tool.calculation if tool else None, "session_state": resolved["state"],
            "retrieval": self._retrieval_rows(documents), "validator": validation.to_dict(),
            "fallback_used": False, "generation_metrics": metrics, "timing": {"total_seconds": elapsed},
            "pipeline": self.version, "external_ai_api": "OFF",
        }
