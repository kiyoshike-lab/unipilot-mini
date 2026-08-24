from __future__ import annotations

import re

from pipeline.campus_composer_v22 import CampusAnswerComposerV22
from pipeline.campus_generalizer_v22 import CampusResponseGeneralizerV22
from pipeline.campus_planner_v22 import CampusAnswerPlannerV22, CampusConversationMemoryV22, INTENT_SIGNALS
from pipeline.campus_retrieval_v22 import CampusKnowledgeRetrieverV22
from pipeline.campus_tools import card
from pipeline.campus_v21 import UniPilotCampusV21


class UniPilotCampusV22(UniPilotCampusV21):
    """Opt-in knowledge expansion; Campus v2.1 routing and tools stay frozen."""

    version = "campus-v2.2"
    UNIVERSITY_POLICY_CUES = (
        "欠席", "追試", "卒業条件", "履修取消", "申請期限", "延納期限", "単位認定",
        "使用条件", "延滞", "学費", "奨学金", "回数", "上限", "規程", "学則",
    )
    INFORMATIONAL_CUES = ("とは", "意味", "定義", "概要", "仕組み", "説明", "重要な理由", "根拠")
    TOOL_REQUEST_CUES = ("計算して", "作って", "書いて", "整理して", "入力", "何点必要", "送る文面")
    PERSONAL_UNIVERSITY_CUES = ("うちの大学", "この大学", "所属大学", "自分の大学")
    GENERIC_UNIVERSITY_NAMES = ("日本の大学", "全国の大学", "一般の大学", "各大学")

    def __init__(self, model=None, tokenizer=None,
                 router_path: str = "data/campus_v2/router/train.jsonl",
                 adversarial_path: str = "data/campus_v21/router/adversarial-train-1500.jsonl"):
        super().__init__(model, tokenizer, router_path, adversarial_path)
        self.knowledge = CampusKnowledgeRetrieverV22.from_files()
        self.knowledge_composer = CampusAnswerComposerV22()
        self.answer_planner = CampusAnswerPlannerV22()
        self.generalizer = CampusResponseGeneralizerV22()
        self.conversation_memory = CampusConversationMemoryV22()

    @staticmethod
    def _high_confidence_faq(resolved: dict) -> bool:
        documents = resolved.get("documents") or []
        if resolved.get("kind") != "faq" or not documents:
            return False
        first = documents[0]
        return bool(first.get("category_match")) and float(first.get("confidence", 0.0)) >= .72

    def _resolve_v22(
        self,
        question: str,
        session_id: str | None,
        tool_inputs: dict | None,
        response_mode: str,
    ) -> dict:
        resolved = super()._resolve(question, session_id, tool_inputs)
        resolved["question"] = question
        university_names = re.findall(r"[ぁ-んァ-ヶー一-龥々A-Za-z0-9・]{2,24}大学", question)
        explicit_university = next((
            name for name in university_names
            if not any(generic in name for generic in self.GENERIC_UNIVERSITY_NAMES)
        ), None)
        university_query = (
            any(cue in question for cue in self.PERSONAL_UNIVERSITY_CUES)
            or bool(explicit_university and any(cue in question for cue in self.UNIVERSITY_POLICY_CUES))
        )
        informational = (
            any(cue in question for cue in self.INFORMATIONAL_CUES)
            and not any(cue in question for cue in self.TOOL_REQUEST_CUES)
        )
        # Priority 1/2: deterministic tools, clarification and high-confidence reviewed FAQ.
        if resolved["kind"] == "tool" and resolved.get("tool") and resolved["tool"].completed:
            return resolved
        if not university_query and (
            (resolved["kind"] == "tool" and not informational)
            or resolved["kind"] == "clarify"
            or self._high_confidence_faq(resolved)
        ):
            return resolved

        university = resolved["state"].get("university")
        documents, retrieval_meta = self.knowledge.search(
            question,
            resolved["category"],
            university=university,
            top_k=5,
            response_mode=response_mode,
        )
        if university_query:
            documents = [
                row for row in documents
                if row.get("university_specific") and row.get("university_name") == university
            ]
        if not documents:
            resolved["v22_retrieval"] = retrieval_meta
            if university_query:
                resolved.pop("tool", None)
                resolved.update(
                    kind="safety",
                    category="university_policy",
                    documents=[],
                    text=(f"{university or '対象大学'}について、再利用可能な公式根拠を確認できないため、"
                          "制度の数字・期限・回数を断定できません。\n"
                          "今やること：対象年度の学生便覧、履修要項、シラバス、大学公式窓口の順に確認してください。"),
                )
            return resolved

        if informational and documents[0].get("score_components", {}).get("title_bonus", 0.0) > 0:
            knowledge_category = documents[0].get("category") or resolved["category"]
            if knowledge_category != resolved["category"]:
                resolved["router"]["knowledge_resolution"] = {
                    "original_category": resolved["category"],
                    "selected_category": knowledge_category,
                    "document_id": documents[0]["id"],
                }
                resolved["category"] = knowledge_category

        text, grounding = self.knowledge_composer.compose_grounded(
            question, resolved["category"], documents, response_mode,
        )
        resolved.update(
            kind="official",
            documents=documents,
            text=text,
            v22_grounding=grounding,
            v22_retrieval=retrieval_meta,
            original_route=resolved.get("kind"),
        )
        return resolved

    @staticmethod
    def _sources(documents: list[dict]) -> list[dict]:
        result = []
        seen = set()
        for row in documents:
            key = (row.get("source_url"), row.get("title"))
            if key in seen:
                continue
            seen.add(key)
            result.append({
                "id": row["id"],
                "title": row["title"],
                "publisher": row.get("publisher") or row.get("source"),
                "url": row.get("source_url"),
                "license": row.get("license"),
                "license_url": row.get("license_url"),
                "retrieved_at": row.get("retrieved_at"),
                "last_verified_at": row.get("last_verified_at"),
                "confidence": row.get("confidence"),
                "stale": bool(row.get("stale")),
                "source_type": row.get("source_type"),
            })
        return result

    def _result_v22(self, resolved: dict, response_mode: str) -> dict:
        full_faq_text = resolved.get("text") if resolved.get("kind") == "faq" else None
        result = super()._static_result(resolved)
        if full_faq_text and response_mode != "short":
            # v2.1's simple-question compressor remains frozen; v2.2 normal/detailed
            # modes retain the complete reviewed FAQ instead of shortening it.
            result["text"] = full_faq_text
            result["raw_text"] = full_faq_text
            documents = resolved.get("documents", [])
            result["validator"] = self.validator.validate(
                resolved.get("question", ""), full_faq_text, grounded=bool(documents), tool_result=False,
                source_urls=[row.get("source_url") for row in documents if row.get("source_url")],
                university_known=bool(resolved["state"].get("university")), category=resolved["category"],
                action=resolved["decision"].action, cards=result.get("cards", []),
            ).to_dict()
        result["response_mode"] = response_mode
        result["pipeline"] = self.version
        result["knowledge_version"] = "campus-v2.2"
        retrieval_meta = resolved.get("v22_retrieval")
        if retrieval_meta:
            result["timing"]["retrieval_ms"] = retrieval_meta["latency_ms"]
            result["retrieval_meta"] = retrieval_meta
        grounding = resolved.get("v22_grounding")
        if not grounding:
            return result
        documents = resolved.get("documents", [])
        sources = self._sources(documents)
        result.update(
            route="rag",
            route_action="RAG",
            executed_action="RAG",
            sources=sources,
            grounding=grounding,
            retrieval=[
                {
                    "id": row["id"],
                    "category": row.get("category"),
                    "sub_category": row.get("sub_category"),
                    "question": row.get("title"),
                    "score": row.get("retrieval_score", 0.0),
                    "source": row.get("source"),
                    "publisher": row.get("publisher"),
                    "source_url": row.get("source_url"),
                    "license": row.get("license"),
                    "retrieved_at": row.get("retrieved_at"),
                    "last_verified_at": row.get("last_verified_at"),
                    "stale": row.get("stale", False),
                    "selected_text": row.get("selected_text"),
                }
                for row in documents
            ],
        )
        result["cards"] = [*result.get("cards", []), card(
            "sources",
            "この回答の出典",
            f"{len(sources)}件のローカル保存済み資料を参照しました。",
            data={"sources": sources, "freshness_warning": any(item["stale"] for item in sources)},
        )]
        return result

    def answer(self, question: str, max_new_tokens: int = 100, temperature: float = 0.0, top_k: int = 40,
               top_p: float = 0.9, repetition_penalty: float = 1.1, response_mode: str = "auto",
               session_id: str | None = None, tool_inputs: dict | None = None) -> dict:
        detail_followup = self.knowledge_composer.is_detail_followup(question)
        previous_question = self.conversation_memory.latest_question(session_id)
        plan = self.answer_planner.plan(
            question,
            previous_question=previous_question,
            response_mode="detailed" if detail_followup else response_mode,
            tool_inputs=tool_inputs,
        )
        prior = previous_question if detail_followup else None
        effective_question = prior or plan.contextual_question
        if not prior and plan.contextual_question != question and plan.intent != "general":
            route_hint = INTENT_SIGNALS.get(plan.intent, (plan.intent,))[0]
            effective_question = f"{route_hint}について：{plan.contextual_question}"
        mode = {"simple": "short", "normal": "normal", "complex": "detailed"}[plan.answer_depth]
        resolved = self._resolve_v22(effective_question, session_id, tool_inputs, mode)
        resolved["question"] = question
        result = self._result_v22(resolved, mode)
        if prior:
            result["followup_of"] = effective_question
        improvement = self.generalizer.improve(question, result["text"], plan, result)
        result["text"] = improvement.text
        result["raw_text"] = improvement.text
        result["answer_depth"] = plan.answer_depth
        result["quality_checks"] = improvement.checks
        result["revision_count"] = improvement.revision_count
        result["planner_hidden"] = True
        if improvement.card and all(card_item.get("kind") != improvement.card["kind"] for card_item in result["cards"]):
            result["cards"].append(improvement.card)
        source_urls = [
            item.get("url") or item.get("source_url")
            for item in (result.get("sources") or result.get("retrieval") or [])
            if item.get("url") or item.get("source_url")
        ]
        result["validator"] = self.validator.validate(
            question,
            improvement.text,
            grounded=bool(source_urls),
            tool_result=result.get("route") == "tool",
            source_urls=source_urls,
            university_known=bool(result.get("session_state", {}).get("university")),
            category=result.get("category"),
            action=result.get("route_action"),
            cards=result.get("cards", []),
        ).to_dict()
        self.conversation_memory.remember(session_id, effective_question, result["category"])
        return result

    def iter_answer(self, question: str, max_new_tokens: int = 100, temperature: float = 0.0, top_k: int = 40,
                    top_p: float = 0.9, repetition_penalty: float = 1.1, response_mode: str = "auto",
                    session_id: str | None = None, tool_inputs: dict | None = None):
        # Static RAG/tool paths execute once. The UI receives one final snapshot and cannot trigger a duplicate inference.
        result = self.answer(question, max_new_tokens, temperature, top_k, top_p, repetition_penalty,
                             response_mode, session_id, tool_inputs)
        yield {
            **result["generation_metrics"],
            "text": result["text"],
            "phase": "complete",
            "pipeline": self.version,
            "route": result["route"],
            "category": result["category"],
            "cards": result["cards"],
            "sources": result.get("sources", []),
            "response_mode": result["response_mode"],
            "answer_depth": result["answer_depth"],
            "revision_count": result["revision_count"],
            "validator": result["validator"],
            "session_state": result["session_state"],
        }
