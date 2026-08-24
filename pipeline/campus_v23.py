from __future__ import annotations

import re

from pipeline.campus_generalizer_v23 import CampusCoverageGeneralizerV23, V23_DEPTH_LIMITS
from pipeline.campus_planner_v22 import CampusConversationMemoryV22
from pipeline.campus_planner_v23 import CampusCompletenessPlannerV23
from pipeline.campus_retrieval_v23 import CONFIDENCE_ORDER, CampusKnowledgeRetrieverV23
from pipeline.campus_v21 import UniPilotCampusV21
from pipeline.campus_v22 import UniPilotCampusV22


class UniPilotCampusV23(UniPilotCampusV22):
    """Opt-in precision/coverage candidate. Production and Campus v2.1 remain untouched."""

    version = "campus-v2.3"

    def __init__(self, model=None, tokenizer=None,
                 router_path: str = "data/campus_v2/router/train.jsonl",
                 adversarial_path: str = "data/campus_v21/router/adversarial-train-1500.jsonl"):
        super().__init__(model, tokenizer, router_path, adversarial_path)
        self.knowledge = CampusKnowledgeRetrieverV23.from_files()
        self.answer_planner = CampusCompletenessPlannerV23()
        self.generalizer = CampusCoverageGeneralizerV23()
        self.conversation_memory = CampusConversationMemoryV22()

    def _manual_tool(self, resolved: dict, question: str, plan, tool_inputs: dict | None) -> dict | None:
        if not plan.need_tool or not self.tools.can_handle(plan.primary_category):
            return None
        tool = self.tools.execute(plan.primary_category, question, resolved["state"], tool_inputs)
        resolved.update(kind="tool", category=plan.primary_category, tool=tool, documents=[])
        resolved["router"]["v23_tool_resolution"] = {
            "router_category": resolved["decision"].primary,
            "planner_category": plan.primary_category,
            "tool_veto_applied": False,
        }
        return resolved

    def _resolve_v23(self, question: str, session_id: str | None, tool_inputs: dict | None,
                     response_mode: str, plan) -> dict:
        # Call the frozen v2.1 resolver directly, then apply v2.3's opt-in capability gate.
        resolved = UniPilotCampusV21._resolve(self, question, session_id, tool_inputs)
        resolved["question"] = question
        resolved["router"]["v23_candidate_categories"] = [
            plan.primary_category, plan.secondary_category,
        ]

        manual_tool = self._manual_tool(resolved, question, plan, tool_inputs)
        if manual_tool is not None:
            return manual_tool
        if resolved.get("kind") == "tool" and not plan.need_tool:
            resolved.pop("tool", None)
            resolved["documents"] = []
            if session_id:
                self.sessions.clear_pending(session_id)
            resolved["router"]["v23_tool_resolution"] = {
                "router_category": resolved["decision"].primary,
                "planner_category": plan.primary_category,
                "tool_veto_applied": True,
            }

        if plan.need_clarification:
            resolved.update(
                kind="clarify", category=plan.primary_category, documents=[],
                text=("今わかる範囲では、期限と困っている対象を分けると次の行動を決められます。"
                      "試験・課題・履修・連絡など、最も近い対象を一つ教えてください。"),
            )
            return resolved

        if (resolved.get("kind") == "faq" and self._high_confidence_faq(resolved)
                and not plan.need_retrieval and len(plan.sub_intents) == 1):
            resolved["category"] = plan.primary_category
            return resolved

        state_university = resolved["state"].get("university")
        personal_university = any(cue in question for cue in self.PERSONAL_UNIVERSITY_CUES)
        explicit_university = next(iter(re.findall(
            r"[ぁ-んァ-ヶー一-龥々A-Za-z0-9・]{2,24}大学", question,
        )), None)
        university = state_university or explicit_university
        if personal_university and not university:
            resolved.update(
                kind="safety", category="university_policy", documents=[],
                text=("大学名・対象年度が分からないため、回数・期限・可否は断定しません。"
                      "一般的には、学生便覧、履修要項、シラバス、LMSの順に確認し、"
                      "資料名と表示内容を添えて担当窓口へ問い合わせてください。"),
            )
            return resolved

        documents, retrieval_meta = self.knowledge.search(
            question,
            plan.primary_category,
            secondary_category=plan.secondary_category,
            intent=plan.intent,
            university=university,
            top_k=5,
            response_mode=response_mode,
            strategy="category_aware_hybrid",
            confidence_policy="precision",
        )
        resolved["v22_retrieval"] = retrieval_meta
        resolved["v23_retrieval"] = retrieval_meta
        accepted = CONFIDENCE_ORDER[retrieval_meta["confidence"]] >= CONFIDENCE_ORDER["MEDIUM"]
        if not documents or not accepted:
            resolved.update(
                kind="safe", category=plan.primary_category, documents=[],
                text=self._generic_answer(plan.primary_category),
            )
            resolved["fallback_used"] = True
            return resolved

        text, grounding = self.knowledge_composer.compose_grounded(
            question, plan.primary_category, documents, response_mode,
        )
        resolved.update(
            kind="official",
            category=plan.primary_category,
            documents=documents,
            text=text,
            v22_grounding=grounding,
            v23_grounding=grounding,
            original_route=resolved.get("kind"),
        )
        return resolved

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
        mode = {"simple": "short", "normal": "normal", "complex": "detailed"}[plan.answer_depth]
        resolved = self._resolve_v23(effective_question, session_id, tool_inputs, mode, plan)
        resolved["question"] = question
        result = self._result_v22(resolved, mode)
        result["pipeline"] = self.version
        result["knowledge_version"] = "campus-v2.3"
        route_actions = {
            "tool": "TOOL", "rag": "RAG", "faq": "FAQ", "clarify": "CLARIFY",
            "safe": "GENERAL_SAFE", "safety": "GENERAL_SAFE",
        }
        result["route_action"] = route_actions.get(result["route"], result.get("route_action"))
        result["executed_action"] = result["route_action"]
        if prior:
            result["followup_of"] = effective_question

        improved = self.generalizer.improve_v23(question, result["text"], plan, result)
        result["text"] = improved.text
        result["raw_text"] = improved.text
        result["answer_depth"] = plan.answer_depth
        result["answer_coverage"] = improved.coverage
        result["coverage_before"] = improved.coverage_before
        result["revision_count"] = improved.revision_count
        result["specificity_repaired"] = improved.specificity_repaired
        result["planner_hidden"] = True
        result["quality_checks"] = {
            "coverage": improved.coverage,
            "character_count": len(improved.text),
            "target_min": V23_DEPTH_LIMITS[plan.answer_depth][0],
            "target_max": V23_DEPTH_LIMITS[plan.answer_depth][1],
            "length_ok": V23_DEPTH_LIMITS[plan.answer_depth][0] <= len(improved.text)
                         <= V23_DEPTH_LIMITS[plan.answer_depth][1],
            "specific_not_generic": not self.generalizer._abstract_only(improved.text),
        }
        if improved.card and all(item.get("kind") != improved.card["kind"] for item in result["cards"]):
            result["cards"].append(improved.card)
        retrieval_meta = resolved.get("v23_retrieval")
        if retrieval_meta:
            result["retrieval_confidence"] = retrieval_meta["confidence"]
            result["retrieval_accepted"] = retrieval_meta["accepted"]
            result["retrieval_meta"] = retrieval_meta
        document_quality = {row["id"]: row.get("knowledge_quality") for row in resolved.get("documents", [])}
        for row in result.get("retrieval", []):
            row["knowledge_quality"] = document_quality.get(row["id"])

        source_urls = [
            item.get("url") or item.get("source_url")
            for item in (result.get("sources") or result.get("retrieval") or [])
            if item.get("url") or item.get("source_url")
        ]
        result["validator"] = self.validator.validate(
            question,
            improved.text,
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
        result = self.answer(question, max_new_tokens, temperature, top_k, top_p, repetition_penalty,
                             response_mode, session_id, tool_inputs)
        yield {
            **result["generation_metrics"],
            "text": result["text"], "phase": "complete", "pipeline": self.version,
            "route": result["route"], "category": result["category"], "cards": result["cards"],
            "sources": result.get("sources", []), "response_mode": result["response_mode"],
            "answer_depth": result["answer_depth"], "answer_coverage": result["answer_coverage"],
            "revision_count": result["revision_count"], "validator": result["validator"],
            "session_state": result["session_state"],
        }
