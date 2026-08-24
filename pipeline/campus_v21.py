from __future__ import annotations

from pipeline.campus_categories import CAMPUS_LABELS
from pipeline.campus_composer_v21 import CampusAnswerComposerV21
from pipeline.campus_retrieval import CampusPublicKnowledge, CampusUniversityKnowledge, load_jsonl
from pipeline.campus_retrieval_v21 import CampusFAQRetrieverV21
from pipeline.campus_router_v21 import CampusRouterV21
from pipeline.campus_state_v2 import CampusSessionStoreV2
from pipeline.campus_tools import ToolResult, card
from pipeline.campus_tools_v2 import CampusToolEngineV2
from pipeline.campus_v2 import UniPilotCampusV2
from pipeline.campus_validator_v2 import CampusValidatorV2


class UniPilotCampusV21(UniPilotCampusV2):
    version = "campus-v2.1"

    def __init__(self, model=None, tokenizer=None,
                 router_path: str = "data/campus_v2/router/train.jsonl",
                 adversarial_path: str = "data/campus_v21/router/adversarial-train-1500.jsonl"):
        self.model = model
        self.tokenizer = tokenizer
        examples = load_jsonl(router_path) + load_jsonl(adversarial_path)
        self.router = CampusRouterV21(examples)
        self.faq = CampusFAQRetrieverV21.from_jsonl()
        self.public_knowledge = CampusPublicKnowledge.from_jsonl()
        self.universities = CampusUniversityKnowledge.from_root()
        self.tools = CampusToolEngineV2()
        self.validator = CampusValidatorV2()
        self.sessions = CampusSessionStoreV2()
        self.composer = CampusAnswerComposerV21()

    def _faq_search(self, question: str, category: str, top_k: int, confidence_band: str = "high") -> list[dict]:
        return self.faq.search(question, category, top_k, confidence_band)

    def _resolve(self, question: str, session_id: str | None, tool_inputs: dict | None) -> dict:
        resolved = super()._resolve(question, session_id, tool_inputs)
        if resolved["kind"] == "faq" and not resolved.get("documents"):
            resolved.update(kind="safe", text=self.composer.safe_no_match())
        return resolved

    def _clarification_cards(self, resolved: dict) -> list[dict]:
        decision = resolved["decision"]
        options = []
        for category in decision.top2:
            if category == "general":
                continue
            options.append({"category": category, "label": CAMPUS_LABELS.get(category, category),
                            "prompt": f"{CAMPUS_LABELS.get(category, category)}について相談したい"})
        defaults = (("exam", "試験"), ("assignment", "課題"), ("credit", "単位"),
                    ("professor_email", "教授への連絡"))
        for category, label in defaults:
            if len(options) >= 4:
                break
            if category not in {item["category"] for item in options}:
                options.append({"category": category, "label": label, "prompt": f"{label}について相談したい"})
        return [card("clarify", "相談内容を確認", "近い項目を選ぶか、対象を一言追加してください。",
                     data={"options": options[:4]})]

    def _static_result(self, resolved: dict) -> dict:
        tool: ToolResult | None = resolved.get("tool")
        if tool:
            resolved["tool"] = self.composer.compose_tool(resolved.get("question", ""), tool)
        elif resolved["kind"] == "faq":
            resolved["text"] = self.composer.compose_faq(
                resolved.get("question", ""), resolved["text"],
                multi_intent=len(resolved["decision"].intents) > 1,
            )
        result = super()._static_result(resolved)
        result["response_mode"] = self.composer.response_mode(
            resolved.get("question", ""), tool=resolved.get("tool"),
            multi_intent=len(resolved["decision"].intents) > 1,
        )
        if resolved["kind"] == "clarify":
            result["cards"] = self._clarification_cards(resolved)
        return result
