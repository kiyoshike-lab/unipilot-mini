from __future__ import annotations

import time

from inference.generate import generate_text, iter_generate_text
from pipeline.retrieval_v08 import StandardHybridRetriever
from pipeline.validator_v08 import StandardAnswerValidator


SYSTEM_V08 = (
    "あなたは大学生活支援に特化した完全ローカルのUniPilot Standardです。"
    "検索文脈に根拠がある場合はそれを使い、不足する情報は推測せず確認方法を案内します。"
)
MODE_LIMITS = {"short": 80, "normal": 200, "detailed": 400}


class V08Pipeline:
    version = "v0.8"

    def __init__(self, model, tokenizer, retrieval_method: str = "tfidf", top_k: int = 3):
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = StandardHybridRetriever.from_files()
        self.validator = StandardAnswerValidator()
        self.retrieval_method = retrieval_method
        self.top_k = top_k

    @staticmethod
    def choose_mode(question: str, requested: str = "auto") -> str:
        if requested in MODE_LIMITS:
            return requested
        if len(question) <= 18 and not any(word in question for word in ("作って", "計画", "比較", "どうすれば")):
            return "short"
        if any(word in question for word in ("計画", "メール", "比較", "複数", "両方", "詳しく", "優先順位")):
            return "detailed"
        return "normal"

    def build_prompt(self, question: str, documents: list[dict], mode: str,
                     answer_reserve: int | None = None) -> tuple[str, str, int, int]:
        fixed = f"<BOS><SYSTEM>\n{SYSTEM_V08}\n<CONTEXT>\n<USER>\n{question}\n回答モード：{mode}\n<ASSISTANT>\n"
        fixed_tokens = len(self.tokenizer.encode(fixed))
        maximum_answer = max(1, self.model.config.context_length - fixed_tokens)
        desired_answer = answer_reserve or MODE_LIMITS[mode]
        # Keep a useful context window where possible, while guaranteeing that
        # prompt + generated answer never exceeds the configured context.
        reserve = min(desired_answer, max(1, maximum_answer - min(64, maximum_answer // 3)))
        budget = max(0, self.model.config.context_length - fixed_tokens - reserve)
        pieces, used = [], 0
        for document in documents:
            piece = f"[{document['title']}]\n{document['text']}\n"
            ids = self.tokenizer.encode(piece)
            remaining = budget - used
            if remaining <= 0:
                break
            if len(ids) > remaining:
                ids = ids[:remaining]
                piece = self.tokenizer.decode(ids, skip_special=True)
            pieces.append(piece)
            used += len(ids)
        context = "".join(pieces)
        prompt = f"<BOS><SYSTEM>\n{SYSTEM_V08}\n<CONTEXT>\n{context}<USER>\n{question}\n回答モード：{mode}\n<ASSISTANT>\n"
        return prompt, context, used, reserve

    def prepare(self, question: str, response_mode: str = "auto") -> dict:
        started = time.perf_counter()
        category, confidence = self.retriever.predict_category(question)
        documents = self.retriever.search(question, self.top_k, self.retrieval_method)
        mode = self.choose_mode(question, response_mode)
        prompt, context, context_tokens, max_answer_tokens = self.build_prompt(question, documents, mode)
        return {"question": question, "category": category, "category_confidence": confidence,
                "documents": documents, "mode": mode, "prompt": prompt, "context": context,
                "context_tokens": context_tokens, "max_answer_tokens": max_answer_tokens,
                "prepare_seconds": time.perf_counter() - started}

    def answer(self, question: str, max_new_tokens: int = 200, temperature: float = 0.0, top_k: int = 40,
               top_p: float = 0.9, repetition_penalty: float = 1.1, response_mode: str = "auto") -> dict:
        prepared = self.prepare(question, response_mode)
        cap = min(max_new_tokens, MODE_LIMITS[prepared["mode"]], prepared["max_answer_tokens"])
        text, metrics = generate_text(self.model, self.tokenizer, prepared["prompt"], cap, temperature,
                                      top_k, top_p, repetition_penalty)
        validation = self.validator.validate(question, text, prepared["context"])
        fallback = not validation.valid
        final = self.validator.fallback(prepared["category"]) if fallback else text
        return {
            "text": final, "raw_text": text, "category": prepared["category"],
            "category_confidence": prepared["category_confidence"], "response_mode": prepared["mode"],
            "context_tokens": prepared["context_tokens"], "retrieval": [
                {"id": row["id"], "title": row["title"], "category": row.get("category"),
                 "score": row["retrieval_score"], "source": row.get("source"), "source_url": row.get("source_url")}
                for row in prepared["documents"]],
            "validator": validation.to_dict(), "fallback_used": fallback, "generation_metrics": metrics,
            "timing": {"prepare_seconds": prepared["prepare_seconds"],
                       "total_seconds": prepared["prepare_seconds"] + metrics["seconds"]},
            "pipeline": self.version, "external_ai_api": "OFF",
        }

    def iter_answer(self, question: str, max_new_tokens: int = 200, temperature: float = 0.0, top_k: int = 40,
                    top_p: float = 0.9, repetition_penalty: float = 1.1, response_mode: str = "auto"):
        prepared = self.prepare(question, response_mode)
        cap = min(max_new_tokens, MODE_LIMITS[prepared["mode"]], prepared["max_answer_tokens"])
        last = None
        for snapshot in iter_generate_text(self.model, self.tokenizer, prepared["prompt"], cap, temperature,
                                           top_k, top_p, repetition_penalty):
            last = snapshot
            yield {**snapshot, "pipeline": self.version, "phase": "generating",
                   "category": prepared["category"], "response_mode": prepared["mode"]}
        raw = last["text"] if last else ""
        validation = self.validator.validate(question, raw, prepared["context"])
        if not validation.valid:
            yield {"text": self.validator.fallback(prepared["category"]), "tokens": last["tokens"] if last else 0,
                   "seconds": last["seconds"] if last else prepared["prepare_seconds"], "tokens_per_sec": last["tokens_per_sec"] if last else 0.0,
                   "eos_reached": True, "kv_cache": True, "pipeline": self.version, "phase": "validated_replacement",
                   "category": prepared["category"], "response_mode": prepared["mode"], "fallback_used": True,
                   "validator": validation.to_dict()}
