from __future__ import annotations

import json
from pathlib import Path
import time

from inference.generate import generate_text
from pipeline.categories import LENGTH_POLICY
from pipeline.classifier import BM25CategoryClassifier
from pipeline.rag import KnowledgeRetriever
from pipeline.validator import AnswerValidator
from training.dataset_v03 import SYSTEM_TEXT


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


class V07Pipeline:
    version = "v0.7"

    def __init__(self, model, tokenizer, classifier_path: str = "data/v07/classifier/train.jsonl",
                 knowledge_path: str = "data/v07/knowledge/documents.jsonl", top_k: int = 1):
        self.model = model
        self.tokenizer = tokenizer
        examples = load_jsonl(classifier_path)
        self.classifier = BM25CategoryClassifier(examples)
        self.retriever = KnowledgeRetriever.from_jsonl(knowledge_path)
        self.validator = AnswerValidator()
        self.top_k = top_k

    def build_prompt(self, question: str, documents: list[dict], reserve_answer_tokens: int = 48) -> tuple[str, int]:
        fixed = f"<BOS><SYSTEM>\n{SYSTEM_TEXT}\n<CONTEXT>\n<USER>\n{question}\n<ASSISTANT>\n"
        budget = max(16, self.model.config.context_length - len(self.tokenizer.encode(fixed)) - reserve_answer_tokens)
        pieces, used = [], 0
        for document in documents:
            header = f"[{document['title']}]\n"
            text = document.get("answer") or document["text"]
            candidate = header + text + "\n"
            ids = self.tokenizer.encode(candidate)
            remaining = budget - used
            if remaining <= 0:
                break
            if len(ids) > remaining:
                candidate = self.tokenizer.decode(ids[:remaining], skip_special=True)
                ids = ids[:remaining]
            pieces.append(candidate)
            used += len(ids)
        context = "".join(pieces)
        prompt = f"<BOS><SYSTEM>\n{SYSTEM_TEXT}\n<CONTEXT>\n{context}<USER>\n{question}\n<ASSISTANT>\n"
        return prompt, used

    def answer(self, question: str, max_new_tokens: int = 100, temperature: float = 0.0, top_k: int = 40,
               top_p: float = 0.9, repetition_penalty: float = 1.1, candidates: int = 1,
               use_validator: bool = True, use_retrieval: bool = True, force_model: bool = False) -> dict:
        started = time.perf_counter()
        category, category_confidence, _ = self.classifier.predict(question)
        classified_at = time.perf_counter()
        documents = self.retriever.retrieve(question, category, self.top_k) if use_retrieval else []
        retrieved_at = time.perf_counter()
        grounded = self.retriever.grounded_answer(documents, category)
        length = LENGTH_POLICY.get(category, "normal")
        caps = {"short": 48, "normal": 80, "detailed": 112}
        generation_cap = min(max_new_tokens, caps[length])
        generated = []
        # A reviewed FAQ is already a final answer, not raw reference prose. It
        # can be validated and selected without paying generation latency. Raw
        # public knowledge or a retrieval miss still goes through the model.
        should_generate = self.model is not None and (force_model or not use_validator or grounded is None)
        if should_generate:
            prompt, context_tokens = self.build_prompt(question, documents)
            for _ in range(max(1, candidates)):
                text, metrics = generate_text(self.model, self.tokenizer, prompt, generation_cap, temperature,
                                              top_k, top_p, repetition_penalty)
                validation = self.validator.validate(question, text, category, grounded)
                generated.append({"text": text, "metrics": metrics, "validation": validation.to_dict()})
        else:
            context_tokens = 0
        if generated:
            model_selected = max(generated, key=lambda row: row["validation"]["score"])
            raw_text = model_selected["text"]
            raw_validation = self.validator.validate(question, raw_text, category, grounded)
        else:
            raw_text = ""
            raw_validation = self.validator.validate(question, raw_text, category, grounded)
            model_selected = {"text": "", "metrics": {"tokens": 0, "seconds": 0.0, "tokens_per_sec": 0.0,
                                                          "eos_reached": False},
                              "validation": raw_validation.to_dict()}
        selection_pool = list(generated)
        if use_validator and grounded:
            grounded_validation = self.validator.validate(question, grounded, category, grounded)
            selection_pool.append({"text": grounded, "metrics": model_selected["metrics"],
                                   "validation": grounded_validation.to_dict(), "source": "grounded_faq"})
        selected = max(selection_pool, key=lambda row: row["validation"]["score"]) if selection_pool else model_selected
        selected_source = selected.get("source", "model")
        selected_validation = self.validator.validate(question, selected["text"], category, grounded)
        fallback_used = use_validator and not selected_validation.valid
        final = self.validator.fallback(category, grounded) if fallback_used else (selected["text"] if use_validator else raw_text)
        final_validation = self.validator.validate(question, final, category, grounded)
        elapsed = time.perf_counter() - started
        return {
            "text": final, "raw_text": raw_text, "category": category, "category_confidence": category_confidence,
            "retrieval": [{"id": row["id"], "title": row["title"], "category": row["category"],
                           "score": row["retrieval_score"], "source": row["source"], "source_url": row["source_url"]}
                          for row in documents],
            "context_tokens": context_tokens, "length_policy": length, "validator": final_validation.to_dict(),
            "raw_validator": raw_validation.to_dict(), "fallback_used": fallback_used,
            "grounded_selected": selected_source == "grounded_faq", "selected_source": selected_source,
            "candidate_count": max(1, candidates) if should_generate else 0,
            "model_generation_skipped": not should_generate, "generation_metrics": model_selected["metrics"],
            "timing": {"classification_seconds": classified_at - started,
                       "retrieval_seconds": retrieved_at - classified_at, "total_seconds": elapsed},
            "pipeline": self.version, "external_ai_api": "OFF",
        }
