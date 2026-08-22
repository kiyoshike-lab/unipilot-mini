from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from pipeline.classifier import BM25CategoryClassifier
from pipeline.v07 import V07Pipeline, load_jsonl
from pipeline.validator import AnswerValidator
from tokenizer.tokenizer import BPETokenizer


def jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_v07_dataset_counts_and_required_knowledge_fields():
    faq = jsonl("data/v07/faq/faq.jsonl")
    direct = sum((jsonl(f"data/v07/training/{split}.jsonl") for split in ("train", "validation", "test")), [])
    corrected = jsonl("data/v07/corrected/corrected.jsonl")
    knowledge = jsonl("data/v07/knowledge/documents.jsonl")
    required = {"id", "title", "text", "category", "source", "source_url", "license", "retrieved_at"}
    assert len(faq) == 420
    assert len([row for row in direct if row["source_type"] == "project_authored_direct_answer"]) == 2100
    assert len(corrected) == 840
    assert set(Counter(row["bad_reason"] for row in corrected).values()) == {140}
    assert all(required.issubset(row) for row in knowledge)


def test_v07_bm25_classifier_handles_required_examples():
    classifier = BM25CategoryClassifier(load_jsonl("data/v07/classifier/train.jsonl"))
    assert classifier.predict("教授に欠席メールを送りたい")[0] == "professor_email"
    assert classifier.predict("単位を落としそう")[0] == "credit"
    assert classifier.predict("GPAって何？")[0] == "gpa"


def test_v07_fast_pipeline_selects_grounded_answer_without_fallback():
    pipeline = V07Pipeline(None, BPETokenizer.load("tokenizer/vocab-v02-512.json"))
    result = pipeline.answer("GPAって何？")
    assert result["category"] == "gpa"
    assert "成績" in result["text"]
    assert result["grounded_selected"] is True
    assert result["fallback_used"] is False
    assert result["model_generation_skipped"] is True


def test_v07_validator_rejects_invented_subject_and_policy():
    result = AnswerValidator().validate("単位を落としそう", "経済学ならどの大学でも必ず認められます。", "credit")
    assert not result.valid
    assert any(issue.startswith("invented_subject") for issue in result.issues)
    assert "university_specific_hallucination" in result.issues


def test_v07_prompt_contains_bounded_context():
    tokenizer = BPETokenizer.load("tokenizer/vocab-v02-512.json")
    model = UniPilotTransformer(ModelConfig(vocab_size=tokenizer.vocab_size, context_length=256,
                                             embedding_dim=16, n_layers=1, n_heads=4, ffn_dim=32, dropout=0.0))
    pipeline = V07Pipeline(model, tokenizer)
    category = pipeline.classifier.predict("GPAって何？")[0]
    documents = pipeline.retriever.retrieve("GPAって何？", category, 1)
    prompt, context_tokens = pipeline.build_prompt("GPAって何？", documents)
    assert "<CONTEXT>" in prompt and "<USER>" in prompt and context_tokens > 0
    assert len(tokenizer.encode(prompt)) <= model.config.context_length - 48
