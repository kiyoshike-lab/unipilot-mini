from __future__ import annotations

import json
from pathlib import Path

from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from pipeline.v08 import V08Pipeline
from pipeline.validator_v08 import StandardAnswerValidator
from tokenizer.tokenizer import BPETokenizer


def jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def test_standard_architecture_is_separate_45m_candidate():
    settings = json.loads(Path("configs/unipilot-standard-v08.json").read_text(encoding="utf-8"))
    model = UniPilotTransformer(ModelConfig(**settings["model"]))
    assert model.parameter_count() == 44_920_832
    assert model.config.vocab_size == 1024 and model.config.context_length == 512
    assert model.config.embedding_dim // model.config.n_heads == 64


def test_standard_tokenizer_has_context_token_and_roundtrips():
    tokenizer = BPETokenizer.load("tokenizer/vocab-standard-v08-1024.json")
    text = "<BOS><SYSTEM>\n確認\n<CONTEXT>\n根拠\n<USER>\n質問\n<ASSISTANT>\n回答<EOS>"
    assert tokenizer.vocab_size == 1024
    assert "<CONTEXT>" in tokenizer.special_to_id
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_standard_dataset_counts_and_blind_separation():
    quality = json.loads(Path("evaluation/dataset-quality-v08.json").read_text(encoding="utf-8"))
    assert quality["semantic_scenario_cells"] == 3300
    assert quality["high_quality_instruction_rows"] == 9900
    assert quality["corrected_rows"] == 3300
    assert quality["conversation_rows"] == 6000
    assert quality["compound_rows"] == 1000
    assert quality["blind_questions"] == 528
    assert quality["exact_train_blind_question_overlap"] == 0
    assert quality["exact_knowledge_title_blind_overlap"] == 0


def test_standard_prompt_reserves_answer_inside_context():
    tokenizer = BPETokenizer.load("tokenizer/vocab-standard-v08-1024.json")
    model = UniPilotTransformer(ModelConfig(vocab_size=1024, context_length=512, embedding_dim=16,
                                             n_layers=1, n_heads=4, ffn_dim=32, dropout=0.0))
    pipeline = V08Pipeline(model, tokenizer, top_k=1)
    prepared = pipeline.prepare("教授へ欠席連絡を詳しく作ってください", "detailed")
    prompt_tokens = len(tokenizer.encode(prepared["prompt"]))
    assert "<CONTEXT>" in prepared["prompt"]
    assert prompt_tokens + prepared["max_answer_tokens"] <= model.config.context_length
    assert 200 <= prepared["max_answer_tokens"] <= 400


def test_standard_length_modes_and_validator():
    assert V08Pipeline.choose_mode("GPAとは？") == "short"
    assert V08Pipeline.choose_mode("試験と課題の優先順位を詳しく比較して") == "detailed"
    result = StandardAnswerValidator().validate("履修について教えて", "どの大学でも2026年9月1日までに必ず単位が認められます。")
    assert not result.valid
    assert "university_specific_hallucination" in result.issues
    assert "unsupported_date_or_fee" in result.issues


def test_standard_human_form_is_unscored():
    path = Path("evaluation/results-standard-v08-a100-blind-human-100.json")
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert len(rows) == 100
    assert all(row["score_0_to_5"] is None and row["blind"] for row in rows)
