from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

from fastapi.testclient import TestClient

import api.main as api_main
from model.config import ModelConfig
from model.transformer import UniPilotTransformer
from pipeline.campus_retrieval import CampusPublicKnowledge
from pipeline.campus_v1 import UniPilotCampusV1
from tokenizer.tokenizer import BPETokenizer


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_campus_dataset_counts_schema_and_blind_independence():
    faq = load_jsonl("data/campus_v1/faq/faq.jsonl")
    blind = json.loads(Path("data/campus_v1/blind/evaluation.json").read_text(encoding="utf-8"))
    required = {"id", "category", "question", "answer", "keywords", "source", "university_specific", "needs_confirmation"}
    assert len(faq) == 1000 and all(required.issubset(row) for row in faq)
    assert len(blind) == 1000
    assert Counter(row["difficulty"] for row in blind) == {"easy": 300, "medium": 300, "hard": 250, "compound": 150}
    assert not ({row["question"].replace(" ", "") for row in faq} & {row["prompt"].replace(" ", "") for row in blind})


def test_campus_public_rag_keeps_only_source_attributed_records():
    knowledge = CampusPublicKnowledge.from_jsonl()
    assert len(knowledge.rows) == 24
    assert all(row["source_url"] and row["license"] for row in knowledge.rows)


def test_campus_gpa_and_required_score_are_deterministic():
    pipeline = UniPilotCampusV1()
    gpa = pipeline.answer("A 2単位、B 2単位、S 1単位でGPA計算")
    score = pipeline.answer("現在40点、合格60点、残り評価30%で何点必要")
    assert gpa["tool"] == "gpa" and gpa["calculation"]["gpa"] == 2.8
    assert score["tool"] == "grade_simulator"
    assert round(score["calculation"]["required_average_percent"], 2) == 66.67


def test_campus_email_returns_copyable_complete_card():
    result = UniPilotCampusV1().answer("教授に欠席メール送りたい")
    assert result["tool"] == "absence_email" and result["route"] == "tool"
    assert "件名" in result["text"] and result["cards"][0]["copy_text"]


def test_campus_university_policy_never_guesses_without_official_record():
    result = UniPilotCampusV1().answer("うちの大学って追試ある？")
    assert result["route"] == "safety" and "断定できません" in result["text"]
    assert "学生便覧" in result["text"] and not result["validator"]["issues"]


def test_campus_session_continues_pending_study_plan():
    pipeline = UniPilotCampusV1()
    first = pipeline.answer("明日テスト", session_id="session-test")
    second = pipeline.answer("数学で3時間", session_id="session-test")
    assert first["tool"] == "study_plan" and first["missing_fields"]
    assert second["tool"] == "study_plan" and second["calculation"]["days"] == 1
    assert second["calculation"]["subject"] == "数学"


def test_campus_static_routes_return_instant_generation_compatible_metrics():
    result = UniPilotCampusV1().answer("GPA計算したい")
    assert result["pipeline"] == "campus-v1" and result["generation_metrics"]["eos_reached"]
    assert result["generation_metrics"]["tokens"] == 0 and result["external_ai_api"] == "OFF"


def test_campus_model_route_streams_incremental_tokens_without_double_generation(monkeypatch):
    tokenizer = BPETokenizer()
    model = UniPilotTransformer(ModelConfig(vocab_size=tokenizer.vocab_size, context_length=64,
                                             embedding_dim=16, n_layers=1, n_heads=4, ffn_dim=32, dropout=0.0)).eval()
    pipeline = UniPilotCampusV1(model, tokenizer)
    monkeypatch.setattr(pipeline, "_resolve", lambda *args: {
        "kind": "model", "category": "general", "confidence": .5, "router": {}, "state": {},
        "documents": [], "started": time.perf_counter(),
    })
    snapshots = list(pipeline.iter_answer("確認", max_new_tokens=3, temperature=0))
    generating = [row for row in snapshots if row["phase"] == "generating"]
    assert generating and [row["tokens"] for row in generating] == list(range(1, len(generating) + 1))
    assert all(row["pipeline"] == "campus-v1" and row["kv_cache"] for row in generating)


def test_campus_api_keeps_chat_and_stream_contract(monkeypatch):
    previous = dict(api_main.runtime)
    monkeypatch.setattr(api_main, "load_runtime", lambda checkpoint=None: None)
    api_main.runtime.update(model=object(), tokenizer=None, device="cpu", checkpoint="test", payload={},
                            pipeline=UniPilotCampusV1())
    request = {"prompt": "教授に欠席メール送りたい", "session_id": "api-campus", "max_new_tokens": 3}
    try:
        with TestClient(api_main.app) as client:
            regular = client.post("/chat", json=request)
            streamed = client.post("/chat/stream", json=request)
        assert regular.status_code == 200 and regular.json()["cards"][0]["kind"] == "email"
        snapshots = [json.loads(line) for line in streamed.text.splitlines()]
        assert snapshots and snapshots[-1]["pipeline"] == "campus-v1" and snapshots[-1]["cards"]
    finally:
        api_main.runtime.clear(); api_main.runtime.update(previous)


def test_campus_human_score_endpoint_persists_zero_to_five(tmp_path, monkeypatch):
    path = tmp_path / "campus-human.json"
    path.write_text(json.dumps([{"id": "campus-manual-000", "scores": {"campus": None}, "winners": {},
                                 "chatgpt_answer": None, "gemini_answer": None, "notes": ""}], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(api_main, "HUMAN_CAMPUS", path)
    monkeypatch.setattr(api_main, "load_runtime", lambda checkpoint=None: None)
    payload = {"item_id": "campus-manual-000", "campus_score": 5, "correct_winner": "campus",
               "specific_winner": "campus", "usable_winner": "campus", "fast_winner": "campus",
               "student_preference": "campus"}
    with TestClient(api_main.app) as client:
        response = client.post("/human-eval/campus", json=payload)
    saved = json.loads(path.read_text(encoding="utf-8"))[0]
    assert response.status_code == 200 and saved["scores"]["campus"] == 5
