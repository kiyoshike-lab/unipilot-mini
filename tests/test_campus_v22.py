from __future__ import annotations

import json
from pathlib import Path

from pipeline.campus_retrieval import load_jsonl
from pipeline.campus_retrieval_v22 import CampusKnowledgeRetrieverV22
from pipeline.campus_v21 import UniPilotCampusV21
from pipeline.campus_v22 import UniPilotCampusV22


REQUIRED = {
    "id", "title", "text", "category", "sub_category", "source", "source_url",
    "retrieved_at", "license", "publisher", "university_name", "university_specific",
    "last_verified_at",
}


def knowledge_rows() -> list[dict]:
    rows = []
    for path in Path("data/campus_v22/knowledge").glob("*.jsonl"):
        rows.extend(load_jsonl(path))
    return rows


def test_v22_knowledge_has_provenance_and_no_unknown_license():
    rows = knowledge_rows()
    assert len(rows) >= 500
    assert all(REQUIRED <= row.keys() for row in rows)
    assert all(row["license"] and row["source_url"].startswith("https://") for row in rows)
    assert all(row.get("source_type") != "unknown" for row in rows)
    assert any(row["source_type"] == "official_government" for row in rows)
    assert any(row["source_type"] == "wikipedia" for row in rows)


def test_university_records_require_exact_session_match():
    retriever = CampusKnowledgeRetrieverV22.from_files()
    university_rows = [row for row in retriever.rows if row.get("university_specific")]
    if not university_rows:
        return
    target = university_rows[0]
    without, _ = retriever.search(target["title"], target["category"], university=None, threshold=0.0)
    wrong, _ = retriever.search(target["title"], target["category"], university="別の大学", threshold=0.0)
    matching, _ = retriever.search(target["title"], target["category"], university=target["university_name"], threshold=0.0)
    assert target["id"] not in {row["id"] for row in without}
    assert target["id"] not in {row["id"] for row in wrong}
    assert any(row.get("university_name") == target["university_name"] for row in matching)


def test_v22_preserves_v21_deterministic_tool_output():
    question = "GPAを計算して。A 2単位、B 2単位"
    v21 = UniPilotCampusV21()
    v22 = UniPilotCampusV22()
    old = v21.answer(question, tool_inputs={"courses": [
        {"name": "科目A", "grade": "A", "credits": 2},
        {"name": "科目B", "grade": "B", "credits": 2},
    ]})
    new = v22.answer(question, tool_inputs={"courses": [
        {"name": "科目A", "grade": "A", "credits": 2},
        {"name": "科目B", "grade": "B", "credits": 2},
    ]})
    assert old["route"] == new["route"] == "tool"
    assert old["text"] == new["text"]
    assert old["calculation"] == new["calculation"]


def test_v22_grounded_answer_sources_modes_and_followup():
    pipeline = UniPilotCampusV22()
    wikipedia = next(row for row in pipeline.knowledge.rows if row["source_type"] == "wikipedia")
    question = f"一般教養として『{wikipedia['title']}』について根拠付きで教えてください"
    normal = pipeline.answer(question, session_id="v22-followup-test")
    assert normal["pipeline"] == "campus-v2.2"
    assert normal["response_mode"] == "normal"
    assert normal["route"] == "rag"
    assert normal["sources"]
    assert any(card["kind"] == "sources" for card in normal["cards"])
    detailed = pipeline.answer("もっと詳しく", session_id="v22-followup-test")
    assert detailed["response_mode"] == "detailed"
    assert detailed["followup_of"] == question
    assert detailed["route"] == "rag"


def test_v22_benchmark_sizes_and_human_gate():
    benchmark = load_jsonl("data/campus_v22/benchmarks/knowledge-1000.jsonl")
    hallucination = load_jsonl("data/campus_v22/benchmarks/hallucination-500.jsonl")
    human = load_jsonl("data/campus_v22/benchmarks/human-knowledge-100.jsonl")
    assert len(benchmark) == 1000
    assert len(hallucination) == 500
    assert len(human) == 100
    assert len({row["question"] for row in benchmark}) == 1000
    assert all(row["evaluation_status"] == "PENDING_HUMAN_REVIEW" for row in human)
    assert all(all(value is None for value in row["scores"].values()) for row in human)


def test_v22_report_keeps_human_gate_closed():
    report = json.loads(Path("evaluation/campus-v22-results.json").read_text(encoding="utf-8"))
    assert report["human_gate"]["status"] == "PENDING"
    assert report["human_gate"]["production_promotion_allowed"] is False
