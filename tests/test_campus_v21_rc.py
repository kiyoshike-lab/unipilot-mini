from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

import api.main as api_main


RC = "0dc18789be28613a8c651cfefde63fb659ee2019"


def read(path: str) -> dict | list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_rc_manifest_freezes_answer_logic_checkpoint_and_tokenizer():
    manifest = read("evaluation/campus-v21-rc-manifest.json")
    assert manifest["rc_source_commit"] == RC
    assert manifest["answer_logic_mutation_policy"] == "prohibited during human evaluation"
    for relative, expected in manifest["logic_sha256"].items():
        assert hashlib.sha256(Path(relative).read_bytes()).hexdigest() == expected
    assert manifest["checkpoint"]["model"] == "v0.4" and manifest["checkpoint"]["step"] == 2000
    assert manifest["tokenizer"]["vocab"] == 512
    assert manifest["external_ai_api"] == "OFF" and manifest["production_changed"] is False


def test_human_100_is_balanced_unique_and_has_manual_comparison_schema():
    rows = read("evaluation/human-comparison-campus-v21.json")
    audit = read("evaluation/campus-v21-human-audit.json")
    assert len(rows) == 100 and audit["unique_semantic_questions"] == 100
    assert Counter(row["evaluation_bucket"] for row in rows) == {
        "easy": 25, "medium": 25, "hard": 25, "compound_ambiguous": 25,
    }
    assert all(audit["required_topics_present"].values())
    assert audit["compound_present"] and audit["ambiguous_present"] and audit["colloquial_present"]
    assert not audit["easy_only"]
    for row in rows:
        assert row["rc_source_commit"] == RC and row["evaluation_status"] == "PENDING_MANUAL_REVIEW"
        assert row["chatgpt_answer"] == row["gemini_answer"] == ""
        assert set(row["scores"]) == {"correctness", "relevance", "actionable", "naturalness", "would_use_again"}
        assert set(row["pairwise"]) == {"chatgpt", "gemini"}
        assert set(row["ux"]) == {"tool_card", "copy_action", "input_flow", "clarification", "streaming", "latency"}
        assert all(value is not None for value in row["automatic_evaluation"].values())


def test_known_issue_queue_has_exact_frozen_rc_candidates():
    payload = read("evaluation/campus-v21-rc-known-issues.json")
    assert payload["rc_source_commit"] == RC and payload["answer_logic_mutation"] == "prohibited"
    assert payload["counts"] == {"hallucination": 13, "router": 3, "retrieval": 7, "total": 23}
    assert all(item["human_review"]["status"] == "pending"
               for group in payload["groups"].values() for item in group)
    assert all(item["gold_category"] == "toeic_plan" and item["predicted_category"] == "study_plan"
               for item in payload["groups"]["router"])
    assert Counter(item["automatic_flag"] for item in payload["groups"]["retrieval"]) == {
        "WRONG_FAQ": 5, "FALSE_NO_MATCH": 2,
    }


def test_e2e_and_route_speed_results_are_complete():
    e2e = read("evaluation/campus-v21-rc-e2e.json")
    routes = read("evaluation/campus-v21-rc-route-speed.json")
    assert e2e["scenarios"] == e2e["passed"] == 25 and e2e["success_rate"] == 1.0
    assert all(record["passed"] for record in e2e["records"])
    assert sum(row["count"] for row in routes["canonical_route_mix"].values()) == 100
    assert set(routes["canonical_route_mix"]) == {"TOOL", "FAQ", "RAG", "MODEL", "CLARIFY"}
    assert routes["local_latency"]["under_one_second_count"] == 100
    assert routes["local_latency"]["under_one_second_share"] == 1.0
    assert routes["actual_model_generation"] == {"count": 0, "share": 0.0, "non_model_share": 1.0}
    assert routes["planned_model_assisted_actions"]["count"] == 19
    assert "not Render" in routes["measurement_scope"]


def test_human_gate_stays_pending_without_invented_scores():
    report = read("evaluation/campus-v21-rc-human-report.json")
    assert report["automatic_gate"] == "PASS"
    assert report["human_evaluation"]["completed"] == 0
    assert all(value is None for value in report["human_evaluation"]["averages_0_to_5"].values())
    assert report["automatic_vs_human"]["automatic_correctness_percent"] == 99.2
    assert report["automatic_vs_human"]["human_correctness_percent"] is None
    assert report["human_production_gate"]["status"] == "PENDING"
    assert report["production_promotion_recommended"] == report["beta_start_recommended"] == "NO"
    assert report["standard_50m_needed"] == "NO; remains stopped"
    assert report["production_changed"] is False and report["push_or_deploy_performed"] is False


def test_known_issue_api_persists_only_human_review(tmp_path, monkeypatch):
    path = tmp_path / "known.json"
    path.write_text(json.dumps({"counts": {"hallucination": 1, "router": 0, "retrieval": 0, "total": 1},
                                "groups": {"hallucination": [{"id": "issue-1", "question": "q",
                                    "human_review": {"status": "pending", "severity": "unreviewed",
                                                     "blocks_production": False, "notes": ""}}],
                                           "router": [], "retrieval": []}}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(api_main, "HUMAN_CAMPUS_V21_KNOWN_ISSUES", path)
    monkeypatch.setattr(api_main, "load_runtime", lambda checkpoint=None: None)
    with TestClient(api_main.app) as client:
        response = client.post("/human-eval/campus-v21/known-issues", json={
            "item_id": "issue-1", "group": "hallucination", "status": "confirmed",
            "severity": "high", "blocks_production": True, "notes": "human confirmed",
        })
    assert response.status_code == 200
    saved = json.loads(path.read_text(encoding="utf-8"))["groups"]["hallucination"][0]
    assert saved["question"] == "q"
    assert saved["human_review"] == {"status": "confirmed", "severity": "high",
                                      "blocks_production": True, "notes": "human confirmed"}
