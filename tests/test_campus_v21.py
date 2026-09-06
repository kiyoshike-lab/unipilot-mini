from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from fastapi.testclient import TestClient

import api.main as api_main
from pipeline.campus_retrieval import load_jsonl
from pipeline.campus_router import normalize
from pipeline.campus_router_v21 import CampusRouterV21, parse_negation_contrast
from pipeline.campus_v21 import UniPilotCampusV21


_PIPELINE = None


def pipeline() -> UniPilotCampusV21:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = UniPilotCampusV21()
    return _PIPELINE


def test_campus_v21_data_boundaries_counts_and_real_distribution():
    manifest = json.loads(Path("data/campus_v21/manifest.json").read_text(encoding="utf-8"))
    assert manifest["adversarial_train"] == 1500
    assert manifest["adversarial_validation"] == 300
    assert manifest["adversarial_types"] == {"NEGATION": 300, "CORRECTION": 300, "CONTRAST": 300,
                                               "CATEGORY_COLLISION": 300, "SHORT_BUT_CLEAR": 300}
    assert manifest["clarification_validation"] == 1200 and manifest["real_student"] == 500
    assert manifest["real_existing_normalized_overlap"] == 0
    assert manifest["existing_adversarial_300_role"].startswith("test-only")
    rows = json.loads(Path("data/campus_v21/real-student/evaluation-500.json").read_text(encoding="utf-8"))
    assert Counter(row["surface_type"] for row in rows) == {
        "very_short": 100, "colloquial": 100, "correction": 100, "normal": 100, "compound": 100,
    }


def test_existing_adversarial_test_does_not_overlap_v21_training():
    train = load_jsonl("data/campus_v21/router/adversarial-train-1500.jsonl")
    test = json.loads(Path("data/campus_v2/adversarial/negation-300.json").read_text(encoding="utf-8"))
    assert not ({normalize(row["question"]) for row in train} &
                {normalize(row.get("question") or row["prompt"]) for row in test})


def test_clarification_gate_uses_validation_only_and_meets_targets():
    config = json.loads(Path("data/campus_v21/router/clarification-config.json").read_text(encoding="utf-8"))
    assert config["validation"] == 1200 and config["search_candidates"] == 192
    assert config["selected_metrics"]["ambiguous_handling"] >= .97
    assert config["selected_metrics"]["unnecessary_clarify"] <= .02
    assert config["selected_metrics"]["constraints_passed"]


def test_ambiguous_questions_clarify_but_short_clear_questions_route():
    ambiguous = pipeline().answer("やばい")
    short_clear = pipeline().answer("単位やばい")
    assert ambiguous["route_action"] == "CLARIFY" and ambiguous["category"] == "general"
    options = ambiguous["cards"][0]["data"]["options"]
    assert 2 <= len(options) <= 4 and all(row["label"] and row["prompt"] for row in options)
    assert short_clear["route_action"] != "CLARIFY" and short_clear["category"] == "credit"


def test_negation_contrast_parser_suppresses_negative_intent():
    parsed = parse_negation_contrast("欠席メールではなく遅刻メールを作りたい")
    assert parsed and parsed["negative_text"] == "欠席メール" and "遅刻メール" in parsed["positive_text"]
    router = CampusRouterV21(load_jsonl("data/campus_v2/router/train.jsonl") +
                             load_jsonl("data/campus_v21/router/adversarial-train-1500.jsonl"))
    decision = router.decide("欠席メールではなく遅刻メールを作りたい")
    assert decision.primary == "lateness_email" and decision.evidence["negation_contrast"]["negative_intent"] == "absence_email"


def test_retrieval_independent_test_meets_targets_and_no_match_is_safe():
    result = json.loads(Path("evaluation/campus-v21-retrieval.json").read_text(encoding="utf-8"))
    test = result["test"]
    assert result["test_is_independent_of_threshold_selection"]
    assert test["recall_at_1"] >= .90 and test["recall_at_3"] >= .95 and test["mrr"] >= .92
    assert test["false_faq_match"] <= .02
    rows = json.loads(Path("data/campus_v21/retrieval/test.json").read_text(encoding="utf-8"))
    no_match = next(row for row in rows if not row["has_match"])
    assert pipeline().faq.search(no_match["query"], no_match["category"], 3, "high") == []
    answer = pipeline().answer("学食の今日限定メニューを今すぐ知りたい")
    assert "推測で返しません" in answer["text"] or answer["route"] == "safety"


def test_v21_benchmark_passes_automatic_gate_but_not_pending_human_gate():
    benchmark = json.loads(Path("evaluation/campus-v21-benchmark.json").read_text(encoding="utf-8"))
    gate = benchmark["production_gate"]
    assert gate["automatic_passed"] and all(gate["automatic_checks"].values())
    assert benchmark["adversarial_validation_300"]["metrics"]["determinate_category_accuracy"] >= .95
    assert not gate["human_passed"] and not gate["passed"]
    assert benchmark["production_changed"] is False and benchmark["external_ai_api"] == "OFF"


def test_campus_v21_human_endpoint_persists_five_axes(tmp_path, monkeypatch):
    path = tmp_path / "campus-v21-human.json"
    path.write_text(json.dumps([{"id": "v21-x", "scores": {"correctness": None},
                                "competitor_scores": {}, "chatgpt_answer": "", "gemini_answer": "",
                                "notes": ""}], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(api_main, "HUMAN_CAMPUS_V21", path)
    monkeypatch.setattr(api_main, "HUMAN_CAMPUS_V21_RESULTS", tmp_path / "campus-v21-human-results.json")
    monkeypatch.setattr(api_main, "HUMAN_CAMPUS_V21_REPORT", tmp_path / "campus-v21-human-report.md")
    monkeypatch.setattr(api_main, "load_runtime", lambda checkpoint=None: None)
    payload = {"item_id": "v21-x", "correctness": 5, "relevance": 5, "actionable": 4,
               "naturalness": 5, "would_use_again": 4, "notes": "manual"}
    with TestClient(api_main.app) as client:
        response = client.post("/human-eval/campus-v21", json=payload)
    assert response.status_code == 200
    saved = json.loads(path.read_text(encoding="utf-8"))[0]
    assert saved["scores"] == {"correctness": 5, "relevance": 5, "actionable": 4,
                               "naturalness": 5, "would_use_again": 4}
    assert saved["issues_reviewed"] is True
    assert saved["evaluation_status"] == "SCORED_MANUALLY"
    assert set(saved["pairwise"]) == {"chatgpt", "gemini"}


def human_row(index: int, score: int = 5, *, reviewed: bool = True) -> dict:
    return {
        "id": f"human-{index:03d}",
        "scores": {"correctness": score, "relevance": score, "actionable": score,
                   "naturalness": score, "would_use_again": score},
        "issues_reviewed": reviewed,
        "issue_flags": {},
        "pairwise": {
            "chatgpt": {axis: "unipilot" for axis in ("correctness", "specificity", "actionability",
                                                         "readability", "would_use")},
            "gemini": {axis: "tie" for axis in ("correctness", "specificity", "actionability",
                                                  "readability", "would_use")},
        },
    }


def test_v21_human_gate_is_pending_until_exactly_100_completed():
    summary = api_main.build_campus_v21_human_summary([human_row(index) for index in range(99)])
    assert summary["status"] == "PENDING"
    assert summary["human_gate"]["status"] == "PENDING"
    assert summary["automated_comparison"] is None
    assert summary["v2_2_priorities"] == []


def test_v21_human_gate_passes_and_compares_automated_only_after_100():
    rows = [human_row(index) for index in range(100)]
    summary = api_main.build_campus_v21_human_summary(rows)
    assert summary["status"] == "COMPLETE"
    assert summary["human_gate"]["status"] == "PASS"
    assert summary["averages_0_to_5"] == {axis: 5.0 for axis in api_main.V21_HUMAN_AXES}
    assert summary["automated_comparison"]["automated_correctness_percent"] == 99.2
    assert summary["automated_comparison"]["human_correctness_percent"] == 100.0
    assert summary["pairwise"]["chatgpt"]["win"] == 500
    assert summary["pairwise"]["gemini"]["tie"] == 500


def test_v21_human_gate_fails_thresholds_and_classifies_errors():
    rows = [human_row(index, score=4) for index in range(100)]
    rows[0]["issue_flags"] = {"critical_error": True, "router_error": True,
                               "factual_error": True, "too_long": True}
    rows[1]["issue_flags"] = {"university_policy_assertion": True, "retrieval_error": True,
                               "tool_error": True, "model_error": True, "other_error": True}
    summary = api_main.build_campus_v21_human_summary(rows)
    assert summary["human_gate"]["status"] == "FAIL"
    assert summary["human_gate"]["critical_error_rate"] == .01
    assert summary["human_gate"]["university_policy_assertion_rate"] == .01
    categories = {entry["category"]: entry["count"] for entry in summary["error_categories"]}
    assert categories == {"ROUTER": 1, "RETRIEVAL": 1, "TOOL": 1, "MODEL": 1,
                          "KNOWLEDGE": 2, "UX": 1, "OTHER": 1}
    assert summary["v2_2_priorities"][0]["category"] == "KNOWLEDGE"


def test_v21_partial_export_writes_required_json_and_markdown(tmp_path):
    source = tmp_path / "human-comparison-campus-v21.json"
    rows = [human_row(0)]
    exported = api_main.export_campus_v21_human_results(rows, source)
    results = tmp_path / "campus-v21-human-results.json"
    report = tmp_path / "campus-v21-human-report.md"
    assert results.exists() and report.exists()
    payload = json.loads(results.read_text(encoding="utf-8"))
    assert payload["human_gate"]["status"] == "PENDING"
    assert payload["answer_logic_changed"] is False and payload["production_changed"] is False
    assert "withheld until 100/100 completion" in report.read_text(encoding="utf-8")
    assert exported["results_path"].endswith("campus-v21-human-results.json")
