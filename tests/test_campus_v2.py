from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import api.main as api_main
from pipeline.campus_retrieval import load_jsonl
from pipeline.campus_router_v2 import CampusRouterV2
from pipeline.campus_tools_v2 import CampusToolEngineV2
from pipeline.campus_v2 import UniPilotCampusV2
from pipeline.campus_validator_v2 import CampusValidatorV2


_PIPELINE = None


def pipeline() -> UniPilotCampusV2:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = UniPilotCampusV2()
    return _PIPELINE


def test_campus_v2_data_is_independent_and_balanced():
    manifest = json.loads(Path("data/campus_v2/manifest.json").read_text(encoding="utf-8"))
    assert manifest["router_train"] >= 2000
    assert manifest["router_train_compound"] >= 500
    assert manifest["blind"] == 2000
    assert manifest["blind_distribution"] == {"colloquial": 500, "normal": 500, "ambiguous": 300,
                                                "compound": 400, "hard_negation_typo": 300}
    assert manifest["train_blind_normalized_overlap"] == 0


def test_campus_v2_router_returns_hierarchy_top2_confidence_action_and_multi_intent():
    router = CampusRouterV2(load_jsonl("data/campus_v2/router/train.jsonl"))
    decision = router.decide("欠席メールを作りたい。それと、試験まで7日の勉強計画もほしい")
    assert decision.primary == "absence_email"
    assert {"absence_email", "study_plan"}.issubset(decision.intents)
    assert decision.level1 == "communication"
    assert len(decision.top2) == 2
    assert decision.confidence_band in ("high", "medium")
    assert decision.action == "TOOL+MODEL"


def test_campus_v2_low_confidence_clarifies_and_does_not_guess():
    result = pipeline().answer("やばい")
    assert result["route_action"] == "CLARIFY"
    assert "何について" in result["text"]
    assert result["cards"][0]["kind"] == "clarify"


def test_campus_v2_new_calculators_are_deterministic_and_university_neutral():
    engine = CampusToolEngineV2()
    target = engine.execute("gpa", "目標GPA", {}, {"current_gpa": 2.5, "current_credits": 60,
                                                    "target_gpa": 3.0, "future_credits": 30})
    credit = engine.execute("credit", "単位進捗", {}, {"earned_credits": 80, "required_credits": 124})
    allocation = engine.execute("presentation_outline", "10分の時間配分", {}, {"total_minutes": 10})
    assert target.calculation["required_future_gpa"] == 4.0
    assert credit.calculation["remaining_credits"] == 44
    assert allocation.calculation["本論"] == 5.5
    assert all(subject not in target.text + credit.text for subject in ("法学", "経済学"))


def test_campus_v2_session_keeps_gpa_credit_deadline_and_tasks():
    store = pipeline().sessions
    state = store.update("campus-v2-test", "現在GPA 2.7、取得済み70単位、必要124単位、試験日2026-09-10、課題: 統計レポート")
    assert state["gpa"] == 2.7 and state["earned_credits"] == 70
    assert state["required_credits"] == 124 and state["exam_date"] == "2026-09-10"
    assert "統計レポート" in state["tasks"]


def test_campus_validator_v2_rejects_policy_calculation_and_mixed_template():
    validator = CampusValidatorV2()
    assert not validator.validate("欠席", "この大学では欠席3回で単位を落とすと決まっています。").valid
    assert not validator.validate("点数", "計算結果：残り42点です。").valid
    mixed = "件名：相談 先生 序論 本論 結論 ES 面接 企業 計算結果 式：1+1=2"
    assert "mixed_templates" in validator.validate("相談", mixed).issues


def test_campus_v2_human_endpoint_persists_five_axes(tmp_path, monkeypatch):
    path = tmp_path / "campus-v2-human.json"
    path.write_text(json.dumps([{"id": "v2-x", "scores": {"correctness": None}, "chatgpt_answer": "",
                                "gemini_answer": "", "notes": ""}], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(api_main, "HUMAN_CAMPUS_V2", path)
    payload = {"item_id": "v2-x", "correctness": 4, "relevance": 5, "actionable": 4,
               "naturalness": 5, "would_use_again": 4, "notes": "manual"}
    with TestClient(api_main.app) as client:
        response = client.post("/human-eval/campus-v2", json=payload)
    assert response.status_code == 200
    saved = json.loads(path.read_text(encoding="utf-8"))[0]
    assert saved["scores"] == {"correctness": 4, "relevance": 5, "actionable": 4,
                               "naturalness": 5, "would_use_again": 4}
