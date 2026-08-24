import json
from pathlib import Path

from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app, run_generation, runtime
from api.schemas import ChatRequest
from evaluation.evaluate_campus_ai_quality import evaluate
from quality.campus_ai_judge import AXES, CampusAIJudge, RedundancyDetector, UnsupportedClaimDetector
from quality.campus_answer_improver import CampusAnswerImprover


def test_local_judge_is_deterministic_and_has_all_axes():
    judge = CampusAIJudge()
    metadata = {"category": "study_plan", "route": "safe", "action": "FAQ"}
    answer = ("結論：試験日と範囲を確認して、苦手な科目から着手します。\n"
              "理由：締切と苦手度を先に比べると優先順位を決めやすいためです。\n"
              "今やること：1. 試験日を書く 2. 範囲を分ける 3. 問題を解いて復習する。\n"
              "注意：授業ごとの条件はシラバスで確認してください。")
    first = judge.evaluate("試験勉強の計画を立てたい 短め", answer, metadata)
    second = judge.evaluate("試験勉強の計画を立てたい 短め", answer, metadata)
    assert first == second
    assert set(first["scores_0_to_5"]) == set(AXES)
    assert 0 <= first["overall_score"] <= 100
    assert first["judge_type"] == "deterministic_local"
    assert first["external_ai_api"] == "OFF"


def test_redundancy_and_unsupported_claim_detection():
    repeated = "履修要項を確認してください。履修要項を確認してください。教務にも確認してください。"
    assert RedundancyDetector().analyse(repeated).rate > 0
    claims = UnsupportedClaimDetector().analyse("出席について教えて", "欠席は必ず3回まで認められます。", [])
    assert claims["unsupported_claim_rate"] == 1
    grounded = UnsupportedClaimDetector().analyse("欠席は3回まで？", "資料では欠席は3回までです。", ["欠席は3回まで"])
    assert grounded["unsupported_claim_rate"] == 0


def test_improver_rewrites_at_most_once_and_does_not_train():
    result = CampusAnswerImprover().improve(
        "欠席回数を確認したい 短め",
        "結論：一致度の高いFAQを確認できなかったため、推測しません。",
        {"category": "attendance", "route": "safe", "action": "FAQ"},
        force=True,
    )
    assert result["rewrite_count"] == 1
    assert result["improved_answer"] != result["original"]
    assert result["after_judge"]["overall_score"] > result["before_judge"]["overall_score"]
    assert result["production_eligible_automatically"] is False
    assert result["external_ai_api"] == "OFF"


def test_quality_evaluation_artifacts_and_review_filter():
    summary = evaluate()
    assert summary["human"] == {"good": 4, "close": 15, "bad": 1}
    assert summary["agreement"] >= .70
    assert sum(summary["ai_20"].values()) == 20
    assert sum(summary["ai_100"].values()) == 100
    output_20 = json.loads(Path("evaluation/campus-ai-quality-20.json").read_text(encoding="utf-8"))
    assert output_20["summary"]["ai_improved"] == {"good": 20, "close": 0, "bad": 0}
    assert output_20["summary"]["average_score"]["improved"] >= 90
    queue = json.loads(Path("evaluation/campus-ai-review-queue.json").read_text(encoding="utf-8"))
    assert queue["review_required"] == len(queue["items"])
    for item in queue["items"]:
        assert item["review_reasons"]
        assert item["review_status"] == "pending"
    close = json.loads(Path("evaluation/campus-v21-close-analysis.json").read_text(encoding="utf-8"))
    critical = json.loads(Path("evaluation/campus-v21-critical-failure.json").read_text(encoding="utf-8"))
    assert close["human_close_count"] == 15
    assert critical["count"] == 1


def test_review_queue_adopt_revise_reject_and_memory(tmp_path, monkeypatch):
    queue_path = tmp_path / "queue.json"
    decisions_path = tmp_path / "decisions.json"
    approved_path = tmp_path / "curated" / "approved.jsonl"
    queue_path.write_text(json.dumps({"schema_version": "test", "review_required": 1, "items": [{
        "item_id": "q-1", "question": "質問", "original_answer": "元回答", "improved_answer": "改善回答",
        "category": "general", "route": "safe", "source_ids": [], "ai_judge_score": 70,
        "improved_score": 92, "problems": ["TOO_SHORT"], "review_reasons": ["SCORE_BELOW_80"],
        "critique": ["短い"], "review_status": "pending",
    }]}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(api_main, "CAMPUS_AI_REVIEW_QUEUE", queue_path)
    monkeypatch.setattr(api_main, "CAMPUS_AI_REVIEW_DECISIONS", decisions_path)
    monkeypatch.setattr(api_main, "CAMPUS_AI_APPROVED_ANSWERS", approved_path)
    with TestClient(app) as client:
        initial = client.get("/ai-review/campus")
        adopted = client.post("/ai-review/campus", json={"item_id": "q-1", "decision": "adopt",
                                                          "edited_answer": "人間が確認した回答", "notes": "確認済み"})
        adopted_memory = approved_path.read_text(encoding="utf-8")
        saved = client.get("/ai-review/campus")
        revised = client.post("/ai-review/campus", json={"item_id": "q-1", "decision": "revise",
                                                          "edited_answer": "再修正版", "notes": "再確認"})
        rejected = client.post("/ai-review/campus", json={"item_id": "q-1", "decision": "reject",
                                                           "edited_answer": "", "notes": "不採用"})
        final = client.get("/ai-review/campus")
    assert initial.json()["pending"] == 1
    assert adopted.status_code == 200 and adopted.json()["automatic_training"] is False
    approved = [json.loads(line) for line in adopted_memory.splitlines()]
    assert approved[0]["approved_answer"] == "人間が確認した回答"
    assert approved[0]["requires_training_review"] is True
    assert saved.json()["decision_counts"]["adopt"] == 1
    assert revised.status_code == 200
    assert rejected.status_code == 200
    assert final.json()["decision_counts"] == {"adopt": 0, "revise": 0, "reject": 1}
    assert approved_path.read_text(encoding="utf-8") == ""


def test_chat_quality_mode_is_opt_in(monkeypatch):
    class Pipeline:
        version = "campus-v2.1"

        @staticmethod
        def answer(*_args, **_kwargs):
            return {"text": "結論：一致度の高いFAQを確認できなかったため、推測しません。",
                    "category": "attendance", "route": "safe", "action": "FAQ", "cards": [],
                    "generation_metrics": {"total_seconds": 0.01}}

    monkeypatch.setitem(runtime, "model", object())
    monkeypatch.setitem(runtime, "pipeline", Pipeline())
    plain = run_generation(ChatRequest(prompt="欠席回数を確認したい", quality_mode="off"), True)
    improved = run_generation(ChatRequest(prompt="欠席回数を確認したい", quality_mode="improve"), True)
    assert "original_text" not in plain
    assert improved["quality_mode"] == "improve"
    assert improved["rewrite_count"] <= 1
    assert improved["automatic_training"] is False
