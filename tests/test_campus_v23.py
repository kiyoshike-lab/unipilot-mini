from __future__ import annotations

from collections import Counter
from functools import cache
import json
from pathlib import Path

from evaluation.evaluate_campus_v22_generalization import judge_result
from pipeline.campus_planner_v23 import CampusCompletenessPlannerV23, requirement_coverage
from pipeline.campus_retrieval_v23 import (
    CampusKnowledgeRetrieverV23,
    V23_RETRIEVAL_STRATEGIES,
    rewrite_queries,
)
from pipeline.campus_v23 import UniPilotCampusV23
from quality.campus_ai_judge import CampusAIJudge


@cache
def retriever() -> CampusKnowledgeRetrieverV23:
    return CampusKnowledgeRetrieverV23.from_files()


@cache
def pipeline() -> UniPilotCampusV23:
    return UniPilotCampusV23()


def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_v23_retrieval_is_two_stage_and_quality_scored():
    knowledge = retriever()
    assert len(knowledge.rows) == 1096
    quality_fields = {
        "source_authority", "freshness", "specificity", "university_specific",
        "official_or_public", "license_status", "overall",
    }
    assert all(quality_fields <= row["knowledge_quality"].keys() for row in knowledge.rows)

    target = next(row for row in knowledge.rows if not row.get("university_specific"))
    for strategy in V23_RETRIEVAL_STRATEGIES:
        results, meta = knowledge.search(
            f"『{target['title']}』を根拠付きで説明して", target["category"], top_k=5,
            strategy=strategy, confidence_policy="precision",
        )
        assert results
        assert meta["method"] == strategy and meta["two_stage"] is True
        assert 10 <= meta["stage1_candidates"] <= 20
        assert meta["confidence"] in {"HIGH", "MEDIUM", "LOW", "REJECT"}
        assert isinstance(meta["accepted"], bool)


def test_query_rewriter_keeps_original_and_adds_safe_candidates():
    variants = rewrite_queries("単位やばい、かくにん先どこ", "credits", "attendance")
    assert variants[0] == "単位やばい、かくにん先どこ"
    assert len(variants) >= 3
    assert any("単位取得条件" in value for value in variants)
    assert any("確認" in value for value in variants)


def test_all_v22_false_matches_and_conflicts_are_auditable():
    analysis = read_json("data/campus_v23/retrieval/v22-false-matches.json")
    hard_negatives = read_json("data/campus_v23/retrieval/hard-negatives.json")
    conflicts = read_json("data/campus_v23/retrieval/numeric-conflicts.json")

    assert analysis["summary"]["questions"] == 300
    assert analysis["summary"]["false_matches"] == len(analysis["items"]) == 69
    assert sum(analysis["summary"]["cause_counts"].values()) == 69
    required = {
        "query", "expected_topic", "retrieved_chunk", "retrieved_source",
        "retrieval_score", "rerank_score", "category", "failure_reason",
    }
    assert all(required <= row.keys() for row in analysis["items"])
    assert len(hard_negatives["items"]) == 69 and hard_negatives["automatic_training"] is False
    assert conflicts["candidate_groups"] == len(conflicts["items"]) == 10
    assert conflicts["automatic_truth_selection"] is False
    assert all(row["automatic_truth_selection"] is False for row in conflicts["items"])


def test_completeness_planner_covers_every_multi_intent_requirement():
    question = (
        "テストが明日なのに何もしていません。レポートも今日締切です。"
        "どちらを先にして、残り時間をどう配分すればよいですか？"
    )
    plan = CampusCompletenessPlannerV23().plan(question)
    requirement_ids = {row.id for row in plan.atomic_requirements}
    assert len(plan.sub_intents) >= 2
    assert {"priority", "immediate_action", "time_allocation"} <= requirement_ids

    result = pipeline().answer(question, session_id="v23-coverage-test")
    measured = requirement_coverage(result["text"], plan.atomic_requirements)
    assert result["pipeline"] == "campus-v2.3"
    assert result["planner_hidden"] is True
    assert result["answer_coverage"]["target"] == 1.0
    assert result["answer_coverage"]["score"] == measured["score"] == 1.0
    assert result["quality_checks"]["length_ok"] is True


def test_tool_veto_and_general_fix_remove_all_three_v22_critical_failures():
    rows = read_json("evaluation/campus-v23-v22-critical-root-causes.json")["items"]
    assert len(rows) == 3
    judge = CampusAIJudge()
    for row in rows:
        result = pipeline().answer(row["question"], session_id=f"v23-critical-{row['id']}")
        judged = judge_result(judge, row["question"], row["expected_category"], result)
        assert result["route"] != "tool"
        assert result["category"] == row["expected_category"]
        assert result["answer_coverage"]["target_met"] is True
        assert judged["quality_label"] != "bad"
        assert judged["hallucination_suspected"] is False
        assert judged["unsupported_claims"] == []
        assert "TOOL_ISSUE" not in judged["issues"]


def test_source_conflict_is_safe_specific_and_does_not_invoke_study_tool():
    result = pipeline().answer(
        "TOEICについて公式ページとLMSの表示が違う。どちらを優先し、何を記録して問い合わせればいい？",
        session_id="v23-source-conflict",
    )
    assert result["route"] != "tool"
    assert result["answer_coverage"]["score"] == 1.0
    assert all(token in result["text"] for token in ("公式ページ", "LMS", "記録", "問い合わせ内容"))
    assert "TOEIC 30日計画" not in result["text"]


def test_v23_session_memory_is_in_memory_only():
    instance = pipeline()
    instance.answer("明日の試験対策を整理して", session_id="v23-memory")
    followup = instance.answer("もっと詳しく", session_id="v23-memory")
    assert followup.get("followup_of")
    assert not hasattr(instance.conversation_memory, "path")


def test_v23_holdouts_are_fixed_balanced_and_never_training_data():
    blind_path = Path("data/campus_v23/holdouts/blind-500.json")
    stress_path = Path("data/campus_v23/holdouts/stress-200.json")
    comparison_path = Path("data/campus_v23/holdouts/comparison-50.json")
    if not (blind_path.exists() and stress_path.exists() and comparison_path.exists()):
        return
    blind = read_json(str(blind_path))
    stress = read_json(str(stress_path))
    comparison = read_json(str(comparison_path))
    assert len(blind["items"]) == len({row["question"] for row in blind["items"]}) == 500
    counts = Counter(row["expected_category"] for row in blind["items"])
    assert max(counts.values()) - min(counts.values()) <= 1
    assert max(row["max_reference_similarity"] for row in blind["items"]) < .70
    assert max(row["max_internal_similarity"] for row in blind["items"]) < .90
    assert all(row["holdout"] and not row["used_for_improvement"] for row in blind["items"])
    assert all(row["forbidden_for_training"] and row["forbidden_for_faq_tuning"] for row in blind["items"])
    assert len(stress["items"]) == 200 and set(stress["type_counts"].values()) == {20}
    assert all(row["holdout"] and not row["used_for_improvement"] for row in stress["items"])
    assert len(comparison["items"]) == 50
    assert comparison["external_answers_fabricated"] is False
    assert all(set(row["comparisons"]) == {"unipilot_vs_chatgpt", "unipilot_vs_gemini"}
               for row in comparison["items"])
    assert all(pair["blind_slots"] == {"A": None, "B": None}
               for row in comparison["items"] for pair in row["comparisons"].values())


def test_v23_final_evaluation_exports_are_complete_and_gate_is_mechanical():
    required = {
        "evaluation/campus-v23-old-100.json",
        "evaluation/campus-v23-old-blind-300.json",
        "evaluation/campus-v23-blind-500.json",
        "evaluation/campus-v23-stress-200.json",
        "evaluation/campus-v23-retrieval.json",
        "evaluation/campus-v23-review-queue.json",
        "evaluation/campus-v23-summary.json",
        "evaluation/campus-v23-report.md",
    }
    assert all(Path(path).exists() for path in required)
    old = read_json("evaluation/campus-v23-old-100.json")
    old_blind = read_json("evaluation/campus-v23-old-blind-300.json")
    blind = read_json("evaluation/campus-v23-blind-500.json")
    stress = read_json("evaluation/campus-v23-stress-200.json")
    retrieval = read_json("evaluation/campus-v23-retrieval.json")
    review = read_json("evaluation/campus-v23-review-queue.json")
    summary = read_json("evaluation/campus-v23-summary.json")
    assert len(old["items"]) == 100 and old["development_set"] is True
    assert len(old_blind["items"]) == 300 and old_blind["development_set"] is True
    assert len(blind["items"]) == 500 and blind["holdout"] and not blind["used_for_improvement"]
    assert len(stress["items"]) == 200 and stress["holdout"] and not stress["used_for_improvement"]
    assert retrieval["final_holdout"] is True and retrieval["selected"] in retrieval["strategies"]
    assert review["review_required"] == len(review["items"])
    assert review["automatic_training"] is False
    assert summary["external_ai_api"] == "OFF"
    assert summary["v2_1_rc_changed"] is False and summary["production_changed"] is False
    assert summary["production_gate"] in {"PASS", "FAIL"}
    assert summary["beta_recommended"] == (summary["production_gate"] == "PASS")
