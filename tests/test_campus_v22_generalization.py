from __future__ import annotations

from collections import Counter
from functools import cache
import json
from pathlib import Path

from evaluation.evaluate_campus_v22_generalization import human_agreement
from pipeline.campus_generalizer_v22 import CampusResponseGeneralizerV22, DEPTH_LIMITS
from pipeline.campus_planner_v22 import CampusAnswerPlannerV22, CampusConversationMemoryV22
from pipeline.campus_retrieval_v22 import CampusKnowledgeRetrieverV22, RETRIEVAL_STRATEGIES
from pipeline.campus_v22 import UniPilotCampusV22
from quality.campus_ai_judge import CampusAIJudge


@cache
def retriever() -> CampusKnowledgeRetrieverV22:
    return CampusKnowledgeRetrieverV22.from_files()


def test_planner_selects_depth_and_keeps_plan_internal():
    planner = CampusAnswerPlannerV22()
    simple = planner.plan("GPAって何？")
    normal = planner.plan("奨学金を初めて申請します。募集要項の条件と確認先を順番に教えてください")
    complex_plan = planner.plan(
        "明日テストなのに全然勉強していません。しかもレポートも今日締切です。"
        "どちらから始めて、残り時間をどう配分すればよいですか？"
    )
    assert simple.answer_depth == "simple"
    assert normal.answer_depth == "normal"
    assert complex_plan.answer_depth == "complex"
    assert len(complex_plan.sub_intents) >= 2

    result = UniPilotCampusV22().answer("GPAって何？")
    assert result["planner_hidden"] is True
    assert "plan" not in result and "known_facts" not in result


def test_one_revision_enforces_depth_and_multi_intent_coverage():
    planner = CampusAnswerPlannerV22()
    generalizer = CampusResponseGeneralizerV22()
    questions = {
        "simple": "GPAって何？",
        "normal": "奨学金を初めて申請します。募集要項の条件と確認先を順番に教えてください",
        "complex": (
            "明日テストなのに勉強できていません。しかもレポートも今日締切です。"
            "どちらを先にして、その後どう進めるか詳しく教えてください。"
        ),
    }
    for expected_depth, question in questions.items():
        plan = planner.plan(question)
        improved = generalizer.improve(question, "確認してください。", plan, {"route": "safe", "cards": []})
        minimum, maximum = DEPTH_LIMITS[expected_depth]
        assert plan.answer_depth == expected_depth
        assert minimum <= len(improved.text) <= maximum
        assert improved.revision_count == 1
        assert improved.checks["after"]["needs_revision"] is False
        assert improved.checks["after"]["all_question_elements_answered"] is True


def test_session_memory_links_short_followup_without_persistence():
    memory = CampusConversationMemoryV22(turns_per_session=2)
    planner = CampusAnswerPlannerV22()
    memory.remember("student-session", "明日テスト", "exam")
    contextual = planner.plan("数学", previous_question=memory.latest_question("student-session"))
    assert "明日テスト" in contextual.contextual_question
    assert "数学" in contextual.contextual_question
    assert not hasattr(memory, "path")
    assert memory.clear("student-session") is True
    assert memory.latest_question("student-session") is None


def test_unknown_question_is_useful_and_tool_result_has_action_card():
    pipeline = UniPilotCampusV22()
    unknown = pipeline.answer("やばい")
    assert len(unknown["text"]) >= DEPTH_LIMITS[unknown["answer_depth"]][0]
    assert "今やること" in unknown["text"]

    gpa = pipeline.answer("GPAを計算して。A 2単位、B 2単位", tool_inputs={"courses": [
        {"name": "科目A", "grade": "A", "credits": 2},
        {"name": "科目B", "grade": "B", "credits": 2},
    ]})
    assert gpa["route"] == "tool" and gpa["calculation"]["gpa"] == 2.5
    assert "2.50" in gpa["text"]
    assert any(card["kind"] == "action_plan" and card.get("copy_text") for card in gpa["cards"])


def test_knowledge_chunks_and_retrieval_strategies_have_provenance():
    knowledge = retriever()
    assert len(knowledge.rows) >= 1000
    assert len({row["id"] for row in knowledge.rows}) == len(knowledge.rows)
    assert all(len(row["text"]) <= 300 for row in knowledge.rows)
    required = {"parent_id", "chunk_index", "source_url", "title", "publisher", "retrieved_at",
                "license", "revision_or_date", "summary"}
    assert all(required <= row.keys() for row in knowledge.rows)

    target = next(row for row in knowledge.rows if not row.get("university_specific"))
    for strategy in RETRIEVAL_STRATEGIES:
        results, meta = knowledge.search(target["title"], target["category"], top_k=3,
                                         threshold=0.0, strategy=strategy)
        assert results
        assert meta["method"] == strategy
        assert meta["knowledge_chunks"] >= 1000


def test_blind_and_stress_sets_are_fixed_holdouts():
    blind = json.loads(Path("data/campus_v22/generalization/blind-300.json").read_text(encoding="utf-8"))
    stress = json.loads(Path("data/campus_v22/generalization/stress-100.json").read_text(encoding="utf-8"))
    assert blind["holdout"] is True and blind["used_for_generation_improvement"] is False
    assert len(blind["items"]) == len({row["question"] for row in blind["items"]}) == 300
    category_counts = Counter(row["expected_category"] for row in blind["items"])
    assert min(category_counts.values()) >= 9 and max(category_counts.values()) <= 11
    assert max(row["max_reference_similarity"] for row in blind["items"]) < .78
    assert all(row["forbidden_for_training"] and row["forbidden_for_faq_tuning"] for row in blind["items"])

    assert stress["holdout"] is True and stress["used_for_generation_improvement"] is False
    assert len(stress["items"]) == 100
    assert set(stress["type_counts"].values()) == {10}
    assert len(stress["type_counts"]) == 10


def test_human_judge_calibration_reaches_target_without_answer_keys():
    config = json.loads(Path("quality/campus_ai_judge_calibration.json").read_text(encoding="utf-8"))
    forbidden = set(config["forbidden_features"])
    assert {"question_id", "exact_question", "exact_answer", "human_answer_key"} <= forbidden
    agreement = human_agreement(CampusAIJudge())
    assert agreement["calibrated_rate"] >= .85
    assert agreement["calibration_used_for_answer_generation"] is False


def test_generalization_evaluation_exports_are_complete_and_not_training_data():
    details = json.loads(Path("evaluation/campus-v22-generalization-100-details.json").read_text(encoding="utf-8"))
    blind = json.loads(Path("evaluation/campus-v22-generalization-blind-300.json").read_text(encoding="utf-8"))
    stress = json.loads(Path("evaluation/campus-v22-generalization-stress-100.json").read_text(encoding="utf-8"))
    retrieval = json.loads(Path("evaluation/campus-v22-generalization-retrieval.json").read_text(encoding="utf-8"))
    review = json.loads(Path("evaluation/campus-v22-generalization-review-queue.json").read_text(encoding="utf-8"))
    summary = json.loads(Path("evaluation/campus-v22-generalization-summary.json").read_text(encoding="utf-8"))

    assert len(details["items"]) == 100
    assert all({"question", "category", "v2_1", "v2_2"} <= row.keys() for row in details["items"])
    assert len(blind["items"]) == 300 and blind["used_for_generation_improvement"] is False
    assert len(stress["items"]) == 100 and stress["used_for_generation_improvement"] is False
    assert retrieval["selected"] in retrieval["strategies"]
    assert len(review["items"]) == review["review_required"] == summary["human_review_required"]
    assert review["automatic_training"] is False
    assert summary["external_ai_api"] == "OFF"
    assert summary["v2_1_rc_changed"] is False and summary["production_changed"] is False

    candidates = [json.loads(line) for line in Path(
        "data/curated/campus-v22-generalization-candidates.jsonl"
    ).read_text(encoding="utf-8").splitlines() if line]
    assert len(candidates) == summary["training_candidates"]
    assert all(row["human_approved"] is False and row["automatic_training"] is False for row in candidates)
