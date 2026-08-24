from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any

from pipeline.campus_retrieval import load_jsonl
from pipeline.campus_retrieval_v22 import CampusKnowledgeRetrieverV22, RETRIEVAL_STRATEGIES
from pipeline.campus_v22 import UniPilotCampusV22
from quality.campus_ai_judge import AXES, CampusAIJudge


ROOT = Path(__file__).resolve().parents[1]
BASELINE_100 = ROOT / "evaluation/campus-ai-quality-100.json"
HUMAN_20 = ROOT / "evaluation/campus-ai-quality-20.json"
BLIND_PATH = ROOT / "data/campus_v22/generalization/blind-300.json"
STRESS_PATH = ROOT / "data/campus_v22/generalization/stress-100.json"
RETRIEVAL_BENCHMARK = ROOT / "data/campus_v22/benchmarks/knowledge-1000.jsonl"
KNOWLEDGE_MANIFEST = ROOT / "data/campus_v22/generalization/knowledge-chunk-manifest.json"

DETAILS_100 = ROOT / "evaluation/campus-v22-generalization-100-details.json"
BLIND_RESULTS = ROOT / "evaluation/campus-v22-generalization-blind-300.json"
STRESS_RESULTS = ROOT / "evaluation/campus-v22-generalization-stress-100.json"
RETRIEVAL_RESULTS = ROOT / "evaluation/campus-v22-generalization-retrieval.json"
REVIEW_QUEUE = ROOT / "evaluation/campus-v22-generalization-review-queue.json"
CANDIDATES = ROOT / "data/curated/campus-v22-generalization-candidates.jsonl"
SUMMARY = ROOT / "evaluation/campus-v22-generalization-summary.json"
REPORT = ROOT / "evaluation/campus-v22-generalization-report.md"

LABELS = ("good", "close", "bad")
POLICY_ASSERTIONS = ("一律", "全大学で", "すべての大学で", "どの大学でも", "自動的に", "無条件で", "免除されます")
MANDATORY_POLICY_PREDICATES = ("認めら", "適用され", "公欠にな", "免除され", "取得でき", "申請でき", "扱われ")
BOUNDARIES = ("断定できません", "断定しません", "とは限りません", "大学ごと", "授業ごと", "確認してください", "確認が必要")
POLICY_CONTEXT = ("大学", "学則", "制度", "規程", "公欠", "出席", "欠席", "履修", "単位", "申請", "免除", "奨学金", "学費")


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def source_texts(result: dict) -> list[str]:
    return [row.get("selected_text", "") for row in result.get("retrieval", []) if row.get("selected_text")]


def university_policy_assertion(answer: str) -> bool:
    for sentence in re.split(r"(?<=[。！？])\s*|\n+", answer):
        assertive = (any(token in sentence for token in POLICY_ASSERTIONS)
                     or ("必ず" in sentence and any(token in sentence for token in MANDATORY_POLICY_PREDICATES)))
        if (assertive
                and any(context in sentence for context in POLICY_CONTEXT)
                and not any(boundary in sentence for boundary in BOUNDARIES)):
            return True
    return False


def judge_result(judge: CampusAIJudge, question: str, expected_category: str, result: dict) -> dict:
    metadata = {
        "category": expected_category,
        "predicted_category": result.get("category"),
        "route": result.get("route"),
        "action": result.get("route_action"),
        "cards": result.get("cards", []),
        "sources": [source.get("id") for source in result.get("sources", [])],
        "answer_depth": result.get("answer_depth"),
    }
    judged = judge.evaluate(question, result.get("text", ""), metadata, source_texts(result))
    judged["university_policy_assertion"] = university_policy_assertion(result.get("text", ""))
    return judged


def aggregate(records: list[dict], judge_key: str = "judge") -> dict:
    total = max(1, len(records))
    labels = Counter(row[judge_key]["quality_label"] for row in records)
    axis = {name: round(mean(row[judge_key]["scores_0_to_5"][name] for row in records), 3) for name in AXES}
    route_counts = Counter(row.get("route") for row in records)
    critical = sum(
        row[judge_key]["quality_label"] == "bad"
        or row[judge_key]["hallucination_suspected"]
        or row[judge_key].get("university_policy_assertion", False)
        for row in records
    )
    return {
        "questions": len(records),
        "ai_quality_gate": {label: labels[label] for label in LABELS},
        "average_score": round(mean(row[judge_key]["overall_score"] for row in records), 2),
        "average_characters": round(mean(len(row["answer"]) for row in records), 2),
        "axis_averages_0_to_5": axis,
        "axis_percent": {name: round(value / 5 * 100, 2) for name, value in axis.items()},
        "unsupported_claim_rate": round(sum(bool(row[judge_key]["unsupported_claims"]) for row in records) / total, 4),
        "hallucination_rate": round(sum(row[judge_key]["hallucination_suspected"] for row in records) / total, 4),
        "university_policy_assertion_rate": round(sum(row[judge_key].get("university_policy_assertion", False) for row in records) / total, 4),
        "critical_errors": critical,
        "route_counts": dict(route_counts),
        "route_percent": {route: round(count / total * 100, 2) for route, count in route_counts.items()},
        "tool_usage_rate": round(route_counts["tool"] / total, 4),
        "rag_usage_rate": round(route_counts["rag"] / total, 4),
        "clarification_rate": round(route_counts["clarify"] / total, 4),
        "fallback_rate": round((route_counts["safe"] + route_counts["safety"]) / total, 4),
        "revision_rate": round(sum(row.get("revision_count", 0) > 0 for row in records) / total, 4),
    }


def campus_quality_proxy(records: list[dict]) -> dict:
    """Local rubric measurements only; no external model output is queried or inferred."""
    total = max(1, len(records))
    directness = sum(bool(row["judge"]["checks"].get("direct_answer_or_conclusion")) for row in records) / total
    axis = aggregate(records)["axis_percent"]
    tool_records = [row for row in records if row.get("route") == "tool"]
    tool_card_rate = (
        sum(any(card.get("action_label") or card.get("copy_text") for card in row.get("cards", []))
            for row in tool_records) / len(tool_records)
        if tool_records else 0.0
    )
    dimensions = {
        "directness": round(directness * 100, 2),
        "completeness": axis["completeness"],
        "specificity": axis["specificity"],
        "actionability": axis["actionable"],
        "grounding": axis["grounding"],
        "student_tool_usefulness": round(tool_card_rate * 100, 2),
    }
    return {
        "external_chatgpt_api_used": False,
        "head_to_head_claim_allowed": False,
        "measurement": "Campus local rubric proxy; manual blind pairwise review is required for a ChatGPT comparison",
        "dimensions_percent": dimensions,
        "closer_points": [name for name, value in dimensions.items() if value >= 90],
        "remaining_gaps": [name for name, value in dimensions.items() if value < 90],
        "tool_route_questions": len(tool_records),
    }


def evaluate_questions(pipeline: UniPilotCampusV22, judge: CampusAIJudge, rows: list[dict], prefix: str) -> list[dict]:
    records = []
    for index, row in enumerate(rows):
        question = row["question"]
        expected = row.get("expected_category") or row.get("category") or "general"
        result = pipeline.answer(question, response_mode="auto", session_id=f"{prefix}-{index:03d}")
        judged = judge_result(judge, question, expected, result)
        records.append({
            "id": row.get("id") or f"{prefix}-{index:03d}",
            "question": question,
            "category": expected,
            "predicted_category": result.get("category"),
            "route": result.get("route"),
            "answer": result.get("text", ""),
            "answer_depth": result.get("answer_depth"),
            "revision_count": result.get("revision_count", 0),
            "score": judged["overall_score"],
            "judge": judged,
            "retrieval_result": result.get("retrieval", []),
            "sources": result.get("sources", []),
            "cards": result.get("cards", []),
            "validator": result.get("validator", {}),
            "quality_checks": result.get("quality_checks", {}),
            "feature": row.get("feature") or row.get("stress_type"),
            "must_not_assert_unverified_policy": row.get("must_not_assert_unverified_policy", False),
        })
    return records


def baseline_summary(rows: list[dict]) -> dict:
    labels = Counter(row["judge"]["quality_label"] for row in rows)
    axes = {axis: round(mean(row["judge"]["scores_0_to_5"][axis] for row in rows), 3) for axis in AXES}
    route_counts = Counter(row["route"] for row in rows)
    return {
        "questions": len(rows), "ai_quality_gate": {label: labels[label] for label in LABELS},
        "average_score": round(mean(row["judge"]["overall_score"] for row in rows), 2),
        "average_characters": round(mean(len(row["original_answer"]) for row in rows), 2),
        "axis_averages_0_to_5": axes,
        "axis_percent": {axis: round(score / 5 * 100, 2) for axis, score in axes.items()},
        "unsupported_claim_rate": round(mean(row["judge"]["unsupported_claim_rate"] for row in rows), 4),
        "route_counts": dict(route_counts),
        "route_percent": {route: round(count / len(rows) * 100, 2) for route, count in route_counts.items()},
        "tool_usage_rate": round(route_counts["tool"] / len(rows), 4),
        "rag_usage_rate": 0.0,
        "clarification_rate": round(route_counts["clarify"] / len(rows), 4),
        "fallback_rate": round((route_counts["safe"] + route_counts["safety"]) / len(rows), 4),
    }


def retrieval_comparison(retriever: CampusKnowledgeRetrieverV22, rows: list[dict], limit: int = 300) -> dict:
    step = max(1, len(rows) // limit)
    sample = rows[::step][:limit]
    comparison = {}
    for strategy in RETRIEVAL_STRATEGIES:
        hit1 = hit3 = reciprocal = false_match = no_result = 0
        latencies = []
        for row in sample:
            results, meta = retriever.search(row["question"], row["category"], top_k=3,
                                             response_mode="normal", strategy=strategy)
            urls = [item.get("source_url") for item in results]
            expected = row.get("expected_source_url")
            rank = next((position + 1 for position, url in enumerate(urls) if url == expected), None)
            hit1 += int(rank == 1)
            hit3 += int(rank is not None and rank <= 3)
            reciprocal += 1 / rank if rank else 0
            no_result += int(not urls)
            false_match += int(bool(urls) and urls[0] != expected)
            latencies.append(float(meta.get("latency_ms", 0.0)))
        total = max(1, len(sample))
        comparison[strategy] = {
            "questions": len(sample), "recall_at_1": round(hit1 / total, 4),
            "recall_at_3": round(hit3 / total, 4), "mrr": round(reciprocal / total, 4),
            "false_match_rate": round(false_match / total, 4), "no_result_rate": round(no_result / total, 4),
            "average_latency_ms": round(mean(latencies), 3),
        }
    best = max(comparison, key=lambda name: (comparison[name]["recall_at_3"], comparison[name]["mrr"],
                                              -comparison[name]["false_match_rate"]))
    return {"benchmark_holdout": True, "sample_policy": "fixed stride from source-linked 1000 set",
            "strategies": comparison, "selected": best}


def human_agreement(judge: CampusAIJudge) -> dict:
    rows = read_json(HUMAN_20)["items"]
    raw = calibrated = 0
    mismatches = []
    for row in rows:
        result = row["original_judge"]
        raw_label = result["quality_label"]
        calibrated_label = judge.calibrated_label(row["question"], row["original_answer"], raw_label,
                                                   result["issues"], result["checks"])
        raw += int(raw_label == row["human_rating"])
        calibrated += int(calibrated_label == row["human_rating"])
        if calibrated_label != row["human_rating"]:
            mismatches.append({"item_id": row["item_id"], "human": row["human_rating"],
                               "calibrated_ai": calibrated_label, "raw_ai": raw_label})
    return {"questions": len(rows), "raw_rate": round(raw / len(rows), 4),
            "calibrated_rate": round(calibrated / len(rows), 4), "target": .85,
            "target_met": calibrated / len(rows) >= .85, "mismatches": mismatches,
            "calibration_used_for_answer_generation": False}


def main() -> int:
    generated_at = datetime.now(timezone.utc).isoformat()
    judge = CampusAIJudge()
    pipeline = UniPilotCampusV22()
    baseline_rows = read_json(BASELINE_100)["items"]
    baseline_by_id = {row["item_id"]: row for row in baseline_rows}

    source_100 = read_json(ROOT / "evaluation/human-comparison-campus-v21.json")
    v22_records = evaluate_questions(pipeline, judge, [
        {"id": row["id"], "question": row["question"], "expected_category": row["category"]}
        for row in source_100
    ], "same100")
    details = []
    causes = Counter()
    for record in v22_records:
        before = baseline_by_id[record["id"]]
        causes.update(before["judge"]["issues"] or ["OTHER"])
        details.append({
            "id": record["id"], "question": record["question"], "category": record["category"],
            "v2_1": {"route": before["route"], "answer": before["original_answer"],
                     "score": before["judge"]["overall_score"],
                     **{axis: before["judge"]["scores_0_to_5"][axis] for axis in AXES},
                     "failure_reasons": before["judge"]["issues"],
                     "retrieval_result": [], "sources": before.get("source_ids", []),
                     "unsupported_claims": before["judge"]["unsupported_claims"]},
            "v2_2": {"route": record["route"], "answer": record["answer"], "score": record["score"],
                     **{axis: record["judge"]["scores_0_to_5"][axis] for axis in AXES},
                     "failure_reasons": record["judge"]["issues"],
                     "retrieval_result": record["retrieval_result"], "sources": record["sources"],
                     "unsupported_claims": record["judge"]["unsupported_claims"],
                     "improved_answer": record["answer"], "answer_depth": record["answer_depth"],
                     "revision_count": record["revision_count"]},
        })
    v21_summary = baseline_summary(baseline_rows)
    v22_summary = aggregate(v22_records)
    write_json(DETAILS_100, {"schema_version": "campus-v22-generalization-100-v1", "generated_at": generated_at,
                             "failure_reason_top10": causes.most_common(10), "items": details})

    blind_rows = read_json(BLIND_PATH)["items"]
    blind_records = evaluate_questions(pipeline, judge, blind_rows, "blind")
    blind_summary = aggregate(blind_records)
    write_json(BLIND_RESULTS, {"schema_version": "campus-v22-generalization-blind-results-v1",
                              "generated_at": generated_at, "holdout": True,
                              "used_for_generation_improvement": False,
                              "summary": blind_summary, "items": blind_records})

    stress_rows = read_json(STRESS_PATH)["items"]
    stress_records = evaluate_questions(pipeline, judge, stress_rows, "stress")
    for record in stress_records:
        if record["must_not_assert_unverified_policy"]:
            record["policy_guardrail_pass"] = not university_policy_assertion(record["answer"])
        else:
            record["policy_guardrail_pass"] = True
    stress_summary = {**aggregate(stress_records),
                      "policy_guardrail_failures": sum(not row["policy_guardrail_pass"] for row in stress_records),
                      "by_stress_type": {kind: {"count": len(group),
                          "average_score": round(mean(row["score"] for row in group), 2),
                          "critical_errors": sum(row["judge"]["quality_label"] == "bad" or not row["policy_guardrail_pass"] for row in group)}
                          for kind, group in ((kind, [row for row in stress_records if row["feature"] == kind])
                                              for kind in sorted({row["feature"] for row in stress_records}))}}
    write_json(STRESS_RESULTS, {"schema_version": "campus-v22-generalization-stress-results-v1",
                               "generated_at": generated_at, "holdout": True,
                               "used_for_generation_improvement": False,
                               "summary": stress_summary, "items": stress_records})

    retrieval = retrieval_comparison(pipeline.knowledge, load_jsonl(RETRIEVAL_BENCHMARK))
    write_json(RETRIEVAL_RESULTS, {"schema_version": "campus-v22-generalization-retrieval-v1",
                                  "generated_at": generated_at, **retrieval})

    review = []
    for record in [*v22_records, *blind_records, *stress_records]:
        reasons = []
        if record["judge"]["quality_label"] == "bad": reasons.append("AI_LABEL_BAD")
        if record["score"] < 80: reasons.append("SCORE_BELOW_80")
        if record["judge"]["unsupported_claims"]: reasons.append("UNSUPPORTED_CLAIM")
        if record["judge"]["hallucination_suspected"]: reasons.append("HALLUCINATION_SUSPECTED")
        if record["judge"].get("university_policy_assertion"): reasons.append("UNIVERSITY_POLICY_ASSERTION")
        if reasons:
            review.append({"item_id": record["id"], "question": record["question"],
                           "answer": record["answer"], "score": record["score"],
                           "category": record["category"], "route": record["route"],
                           "reasons": reasons, "sources": record["sources"], "status": "pending"})
    write_json(REVIEW_QUEUE, {"schema_version": "campus-v22-generalization-review-v1",
                              "generated_at": generated_at, "selection_policy": [
                                  "critical error", "AI label bad", "score < 80",
                                  "unsupported claim", "university policy assertion"],
                              "automatic_training": False, "review_required": len(review), "items": review})

    candidate_rows = []
    for record in [*v22_records, *blind_records]:
        validation = record.get("validator", {})
        checks = record.get("quality_checks", {}).get("after", {})
        source_consistent = not record["judge"]["unsupported_claims"]
        if (record["score"] >= 90 and validation.get("valid", False)
                and not checks.get("needs_revision", True) and source_consistent):
            candidate_rows.append({"id": f"candidate-{record['id']}", "question": record["question"],
                                   "answer": record["answer"], "category": record["category"],
                                   "route": record["route"], "ai_score": record["score"],
                                   "sources": record["sources"], "deterministic_validator_pass": True,
                                   "source_consistent": True, "human_approved": False,
                                   "automatic_training": False, "forbidden_for_training_until_human_approval": True})
    write_jsonl(CANDIDATES, candidate_rows)

    agreement = human_agreement(judge)
    knowledge = read_json(KNOWLEDGE_MANIFEST)
    chatgpt_gap = campus_quality_proxy(blind_records)
    goals_100 = {
        "average_score_gte_92": v22_summary["average_score"] >= 92,
        "good_gte_80": v22_summary["ai_quality_gate"]["good"] >= 80,
        "close_lte_20": v22_summary["ai_quality_gate"]["close"] <= 20,
        "bad_eq_0": v22_summary["ai_quality_gate"]["bad"] == 0,
        "correctness_gte_95": v22_summary["axis_percent"]["correctness"] >= 95,
        "relevance_gte_95": v22_summary["axis_percent"]["relevance"] >= 95,
        "actionable_gte_92": v22_summary["axis_percent"]["actionable"] >= 92,
        "completeness_gte_92": v22_summary["axis_percent"]["completeness"] >= 92,
        "specificity_gte_92": v22_summary["axis_percent"]["specificity"] >= 92,
        "naturalness_gte_98": v22_summary["axis_percent"]["naturalness"] >= 98,
        "unsupported_lte_1": v22_summary["unsupported_claim_rate"] <= .01,
    }
    goals_blind = {
        "average_score_gte_88": blind_summary["average_score"] >= 88,
        "correctness_gte_92": blind_summary["axis_percent"]["correctness"] >= 92,
        "relevance_gte_92": blind_summary["axis_percent"]["relevance"] >= 92,
        "actionable_gte_88": blind_summary["axis_percent"]["actionable"] >= 88,
        "completeness_gte_88": blind_summary["axis_percent"]["completeness"] >= 88,
        "specificity_gte_88": blind_summary["axis_percent"]["specificity"] >= 88,
        "hallucination_lte_1": blind_summary["hallucination_rate"] <= .01,
    }
    beta = all(goals_blind.values()) and stress_summary["critical_errors"] == 0
    standard_50m = blind_summary["axis_percent"]["correctness"] < 90 or blind_summary["axis_percent"]["relevance"] < 90
    payload = {
        "schema_version": "campus-v22-generalization-summary-v1", "generated_at": generated_at,
        "external_ai_api": "OFF", "v2_1_rc_changed": False, "production_changed": False,
        "v2_1_100": v21_summary, "v2_2_100": v22_summary,
        "failure_reason_top10": causes.most_common(10), "blind_300": blind_summary,
        "stress_100": stress_summary, "retrieval": retrieval,
        "knowledge": knowledge, "human_ai_agreement": agreement,
        "chatgpt_gap_proxy": chatgpt_gap,
        "human_review_required": len(review), "training_candidates": len(candidate_rows),
        "goals_100": goals_100, "goals_blind": goals_blind,
        "standard_50m_needed": standard_50m,
        "production_gate": "PENDING_HUMAN_REVIEW" if all(goals_100.values()) and beta else "FAIL",
        "beta_recommended": beta,
    }
    write_json(SUMMARY, payload)
    selected_retrieval = retrieval["strategies"][retrieval["selected"]]
    REPORT.write_text(
        "# Campus v2.2 Generalization Report\n\n"
        "AI Quality Gate is deterministic and does not replace Human Gate. Blind and stress sets are holdout-only.\n\n"
        f"- v2.1 -> v2.2 average: {v21_summary['average_score']} -> {v22_summary['average_score']}\n"
        f"- v2.1 -> v2.2 good/close/bad: {v21_summary['ai_quality_gate']} -> {v22_summary['ai_quality_gate']}\n"
        f"- Blind 300 average: {blind_summary['average_score']}\n"
        f"- Stress 100 critical errors: {stress_summary['critical_errors']}\n"
        f"- Retrieval selected: {retrieval['selected']} (R@1 {selected_retrieval['recall_at_1']}, "
        f"R@3 {selected_retrieval['recall_at_3']}, MRR {selected_retrieval['mrr']}, "
        f"false match {selected_retrieval['false_match_rate']})\n"
        f"- Knowledge: {knowledge['unique_sources']} sources / {knowledge['knowledge_chunks']} chunks\n"
        f"- Human-AI agreement: {agreement['raw_rate']} -> {agreement['calibrated_rate']}\n"
        f"- ChatGPT gap proxy (no external comparison): {chatgpt_gap['dimensions_percent']}\n"
        f"- Remaining local-rubric gaps: {chatgpt_gap['remaining_gaps']}\n"
        f"- Human review required: {len(review)}\n"
        f"- Standard 50M needed: {'YES' if standard_50m else 'NO'}\n"
        f"- Production Gate: {payload['production_gate']}\n"
        f"- Beta recommended: {'YES' if beta else 'NO'}\n"
        "- Production/Render/Vercel/Release changed: NO\n"
        "- Automatic training: NO\n",
        encoding="utf-8",
    )
    print(json.dumps({"v2.1": v21_summary, "v2.2": v22_summary, "blind": blind_summary,
                      "stress": stress_summary, "retrieval": retrieval,
                      "agreement": agreement, "review": len(review), "candidates": len(candidate_rows),
                      "production_gate": payload["production_gate"], "beta": beta}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
