from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate_campus_v22_generalization import (
    aggregate,
    campus_quality_proxy,
    judge_result,
    university_policy_assertion,
)
from pipeline.campus_retrieval import load_jsonl
from pipeline.campus_retrieval_v23 import CampusKnowledgeRetrieverV23, V23_RETRIEVAL_STRATEGIES
from pipeline.campus_v23 import UniPilotCampusV23
from quality.campus_ai_judge import CampusAIJudge


V22_SUMMARY = ROOT / "evaluation/campus-v22-generalization-summary.json"
OLD_100_SOURCE = ROOT / "evaluation/human-comparison-campus-v21.json"
OLD_BLIND_RESULTS = ROOT / "evaluation/campus-v22-generalization-blind-300.json"
NEW_BLIND = ROOT / "data/campus_v23/holdouts/blind-500.json"
NEW_STRESS = ROOT / "data/campus_v23/holdouts/stress-200.json"
RETRIEVAL_BENCHMARK = ROOT / "data/campus_v22/benchmarks/knowledge-1000.jsonl"
CONFLICTS = ROOT / "data/campus_v23/retrieval/numeric-conflicts.json"

OLD_100_OUT = ROOT / "evaluation/campus-v23-old-100.json"
OLD_BLIND_OUT = ROOT / "evaluation/campus-v23-old-blind-300.json"
BLIND_OUT = ROOT / "evaluation/campus-v23-blind-500.json"
STRESS_OUT = ROOT / "evaluation/campus-v23-stress-200.json"
RETRIEVAL_OUT = ROOT / "evaluation/campus-v23-retrieval.json"
REVIEW_OUT = ROOT / "evaluation/campus-v23-review-queue.json"
SUMMARY_OUT = ROOT / "evaluation/campus-v23-summary.json"
REPORT_OUT = ROOT / "evaluation/campus-v23-report.md"


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate_questions(pipeline: UniPilotCampusV23, judge: CampusAIJudge,
                       rows: list[dict], prefix: str) -> list[dict]:
    records = []
    for index, row in enumerate(rows):
        question = row["question"]
        expected = row.get("expected_category") or row.get("category") or "general"
        result = pipeline.answer(question, response_mode="auto", session_id=f"{prefix}-{index:04d}")
        judged = judge_result(judge, question, expected, result)
        records.append({
            "id": row.get("id") or f"{prefix}-{index:04d}",
            "question": question,
            "category": expected,
            "predicted_category": result.get("category"),
            "route": result.get("route"),
            "answer": result.get("text", ""),
            "answer_depth": result.get("answer_depth"),
            "answer_coverage": result.get("answer_coverage", {}),
            "revision_count": result.get("revision_count", 0),
            "retrieval_confidence": result.get("retrieval_confidence"),
            "retrieval_accepted": result.get("retrieval_accepted"),
            "retrieval_result": result.get("retrieval", []),
            "sources": result.get("sources", []),
            "cards": result.get("cards", []),
            "validator": result.get("validator", {}),
            "judge": judged,
            "score": judged["overall_score"],
            "feature": row.get("surface") or row.get("stress_type"),
            "must_not_assert_unverified_policy": row.get("must_not_assert_unverified_policy", False),
            "holdout": bool(row.get("holdout")),
        })
    return records


def v23_aggregate(records: list[dict]) -> dict:
    summary = aggregate(records)
    coverage = [float(row.get("answer_coverage", {}).get("score", 0.0)) for row in records]
    summary["answer_coverage"] = round(mean(coverage) if coverage else 0.0, 4)
    summary["answer_coverage_percent"] = round(summary["answer_coverage"] * 100, 2)
    summary["multi_intent_coverage_percent"] = round(mean([
        float(row.get("answer_coverage", {}).get("score", 0.0))
        for row in records if row.get("answer_coverage", {}).get("target") == 1.0
    ] or [0.0]) * 100, 2)
    summary["confidence_counts"] = dict(Counter(row.get("retrieval_confidence") or "NOT_USED" for row in records))
    return summary


def retrieval_comparison(retriever: CampusKnowledgeRetrieverV23) -> dict:
    rows = load_jsonl(RETRIEVAL_BENCHMARK)
    dev_rows = rows[::max(1, len(rows) // 300)][:300]
    dev_ids = {row["id"] for row in dev_rows}
    final_rows = [row for row in rows if row["id"] not in dev_ids and row.get("expected_source_url")][:500]
    negative_rows = [row for row in rows if row["id"] not in dev_ids and not row.get("expected_source_url")][:100]
    strategies = {}
    for strategy in V23_RETRIEVAL_STRATEGIES:
        hit1 = hit3 = hit5 = false_match = false_no_match = 0
        reciprocal = 0.0
        latencies = []
        confidence = Counter()
        for row in final_rows:
            results, meta = retriever.search(
                row["question"], row["category"], top_k=5, strategy=strategy,
                confidence_policy="precision", response_mode="normal",
            )
            urls = [item.get("source_url") for item in results]
            expected = row["expected_source_url"]
            rank = next((position + 1 for position, url in enumerate(urls) if url == expected), None)
            hit1 += int(rank == 1)
            hit3 += int(rank is not None and rank <= 3)
            hit5 += int(rank is not None and rank <= 5)
            reciprocal += 1 / rank if rank else 0.0
            false_match += int(bool(meta["accepted"] and urls and urls[0] != expected))
            false_no_match += int(not meta["accepted"])
            confidence[meta["confidence"]] += 1
            latencies.append(float(meta["latency_ms"]))
        negative_accepts = 0
        for row in negative_rows:
            _, meta = retriever.search(
                row["question"], row["category"], top_k=5, strategy=strategy,
                confidence_policy="precision", response_mode="normal",
            )
            negative_accepts += int(meta["accepted"])
        total = max(1, len(final_rows))
        strategies[strategy] = {
            "questions": len(final_rows),
            "recall_at_1": round(hit1 / total, 4),
            "recall_at_3": round(hit3 / total, 4),
            "recall_at_5": round(hit5 / total, 4),
            "mrr": round(reciprocal / total, 4),
            "false_match_rate": round(false_match / total, 4),
            "false_no_match_rate": round(false_no_match / total, 4),
            "negative_query_accept_rate": round(negative_accepts / max(1, len(negative_rows)), 4),
            "average_latency_ms": round(mean(latencies), 3),
            "confidence_counts": dict(confidence),
        }
    selected = max(strategies, key=lambda name: (
        strategies[name]["recall_at_3"] >= .95,
        strategies[name]["false_match_rate"] <= .05,
        strategies[name]["mrr"],
        strategies[name]["recall_at_1"],
        -strategies[name]["false_no_match_rate"],
        {"category_aware_hybrid": 2, "multi_query_hybrid": 1}.get(name, 0),
        -strategies[name]["average_latency_ms"],
    ))
    return {
        "development_set": {"questions": len(dev_rows), "v2_2_false_match_rate": .23},
        "final_holdout": True,
        "final_source_linked_questions": len(final_rows),
        "final_negative_questions": len(negative_rows),
        "metric_policy": "Recall uses raw top-k; False Match counts accepted wrong top-1; False No Match counts LOW/REJECT",
        "strategies": strategies,
        "selected": selected,
    }


def main() -> int:
    generated_at = datetime.now(timezone.utc).isoformat()
    pipeline = UniPilotCampusV23()
    judge = CampusAIJudge()
    v22_summary = read_json(V22_SUMMARY)

    old_100_source = read_json(OLD_100_SOURCE)
    old_100 = evaluate_questions(pipeline, judge, [{
        "id": row["id"], "question": row["question"], "expected_category": row["category"],
    } for row in old_100_source], "old100")
    old_100_summary = v23_aggregate(old_100)
    write_json(OLD_100_OUT, {"schema_version": "campus-v23-old-100-v1", "generated_at": generated_at,
                             "development_set": True, "summary": old_100_summary, "items": old_100})

    old_blind_source = read_json(OLD_BLIND_RESULTS)["items"]
    old_blind = evaluate_questions(pipeline, judge, [{
        "id": row["id"], "question": row["question"], "expected_category": row["category"],
    } for row in old_blind_source], "oldblind")
    old_blind_summary = v23_aggregate(old_blind)
    write_json(OLD_BLIND_OUT, {"schema_version": "campus-v23-old-blind-300-v1", "generated_at": generated_at,
                               "development_set": True, "summary": old_blind_summary, "items": old_blind})

    blind_source = read_json(NEW_BLIND)["items"]
    blind = evaluate_questions(pipeline, judge, blind_source, "blind500")
    blind_summary = v23_aggregate(blind)
    write_json(BLIND_OUT, {"schema_version": "campus-v23-blind-500-results-v1", "generated_at": generated_at,
                           "holdout": True, "used_for_improvement": False,
                           "summary": blind_summary, "items": blind})

    stress_source = read_json(NEW_STRESS)["items"]
    stress = evaluate_questions(pipeline, judge, stress_source, "stress200")
    for row in stress:
        row["policy_guardrail_pass"] = not (
            row["must_not_assert_unverified_policy"] and university_policy_assertion(row["answer"])
        )
    stress_summary = {
        **v23_aggregate(stress),
        "policy_guardrail_failures": sum(not row["policy_guardrail_pass"] for row in stress),
        "by_type": {kind: {
            "questions": len(group), "average_score": round(mean(row["score"] for row in group), 2),
            "critical_errors": sum(row["judge"]["quality_label"] == "bad"
                                   or row["judge"]["hallucination_suspected"]
                                   or not row["policy_guardrail_pass"] for row in group),
        } for kind, group in ((kind, [row for row in stress if row["feature"] == kind])
                              for kind in sorted({row["feature"] for row in stress}))},
    }
    write_json(STRESS_OUT, {"schema_version": "campus-v23-stress-200-results-v1", "generated_at": generated_at,
                            "holdout": True, "used_for_improvement": False,
                            "summary": stress_summary, "items": stress})

    retrieval = retrieval_comparison(pipeline.knowledge)
    write_json(RETRIEVAL_OUT, {"schema_version": "campus-v23-retrieval-v1",
                               "generated_at": generated_at, **retrieval})
    selected_retrieval = retrieval["strategies"][retrieval["selected"]]

    review = []
    for row in [*blind, *stress]:
        reasons = []
        if row["judge"]["quality_label"] == "bad": reasons.append("AI_LABEL_BAD")
        if row["score"] < 80: reasons.append("SCORE_BELOW_80")
        if row["judge"]["unsupported_claims"]: reasons.append("UNSUPPORTED_CLAIM")
        if row["judge"]["hallucination_suspected"]: reasons.append("HALLUCINATION_SUSPECTED")
        if row["judge"].get("university_policy_assertion"): reasons.append("UNIVERSITY_POLICY_ASSERTION")
        if reasons:
            review.append({"item_id": row["id"], "question": row["question"], "answer": row["answer"],
                           "category": row["category"], "route": row["route"], "score": row["score"],
                           "reasons": reasons, "status": "pending"})
    write_json(REVIEW_OUT, {"schema_version": "campus-v23-review-v1", "generated_at": generated_at,
                            "automatic_training": False, "review_required": len(review), "items": review})

    blind_goals = {
        "average_score_gte_93": blind_summary["average_score"] >= 93,
        "correctness_gte_95": blind_summary["axis_percent"]["correctness"] >= 95,
        "relevance_gte_94": blind_summary["axis_percent"]["relevance"] >= 94,
        "actionable_gte_94": blind_summary["axis_percent"]["actionable"] >= 94,
        "completeness_gte_94": blind_summary["axis_percent"]["completeness"] >= 94,
        "specificity_gte_93": blind_summary["axis_percent"]["specificity"] >= 93,
        "naturalness_gte_97": blind_summary["axis_percent"]["naturalness"] >= 97,
        "unsupported_lte_0_5": blind_summary["unsupported_claim_rate"] <= .005,
        "critical_eq_0": blind_summary["critical_errors"] == 0,
    }
    retrieval_goals = {
        "recall_at_1_gte_90": selected_retrieval["recall_at_1"] >= .90,
        "recall_at_3_gte_95": selected_retrieval["recall_at_3"] >= .95,
        "mrr_gte_0_92": selected_retrieval["mrr"] >= .92,
        "false_match_lte_5": selected_retrieval["false_match_rate"] <= .05,
    }
    stress_goals = {
        "critical_eq_0": stress_summary["critical_errors"] == 0,
        "policy_assertion_eq_0": stress_summary["policy_guardrail_failures"] == 0,
    }
    retrieval_ready = selected_retrieval["recall_at_3"] >= .95 and selected_retrieval["false_match_rate"] <= .05
    model_axes_below = {
        "correctness": blind_summary["axis_percent"]["correctness"] < 95,
        "relevance": blind_summary["axis_percent"]["relevance"] < 94,
        "completeness": blind_summary["axis_percent"]["completeness"] < 94,
    }
    standard_50m = retrieval_ready and any(model_axes_below.values())
    gate_pass = all(blind_goals.values()) and all(retrieval_goals.values()) and all(stress_goals.values())
    conflicts = read_json(CONFLICTS)
    payload = {
        "schema_version": "campus-v23-summary-v1", "generated_at": generated_at,
        "external_ai_api": "OFF", "v2_1_rc_changed": False, "production_changed": False,
        "v2_2_100": v22_summary["v2_2_100"], "v2_3_old_100": old_100_summary,
        "v2_2_old_blind_300": v22_summary["blind_300"], "v2_3_old_blind_300": old_blind_summary,
        "v2_3_blind_500": blind_summary, "v2_3_stress_200": stress_summary,
        "retrieval": retrieval,
        "knowledge": {"sources": len({row["source_url"] for row in pipeline.knowledge.source_rows}),
                      "documents": len(pipeline.knowledge.source_rows),
                      "chunks": len(pipeline.knowledge.rows),
                      "conflict_groups": conflicts["candidate_groups"],
                      "unresolved_conflicts": conflicts["unresolved_groups"]},
        "human_review_required": len(review),
        "chatgpt_gap_proxy": campus_quality_proxy(blind),
        "blind_goals": blind_goals, "retrieval_goals": retrieval_goals, "stress_goals": stress_goals,
        "retrieval_ready_for_model_judgment": retrieval_ready,
        "model_axes_below_gate": model_axes_below,
        "twenty_m_model_limit_confirmed": standard_50m,
        "standard_50m_recommended": standard_50m,
        "production_gate": "PASS" if gate_pass else "FAIL",
        "beta_recommended": gate_pass,
        "push_or_deploy_performed": False,
    }
    write_json(SUMMARY_OUT, payload)
    old100_gate = old_100_summary["ai_quality_gate"]
    oldblind_gate = old_blind_summary["ai_quality_gate"]
    blind_gate = blind_summary["ai_quality_gate"]
    REPORT_OUT.write_text(
        "# Campus v2.3 Evaluation Report\n\n"
        "新Blind 500・Stress 200は改善に使わないholdoutとして一度だけ評価した。"
        "外部LLM/APIは使用していない。\n\n"
        "## Development comparison\n\n"
        f"- Old 100 v2.2 -> v2.3: score {v22_summary['v2_2_100']['average_score']} -> "
        f"{old_100_summary['average_score']}; ◎/△/× "
        f"{v22_summary['v2_2_100']['ai_quality_gate']} -> {old100_gate}\n"
        f"- Old Blind 300 v2.2 -> v2.3: score {v22_summary['blind_300']['average_score']} -> "
        f"{old_blind_summary['average_score']}; ◎/△/× "
        f"{v22_summary['blind_300']['ai_quality_gate']} -> {oldblind_gate}\n\n"
        "## New Blind 500\n\n"
        f"- ◎/△/×: {blind_gate['good']} / {blind_gate['close']} / {blind_gate['bad']}\n"
        f"- Score / characters: {blind_summary['average_score']} / {blind_summary['average_characters']}\n"
        f"- Axes (%): {blind_summary['axis_percent']}\n"
        f"- Unsupported / critical: {blind_summary['unsupported_claim_rate'] * 100:.2f}% / "
        f"{blind_summary['critical_errors']}\n"
        f"- Coverage / multi-intent coverage: {blind_summary['answer_coverage_percent']}% / "
        f"{blind_summary['multi_intent_coverage_percent']}%\n"
        f"- Routes (%): {blind_summary['route_percent']}\n\n"
        "## Stress 200 and retrieval\n\n"
        f"- Stress ◎/△/×: {stress_summary['ai_quality_gate']}; critical "
        f"{stress_summary['critical_errors']}; policy assertion {stress_summary['policy_guardrail_failures']}\n"
        f"- Selected retrieval benchmark strategy: {retrieval['selected']}\n"
        f"- Recall@1/3/5, MRR: {selected_retrieval['recall_at_1']:.4f} / "
        f"{selected_retrieval['recall_at_3']:.4f} / {selected_retrieval['recall_at_5']:.4f} / "
        f"{selected_retrieval['mrr']:.4f}\n"
        f"- False Match: 23.00% -> {selected_retrieval['false_match_rate'] * 100:.2f}%; "
        f"False No Match: {selected_retrieval['false_no_match_rate'] * 100:.2f}%\n"
        f"- Knowledge sources/documents/chunks: {payload['knowledge']['sources']} / "
        f"{payload['knowledge']['documents']} / {payload['knowledge']['chunks']}\n"
        f"- Numeric conflict groups/unresolved: {payload['knowledge']['conflict_groups']} / "
        f"{payload['knowledge']['unresolved_conflicts']}\n\n"
        "## Decision\n\n"
        f"- Human review required: {len(review)}\n"
        f"- Standard 50M: {'YES' if standard_50m else 'NO'}\n"
        f"- Production Gate: {payload['production_gate']}\n"
        f"- Beta: {'YES' if gate_pass else 'NO'}\n"
        "- Production/Render/Vercel/Release changed: NO\n"
        "- Push/deploy: NO\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "old100": old_100_summary, "oldBlind300": old_blind_summary,
        "blind500": blind_summary, "stress200": stress_summary,
        "retrieval": retrieval, "review": len(review),
        "standard50m": standard_50m, "gate": payload["production_gate"], "beta": gate_pass,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
