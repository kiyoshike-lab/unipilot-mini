from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from typing import Any

from quality.campus_ai_judge import AXES, CAUSES, CampusAIJudge
from quality.campus_answer_improver import CampusAnswerImprover, FAQSourceStore


ROOT = Path(__file__).resolve().parents[1]
HUMAN_QUICK = ROOT / "evaluation/campus-v21-quick-human-ratings-snapshot.json"
HUMAN_100 = ROOT / "evaluation/human-comparison-campus-v21.json"
OUTPUT_20 = ROOT / "evaluation/campus-ai-quality-20.json"
OUTPUT_100 = ROOT / "evaluation/campus-ai-quality-100.json"
CLOSE_ANALYSIS = ROOT / "evaluation/campus-v21-close-analysis.json"
CRITICAL_FAILURE = ROOT / "evaluation/campus-v21-critical-failure.json"
REVIEW_QUEUE = ROOT / "evaluation/campus-ai-review-queue.json"
REPORT = ROOT / "evaluation/campus-ai-quality-report.md"

PRIMARY_CAUSE_PRIORITY = ("WRONG_PRIORITY", "ROUTER_ISSUE", "RETRIEVAL_ISSUE", "TOOL_ISSUE",
                          "PARTIAL_ANSWER", "NOT_ACTIONABLE", "TOO_GENERIC", "TOO_SHORT",
                          "MISSING_DETAIL", "WEAK_GROUNDING", "UNNATURAL", "UNCLEAR",
                          "MODEL_ISSUE", "OTHER")


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def grade_counts(records: list[dict], key: str) -> dict[str, int]:
    counter = Counter(record[key]["quality_label"] for record in records)
    return {label: counter[label] for label in ("good", "close", "bad")}


def average_scores(records: list[dict], key: str) -> dict[str, float]:
    return {axis: round(mean(record[key]["scores_0_to_5"][axis] for record in records), 3) for axis in AXES}


def primary_cause(issues: list[str]) -> str:
    return next((cause for cause in PRIMARY_CAUSE_PRIORITY if cause in issues), "OTHER")


def category_route_summary(records: list[dict]) -> tuple[dict, dict]:
    categories: dict[str, list[float]] = defaultdict(list)
    routes: dict[str, list[float]] = defaultdict(list)
    for record in records:
        categories[record["category"]].append(record["judge"]["overall_score"])
        routes[record["route"]].append(record["judge"]["overall_score"])
    summarise = lambda values: {key: {"count": len(scores), "average_score": round(mean(scores), 2)}
                                for key, scores in sorted(values.items())}
    return summarise(categories), summarise(routes)


def evaluate() -> dict[str, Any]:
    judge = CampusAIJudge(ROOT / "quality/campus_answer_rubric.json")
    sources = FAQSourceStore(ROOT / "data/campus_v2/faq/reviewed.jsonl")
    improver = CampusAnswerImprover(judge, sources)
    human_payload = load_json(HUMAN_QUICK)
    human_by_id = {item["item_id"]: item for item in human_payload["items"]}
    source_rows = load_json(HUMAN_100)
    source_by_id = {item["id"]: item for item in source_rows}
    generated_at = datetime.now(timezone.utc).isoformat()

    records_20 = []
    for item_id, human in human_by_id.items():
        row = source_by_id[item_id]
        metadata = row.get("campus_metadata", {})
        judge_metadata = {**metadata, "predicted_category": metadata.get("category"),
                          "category": row["category"]}
        documents = sources.documents(metadata)
        source_texts = [document["answer"] for document in documents]
        original_judge = judge.evaluate(row["question"], row["campus_answer"], judge_metadata, source_texts)
        improvement = improver.improve(row["question"], row["campus_answer"], judge_metadata,
                                       force=human.get("rating") in ("close", "bad"))
        records_20.append({"item_id": item_id, "question": row["question"], "category": row["category"],
                           "route": metadata.get("route"), "action": metadata.get("action"),
                           "source_ids": improvement["source_ids"], "human_rating": human.get("rating"),
                           "human_reason": human.get("reason"), "original_answer": row["campus_answer"],
                           "original_judge": original_judge, "critique": improvement["critique"],
                           "improved_answer": improvement["improved_answer"],
                           "improved_judge": improvement["after_judge"],
                           "rewrite_count": improvement["rewrite_count"]})

    human_counts = Counter(record["human_rating"] for record in records_20)
    agreement_items = [record for record in records_20 if record["human_rating"] == record["original_judge"]["quality_label"]]
    mismatches = [{"item_id": record["item_id"], "question": record["question"],
                   "human": record["human_rating"], "ai": record["original_judge"]["quality_label"],
                   "ai_score": record["original_judge"]["overall_score"],
                   "ai_issues": record["original_judge"]["issues"]}
                  for record in records_20 if record not in agreement_items]
    summary_20 = {
        "human": {label: human_counts[label] for label in ("good", "close", "bad")},
        "ai_original": grade_counts(records_20, "original_judge"),
        "ai_improved": grade_counts(records_20, "improved_judge"),
        "human_ai_agreement": {"matched": len(agreement_items), "total": len(records_20),
                               "rate": round(len(agreement_items) / len(records_20), 4),
                               "target": .70, "target_met": len(agreement_items) / len(records_20) >= .70,
                               "mismatches": mismatches},
        "average_score": {"original": round(mean(r["original_judge"]["overall_score"] for r in records_20), 2),
                          "improved": round(mean(r["improved_judge"]["overall_score"] for r in records_20), 2)},
        "average_characters": {"original": round(mean(len(r["original_answer"]) for r in records_20), 2),
                               "improved": round(mean(len(r["improved_answer"]) for r in records_20), 2)},
        "axis_averages_original": average_scores(records_20, "original_judge"),
        "axis_averages_improved": average_scores(records_20, "improved_judge"),
        "unsupported_claim_rate": {
            "original": round(mean(r["original_judge"]["unsupported_claim_rate"] for r in records_20), 4),
            "improved": round(mean(r["improved_judge"]["unsupported_claim_rate"] for r in records_20), 4),
        },
    }
    output_20 = {"schema_version": "campus-ai-quality-20-v1", "generated_at": generated_at,
                 "judge": "deterministic_local", "external_ai_api": "OFF",
                 "ai_quality_gate_is_human_gate_replacement": False,
                 "rc_answer_logic_changed": False, "production_changed": False,
                 "summary": summary_20, "items": records_20}
    write_json(OUTPUT_20, output_20)

    close_rows = []
    for record in records_20:
        if record["human_rating"] != "close":
            continue
        issues = record["original_judge"]["issues"]
        close_rows.append({"item_id": record["item_id"], "question": record["question"],
                           "original_answer": record["original_answer"], "ai_score": record["original_judge"]["overall_score"],
                           "primary_cause": primary_cause(issues), "all_causes": issues or ["OTHER"],
                           "critique": record["original_judge"]["critique"],
                           "improved_answer": record["improved_answer"]})
    close_counts = Counter(row["primary_cause"] for row in close_rows)
    close_payload = {"schema_version": "campus-v21-close-analysis-v1", "generated_at": generated_at,
                     "human_close_count": len(close_rows),
                     "cause_counts": dict(sorted(close_counts.items(), key=lambda item: (-item[1], item[0]))),
                     "allowed_causes": list(CAUSES), "items": close_rows}
    write_json(CLOSE_ANALYSIS, close_payload)

    bad_records = []
    for record in records_20:
        if record["human_rating"] != "bad":
            continue
        bad_records.append({"item_id": record["item_id"], "question": record["question"],
                            "answer": record["original_answer"], "route": record["route"],
                            "category": record["category"], "source": record["source_ids"],
                            "human_reason": record["human_reason"],
                            "failure_reasons": record["original_judge"]["issues"],
                            "critique": record["original_judge"]["critique"],
                            "improved_answer": record["improved_answer"],
                            "improved_score": record["improved_judge"]["overall_score"]})
    write_json(CRITICAL_FAILURE, {"schema_version": "campus-v21-critical-failure-v1",
                                  "generated_at": generated_at, "count": len(bad_records), "items": bad_records})

    records_100 = []
    for row in source_rows:
        metadata = row.get("campus_metadata", {})
        judge_metadata = {**metadata, "predicted_category": metadata.get("category"),
                          "category": row["category"]}
        documents = sources.documents(metadata)
        judged = judge.evaluate(row["question"], row["campus_answer"], judge_metadata,
                                [document["answer"] for document in documents])
        improvement = improver.improve(row["question"], row["campus_answer"], judge_metadata,
                                       force=judged["overall_score"] < 90)
        records_100.append({"item_id": row["id"], "question": row["question"], "category": row["category"],
                            "route": metadata.get("route"), "action": metadata.get("action"),
                            "source_ids": improvement["source_ids"], "original_answer": row["campus_answer"],
                            "judge": judged, "improved_answer": improvement["improved_answer"],
                            "improved_judge": improvement["after_judge"]})
    categories, routes = category_route_summary(records_100)
    labels_100 = Counter(record["judge"]["quality_label"] for record in records_100)
    averages_100 = {axis: round(mean(record["judge"]["scores_0_to_5"][axis] for record in records_100), 3)
                    for axis in AXES}
    summary_100 = {"total": len(records_100),
                   "ai_quality_gate": {label: labels_100[label] for label in ("good", "close", "bad")},
                   "average_score": round(mean(record["judge"]["overall_score"] for record in records_100), 2),
                   "axis_averages_0_to_5": averages_100,
                   "axis_percent": {axis: round(value / 5 * 100, 2) for axis, value in averages_100.items()},
                   "average_characters": round(mean(len(record["original_answer"]) for record in records_100), 2),
                   "unsupported_claim_rate": round(mean(record["judge"]["unsupported_claim_rate"] for record in records_100), 4),
                   "category_scores": categories, "route_scores": routes}
    write_json(OUTPUT_100, {"schema_version": "campus-ai-quality-100-v1", "generated_at": generated_at,
                            "judge": "deterministic_local", "external_ai_api": "OFF",
                            "ai_quality_gate_is_human_gate_replacement": False,
                            "rc_answer_logic_changed": False, "production_changed": False,
                            "summary": summary_100, "items": records_100})

    queue = []
    error_codes = {"ROUTER_ISSUE", "RETRIEVAL_ISSUE", "TOOL_ISSUE"}
    for record in records_100:
        judged = record["judge"]
        reasons = []
        if judged["quality_label"] == "bad": reasons.append("AI_LABEL_BAD")
        if judged["overall_score"] < 80: reasons.append("SCORE_BELOW_80")
        if judged["hallucination_suspected"]: reasons.append("HALLUCINATION_SUSPECTED")
        if judged["unsupported_claims"]: reasons.append("UNSUPPORTED_CLAIM")
        reasons.extend(issue for issue in judged["issues"] if issue in error_codes)
        if reasons:
            queue.append({"item_id": record["item_id"], "ai_judge_score": judged["overall_score"],
                          "question": record["question"], "category": record["category"], "route": record["route"],
                          "source_ids": record["source_ids"], "original_answer": record["original_answer"],
                          "problems": judged["issues"], "review_reasons": list(dict.fromkeys(reasons)),
                          "critique": judged["critique"], "improved_answer": record["improved_answer"],
                          "improved_score": record["improved_judge"]["overall_score"],
                          "review_status": "pending"})
    queue_payload = {"schema_version": "campus-ai-review-queue-v1", "generated_at": generated_at,
                     "selection_policy": ["AI label is bad", "score < 80", "hallucination suspected",
                                          "unsupported claim", "router/retrieval/tool issue"],
                     "total_source_items": 100, "review_required": len(queue),
                     "automatic_training": False, "production_changed": False, "items": queue}
    write_json(REVIEW_QUEUE, queue_payload)

    top5 = list(close_payload["cause_counts"].items())[:5]
    report_lines = ["# Campus AI Quality Report", "", "AI Quality Gate is not a replacement for Human Gate.", "",
                    "## Human 20", "", "- ◎: 4", "- △: 15", "- ×: 1", "",
                    "## AI Judge 20 (original)", "",
                    f"- ◎: {summary_20['ai_original']['good']}", f"- △: {summary_20['ai_original']['close']}",
                    f"- ×: {summary_20['ai_original']['bad']}",
                    f"- Human-AI agreement: {summary_20['human_ai_agreement']['rate'] * 100:.2f}%", "",
                    "## Before / After", "", f"- Average score: {summary_20['average_score']['original']:.2f} -> {summary_20['average_score']['improved']:.2f}",
                    f"- Average characters: {summary_20['average_characters']['original']:.2f} -> {summary_20['average_characters']['improved']:.2f}",
                    f"- Improved ◎/△/×: {summary_20['ai_improved']['good']} / {summary_20['ai_improved']['close']} / {summary_20['ai_improved']['bad']}", "",
                    "## Human △ primary causes TOP 5", ""]
    report_lines.extend(f"- {cause}: {count}" for cause, count in top5)
    report_lines.extend(["", "## AI Quality Gate 100", "",
                         f"- ◎/△/×: {summary_100['ai_quality_gate']['good']} / {summary_100['ai_quality_gate']['close']} / {summary_100['ai_quality_gate']['bad']}",
                         f"- Average score: {summary_100['average_score']:.2f}",
                         f"- Human review required: {len(queue)} / 100", "",
                         "## Protection", "", "- External LLM API: OFF", "- Existing RC answer logic changed: NO",
                         "- Production/Render/Vercel/Release changed: NO", "- Automatic training from approved memory: NO",
                         "- Standard 50M restart needed: NO (prioritize deterministic answer quality and reviewed data first)"])
    REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {"human": summary_20["human"], "ai_20": summary_20["ai_original"],
            "agreement": summary_20["human_ai_agreement"]["rate"],
            "before": summary_20["average_score"]["original"], "after": summary_20["average_score"]["improved"],
            "ai_100": summary_100["ai_quality_gate"], "review_required": len(queue)}


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
