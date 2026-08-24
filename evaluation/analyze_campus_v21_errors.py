from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from pipeline.campus_categories_v2 import CATEGORY_TO_LEVEL1
from pipeline.campus_retrieval import load_jsonl
from pipeline.campus_router_v2 import CampusRouterV2
from pipeline.campus_router_v21 import CampusRouterV21, parse_negation_contrast


ROOT = Path(__file__).resolve().parents[1]
FAILURE_CLASSES = ("AMBIGUOUS", "NEGATION", "CONTRAST", "SHORT_QUERY", "CATEGORY_COLLISION",
                   "RETRIEVAL_FAILURE", "TOOL_SELECTION", "MULTI_INTENT", "UNKNOWN", "OTHER")


def failure_reason(item: dict, predicted: str, action: str, record: dict | None = None) -> str:
    question = item.get("question") or item["prompt"]
    expected = item["category"]
    if item.get("ambiguous") or item.get("surface_type") == "ambiguous":
        return "AMBIGUOUS"
    parsed = parse_negation_contrast(question)
    if parsed:
        return "NEGATION" if parsed["kind"] == "NEGATION" else "CONTRAST"
    if len(item.get("intent_labels", ())) > 1:
        return "MULTI_INTENT"
    if len(question.replace(" ", "")) <= 18:
        return "SHORT_QUERY"
    if predicted != expected and CATEGORY_TO_LEVEL1.get(predicted) == CATEGORY_TO_LEVEL1.get(expected):
        return "CATEGORY_COLLISION"
    if record and record.get("retrieval_status") == "wrong":
        return "RETRIEVAL_FAILURE"
    if action != item.get("expected_action"):
        return "TOOL_SELECTION"
    if predicted != expected:
        return "UNKNOWN"
    return "OTHER"


def score_summary(router, question: str, decision) -> tuple[float, float, float]:
    features = decision.evidence.get("ambiguity_features")
    if features:
        return features["top1_score"], features["top2_score"], features["score_margin"]
    scores, _ = router._combined_scores(question)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return scores[ordered[0]], scores[ordered[1]], scores[ordered[0]] - scores[ordered[1]]


def analyze(name: str, router, rows: list[dict], pipeline_records: dict[str, dict] | None = None) -> dict:
    failures = []
    reason_counts: Counter[str] = Counter()
    confusion: Counter[str] = Counter()
    for item in rows:
        question = item.get("question") or item["prompt"]
        decision = router.decide(question)
        ambiguous = bool(item.get("ambiguous") or item.get("surface_type") == "ambiguous")
        routing_success = decision.action == "CLARIFY" if ambiguous else decision.primary == item["category"]
        action_success = decision.action == item.get("expected_action")
        record = (pipeline_records or {}).get(item["id"])
        answer_success = True if record is None else bool(record["answer_correct"] and record["relevance"])
        if routing_success and action_success and answer_success:
            continue
        top1_score, top2_score, margin = score_summary(router, question, decision)
        reason = failure_reason(item, decision.primary, decision.action, record)
        reason_counts[reason] += 1
        confusion[f"{item['category']} -> {decision.primary}"] += 1
        failures.append({
            "id": item["id"], "question": question, "gold": item["category"],
            "predicted": decision.primary, "top1": decision.top2[0] if decision.top2 else decision.primary,
            "top2": decision.top2[1] if len(decision.top2) > 1 else None,
            "top1_score": round(top1_score, 6), "top2_score": round(top2_score, 6),
            "margin": round(margin, 6), "action": decision.action,
            "expected_action": item.get("expected_action"), "ambiguous": ambiguous,
            "routing_success": routing_success, "action_success": action_success,
            "answer_success": answer_success, "reason": reason,
        })
    all_counts = {key: reason_counts.get(key, 0) for key in FAILURE_CLASSES}
    return {
        "variant": name, "questions": len(rows), "failures": len(failures),
        "failure_rate": len(failures) / len(rows), "reason_counts": all_counts,
        "reason_percent_of_failures": {key: (value / len(failures) if failures else 0.0)
                                        for key, value in all_counts.items()},
        "top_confusions": [{"pair": key, "count": count} for key, count in confusion.most_common(10)],
        "all_failures": failures,
    }


def main() -> None:
    base = load_jsonl(ROOT / "data/campus_v2/router/train.jsonl")
    adversarial_train = load_jsonl(ROOT / "data/campus_v21/router/adversarial-train-1500.jsonl")
    blind = json.loads((ROOT / "data/campus_v2/blind/evaluation-2000.json").read_text(encoding="utf-8"))
    adversarial_test = json.loads((ROOT / "data/campus_v2/adversarial/negation-300.json").read_text(encoding="utf-8"))
    v2 = CampusRouterV2(base)
    v21 = CampusRouterV21(base + adversarial_train)
    benchmark_path = ROOT / "evaluation/campus-v21-benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8")) if benchmark_path.exists() else None
    record_maps = {}
    if benchmark:
        for row in benchmark["comparisons"]["blind_2000"]:
            record_maps[(row["variant"], "blind")] = {item["id"]: item for item in row["records"]}
        for row in benchmark["adversarial_300"]:
            record_maps[(row["variant"], "adversarial")] = {item["id"]: item for item in row["records"]}
    output = {
        "blind_test_boundary": "Campus v2 blind 2000 is evaluation-only; no row was added to training or threshold validation.",
        "existing_adversarial_boundary": "Existing adversarial 300 is test-only and absent from v2.1 train/validation.",
        "blind_2000": [analyze("Campus v2", v2, blind, record_maps.get(("Campus v2", "blind"))),
                       analyze("Campus v2.1", v21, blind, record_maps.get(("Campus v2.1", "blind")))],
        "adversarial_300": [analyze("Campus v2", v2, adversarial_test,
                                    record_maps.get(("Campus v2", "adversarial"))),
                            analyze("Campus v2.1", v21, adversarial_test,
                                    record_maps.get(("Campus v2.1", "adversarial")))],
        "external_ai_api": "OFF",
    }
    path = ROOT / "evaluation/campus-v21-full-failure-analysis.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"blind": [(row["variant"], row["failures"]) for row in output["blind_2000"]],
                      "adversarial": [(row["variant"], row["failures"]) for row in output["adversarial_300"]]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
