from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from pipeline.campus_retrieval import load_jsonl
from pipeline.campus_router import CampusHybridRouter


ROOT = Path(__file__).resolve().parents[1]


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def failure_pattern(expected: str, predicted: str, confidence: float) -> str:
    email = {"professor_email", "absence_email", "lateness_email", "late_submission_email"}
    planning = {"schedule", "study_plan", "assignment_priority", "deadline_organizer"}
    if expected in email and predicted in email | {"attendance", "lateness", "assignment"}:
        return "email-vs-event boundary"
    if expected in planning and predicted in planning | {"assignment", "exam"}:
        return "planning subcategory boundary"
    if expected in {"gpa", "credit", "grade_simulator"} and predicted in {"gpa", "credit", "grade_simulator"}:
        return "grade/credit calculation boundary"
    if confidence < 0.58:
        return "low-confidence nearest-document jump"
    return "lexical collision or missing expression"


def improvement(pattern: str) -> str:
    return {
        "email-vs-event boundary": "Separate communication intent at Level1 and require an email/compose signal.",
        "planning subcategory boundary": "Use the requested output (plan, order, deadline list) as a Level2 feature.",
        "grade/credit calculation boundary": "Use calculation operands and definition-vs-action signals before topic words.",
        "low-confidence nearest-document jump": "Clarify instead of forcing the nearest category; calibrate confidence on dev.",
        "lexical collision or missing expression": "Add the expression to router-only data and use character n-grams.",
    }[pattern]


def main() -> None:
    results = json.loads((ROOT / "evaluation" / "results-campus-v1-d.json").read_text(encoding="utf-8"))
    blind = json.loads((ROOT / "data" / "campus_v1" / "blind" / "evaluation.json").read_text(encoding="utf-8"))
    questions = {row["id"]: row["prompt"] for row in blind}
    router = CampusHybridRouter(load_jsonl(ROOT / "data" / "campus_v1" / "router" / "train.jsonl"))
    labels = sorted({row["expected_category"] for row in results["generations"]} |
                    {row["predicted_category"] for row in results["generations"]})
    confusion = {expected: {predicted: 0 for predicted in labels} for expected in labels}
    failed = []
    for item in results["generations"]:
        expected, predicted = item["expected_category"], item["predicted_category"]
        confusion[expected][predicted] += 1
        if expected == predicted:
            continue
        _, confidence, evidence = router.predict(questions[item["id"]])
        pattern = failure_pattern(expected, predicted, confidence)
        failed.append({
            "id": item["id"], "question": questions[item["id"]], "expected": expected,
            "predicted": predicted, "confidence": confidence,
            "failure_pattern": pattern, "proposed_improvement": improvement(pattern),
            "router_source": evidence.get("source"),
        })
    per_category = {}
    for label in labels:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted_count = sum(confusion[expected][label] for expected in labels)
        precision, recall = safe_div(tp, predicted_count), safe_div(tp, support)
        per_category[label] = {
            "precision": precision, "recall": recall,
            "f1": safe_div(2 * precision * recall, precision + recall), "support": support,
        }
    patterns = Counter(row["failure_pattern"] for row in failed)
    output = {
        "source_result": "evaluation/results-campus-v1-d.json", "questions": len(results["generations"]),
        "correct": len(results["generations"]) - len(failed), "incorrect": len(failed),
        "accuracy": 1 - len(failed) / len(results["generations"]),
        "confusion_matrix": {"labels": labels, "rows": confusion}, "per_category": per_category,
        "failure_patterns": dict(patterns.most_common()), "failed_rows": failed,
        "external_ai_api": "OFF",
    }
    path = ROOT / "evaluation" / "campus-v1-router-error-analysis.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"accuracy": output["accuracy"], "incorrect": len(failed),
                      "patterns": output["failure_patterns"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

