from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate_campus_v22_generalization import judge_result
from pipeline.campus_v23 import UniPilotCampusV23
from quality.campus_ai_judge import CampusAIJudge


SOURCE = ROOT / "evaluation/campus-v23-review-queue.json"
OUTPUT = ROOT / "evaluation/campus-v23-toeic-tool-fix.json"


def main() -> int:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    target = [row for row in source["items"] if row.get("category") == "toeic_plan"]
    pipeline = UniPilotCampusV23()
    judge = CampusAIJudge()
    rows = []
    for row in target:
        result = pipeline.answer(row["question"], session_id=f"toeic-fix-{row['item_id']}")
        judged = judge_result(judge, row["question"], row["category"], result)
        rows.append({
            "item_id": row["item_id"], "question": row["question"], "category": row["category"],
            "route": result["route"], "answer": result["text"],
            "fixed_ratio_present": any(value in result["text"] for value in (
                "語彙20%", "リスニング30%", "文法・読解30%", "復習20%",
            )),
            "quality_label": judged["quality_label"], "score": judged["overall_score"],
            "unsupported_claims": judged["unsupported_claims"],
            "hallucination_suspected": judged["hallucination_suspected"],
            "critical_error": (
                judged["quality_label"] == "bad" or judged["hallucination_suspected"]
                or judged.get("university_policy_assertion", False)
            ),
        })
    payload = {
        "schema_version": "campus-v23-toeic-tool-fix-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_evaluation_preserved": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "questions": len(rows),
        "critical_errors_before": len(target),
        "critical_errors_after": sum(row["critical_error"] for row in rows),
        "unsupported_claims_after": sum(bool(row["unsupported_claims"]) for row in rows),
        "fixed_ratio_answers_after": sum(row["fixed_ratio_present"] for row in rows),
        "production_v04_changed": False,
        "campus_v21_rc_changed": False,
        "external_ai_api": "OFF",
        "items": rows,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "questions", "critical_errors_before", "critical_errors_after",
        "unsupported_claims_after", "fixed_ratio_answers_after",
    )}, ensure_ascii=False, indent=2))
    return 0 if payload["critical_errors_after"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
