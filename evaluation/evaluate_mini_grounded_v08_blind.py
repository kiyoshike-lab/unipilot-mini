from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics

from inference.generate import load_model
from pipeline.v07 import V07Pipeline
from training.train_v08 import PROBE_STOP_BIGRAMS


CATEGORY_MAP = {
    "exam": "exam", "assignment": "assignment", "credit": "credit", "gpa": "gpa",
    "registration": "registration", "attendance": "attendance", "lateness": "lateness",
    "professor_email": "professor_email", "report": "report", "citation": "citation",
    "presentation": "presentation", "seminar": "campus_life", "laboratory": "campus_life",
    "thesis": "report", "career": "career", "internship": "internship", "qualification": "career",
    "toeic": "career", "study_abroad": "career", "scholarship": "scholarship", "tuition": "scholarship",
    "part_time": "campus_life", "campus_life": "campus_life", "relationships": "campus_life",
    "time_management": "study", "study": "study", "pc": "campus_life", "programming": "programming",
    "ai_usage": "ai_usage", "information_literacy": "citation", "statistics": "statistics",
    "math": "math", "general_education": "general",
}


def expected_hits(text: str, points: list[str]) -> int:
    fragments = {fragment for point in points for fragment in
                 (point[index:index + 2] for index in range(max(0, len(point) - 1)))
                 if fragment not in PROBE_STOP_BIGRAMS and re.search(r"[ぁ-んァ-ヶー一-龥々A-Za-z0-9]", fragment)}
    return sum(fragment in text for fragment in fragments)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-model", action="store_true",
                        help="Matched ablation: keep the shared category router, but disable retrieval, validation, and fallback.")
    args = parser.parse_args()
    blind = json.loads(Path("data/v08/blind/evaluation.json").read_text(encoding="utf-8"))
    model, tokenizer, _, payload = load_model(args.checkpoint, "tokenizer/vocab-v02-512.json", "cpu")
    pipeline = V07Pipeline(model, tokenizer, top_k=1)
    rows = []
    for item in blind:
        result = pipeline.answer(
            item["prompt"], 64, temperature=0.0, top_k=40, top_p=0.9, repetition_penalty=1.1,
            use_retrieval=not args.raw_model, use_validator=not args.raw_model, force_model=args.raw_model,
        )
        text = result["text"]
        relevant = expected_hits(text, item["expected_key_points"]) >= 4
        category_correct = result["category"] == CATEGORY_MAP[item["category"]]
        issues = result["validator"]["issues"]
        hallucination = any(issue == "university_specific_hallucination" or issue.startswith("invented_subject") for issue in issues)
        complete = len(text.strip()) >= 12
        rows.append({
            "id": item["id"], "expected_category": item["category"], "mapped_category": CATEGORY_MAP[item["category"]],
            "predicted_category": result["category"], "text": text, "relevant": relevant,
            "category_correct": category_correct, "correct": relevant and category_correct and not hallucination,
            "natural": bool(text.strip()) and "�" not in text, "complete": complete,
            "effective_eos": result["grounded_selected"] or result["generation_metrics"]["eos_reached"],
            "hallucination": hallucination, "fallback": result["fallback_used"],
            "grounded_selected": result["grounded_selected"], "seconds": result["timing"]["total_seconds"],
        })
        if len(rows) % 25 == 0 or len(rows) == len(blind):
            print(f"evaluated {len(rows)}/{len(blind)}", flush=True)
    count = len(rows)
    rate = lambda key: sum(row[key] for row in rows) / count
    report = {
        "label": args.label, "checkpoint": args.checkpoint, "parameters": model.parameter_count(),
        "vocab": tokenizer.vocab_size, "context": model.config.context_length, "step": payload.get("step"),
        "blind_dataset": "data/v08/blind/evaluation.json",
        "retrieval_enabled": not args.raw_model, "validator_enabled": not args.raw_model,
        "matched_ablation_note": (
            "Raw-model control retains the same question category router and prompt wrapper so the retrieval effect is isolated."
            if args.raw_model else None
        ),
        "metrics": {
            "questions": count, "relevance_rate": rate("relevant"), "correctness_rate": rate("correct"),
            "category_accuracy_mapped_21": rate("category_correct"), "natural_japanese_rate": rate("natural"),
            "completion_rate": rate("complete"), "effective_eos_rate": rate("effective_eos"),
            "hallucination_rate": rate("hallucination"), "fallback_rate": rate("fallback"),
            "grounded_selection_rate": rate("grounded_selected"),
            "mean_total_seconds": statistics.fmean(row["seconds"] for row in rows),
        },
        "mapping_limit": "The Standard 33 categories are mapped to the older Mini 21-category taxonomy.",
        "generations": rows, "external_ai_api": "OFF",
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**report, "generations": f"{len(rows)} rows written"}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
