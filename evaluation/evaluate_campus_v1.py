from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics
import time

import psutil

from inference.generate import load_model
from pipeline.campus_categories import CAMPUS_KEYWORDS
from pipeline.campus_v1 import UniPilotCampusV1
from pipeline.v07 import V07Pipeline
from training.train_v08 import PROBE_STOP_BIGRAMS


MINI_CATEGORY = {
    "exam": "exam", "assignment": "assignment", "credit": "credit", "gpa": "gpa",
    "grade_simulator": "credit", "attendance": "attendance", "lateness": "lateness",
    "professor_email": "professor_email", "absence_email": "professor_email",
    "lateness_email": "professor_email", "late_submission_email": "professor_email",
    "registration": "registration", "schedule": "study", "study_plan": "study",
    "assignment_priority": "assignment", "deadline_organizer": "assignment",
    "report_outline": "report", "citation_check": "citation", "presentation_outline": "presentation",
    "career_schedule": "career", "es_outline": "career", "toeic_plan": "career",
    "internship": "internship", "scholarship": "scholarship", "tuition": "scholarship",
    "part_time_job": "campus_life", "campus_life": "campus_life", "relationship": "campus_life",
    "programming": "programming", "ai_usage": "ai_usage", "math": "math", "statistics": "statistics",
    "university_policy": "general", "faq_search": "general", "general": "general",
}
FORBIDDEN = ("どの大学でも", "全国の大学で", "必ず認められる", "必ず追試", "必ず合格")


def keypoint_hits(answer: str, points: list[str]) -> int:
    fragments = {fragment for point in points for fragment in
                 (point[index:index + 2] for index in range(max(0, len(point) - 1)))
                 if fragment not in PROBE_STOP_BIGRAMS and re.search(r"[ぁ-んァ-ヶー一-龥々A-Za-z0-9]", fragment)}
    return sum(fragment in answer for fragment in fragments)


def natural(answer: str) -> bool:
    return bool(answer.strip()) and "�" not in answer and not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", answer)


def unsafe_forbidden_claim(answer: str) -> bool:
    for phrase in FORBIDDEN:
        start = answer.find(phrase)
        while start >= 0:
            tail = answer[start + len(phrase):start + len(phrase) + 14]
            if not any(negation in tail for negation in ("ない", "ません", "とは限", "でない", "と断定")):
                return True
            start = answer.find(phrase, start + 1)
    return False


def actionable_proxy(answer: str, *, tool_completed: bool = False, grounded: bool = False) -> float:
    if tool_completed:
        return 5.0
    markers = sum(marker in answer for marker in ("今やること", "まず", "1.", "①", "件名", "入力", "確認", "計算結果", "順"))
    base = 2.5 if grounded else 1.0
    return min(4.5 if grounded else 3.5, base + markers * .5)


def evaluate(variant: str, blind: list[dict], checkpoint: str, tokenizer_path: str) -> dict:
    model, tokenizer, _, payload = load_model(checkpoint, tokenizer_path, "cpu")
    campus = UniPilotCampusV1(model, tokenizer) if variant == "d" else None
    legacy = V07Pipeline(model, tokenizer, top_k=1) if variant != "d" else None
    process = psutil.Process(); peak = process.memory_info().rss / 1024**2
    rows = []
    for index, item in enumerate(blind):
        started = time.perf_counter()
        if variant == "d":
            result = campus.answer(item["prompt"], 32, temperature=0.0, top_k=40, top_p=.9,
                                   repetition_penalty=1.1, session_id=f"blind-{index}")
            answer = result["text"]
            predicted = result["category"]
            category_correct = predicted == item["category"]
            matched_tool = result["tool"] == item["expected_tool"] if item["expected_tool"] else result["tool"] is None
            grounded = result["route"] in ("faq", "official")
            tool_completed = result["route"] == "tool" and not result["missing_fields"]
            validator_issues = result["validator"]["issues"]
            hall = any(issue in validator_issues for issue in
                       ("university_specific_assertion", "unsupported_money_or_date")) or unsafe_forbidden_claim(answer)
            completion = len(answer.strip()) >= 16
            actionable = result["validator"]["actionable_score"]
            route = result["route"]
            fallback = result["fallback_used"]
            source_grounded = grounded or (item["source_grounding_required"] and "公式" in answer)
            calculation_correct = (not item["calculation_check"] or result["calculation"] is not None or
                                   bool(result["missing_fields"]))
        else:
            use_grounding = variant in ("b", "c")
            result = legacy.answer(item["prompt"], 32, temperature=0.0, top_k=40, top_p=.9,
                                   repetition_penalty=1.1, use_retrieval=use_grounding,
                                   use_validator=use_grounding, force_model=not use_grounding)
            answer = result["text"]
            predicted = result["category"]
            category_correct = predicted == MINI_CATEGORY[item["category"]]
            matched_tool = item["expected_tool"] is None
            grounded = result["grounded_selected"]
            tool_completed = False
            validator_issues = result["validator"]["issues"]
            hall = any(issue.startswith(("invented_subject", "university_specific_hallucination")) for issue in validator_issues) or unsafe_forbidden_claim(answer)
            completion = len(answer.strip()) >= 16
            actionable = actionable_proxy(answer, grounded=grounded)
            route = "grounded" if grounded else "model"
            fallback = result["fallback_used"]
            source_grounded = grounded or not item["source_grounding_required"]
            calculation_correct = item["calculation_check"] is None
        elapsed = time.perf_counter() - started
        hits = keypoint_hits(answer, item["required_key_points"])
        category_terms = CAMPUS_KEYWORDS[item["category"]]
        semantic_hit = hits >= 1 or any(term in answer.lower() for term in category_terms)
        relevant = category_correct and (semantic_hit or matched_tool) and not hall
        correct = relevant and matched_tool and calculation_correct and (source_grounded or not item["source_grounding_required"])
        rows.append({
            "id": item["id"], "expected_category": item["category"], "predicted_category": predicted,
            "expected_tool": item["expected_tool"], "route": route, "answer": answer,
            "relevance": relevant, "correctness": correct, "category_correct": category_correct,
            "hallucination": hall, "completion": completion, "natural_japanese": natural(answer),
            "actionable_score": actionable, "matched_tool": matched_tool, "calculation_correct": calculation_correct,
            "source_grounded": source_grounded, "fallback": fallback, "response_seconds": elapsed,
            "validator_issues": validator_issues,
        })
        peak = max(peak, process.memory_info().rss / 1024**2)
        if (index + 1) % 50 == 0 or index + 1 == len(blind):
            print(f"{variant}: evaluated {index + 1}/{len(blind)}", flush=True)
    count = len(rows); rate = lambda key: sum(row[key] for row in rows) / count
    times = sorted(row["response_seconds"] for row in rows)
    metrics = {
        "questions": count, "correctness": rate("correctness"), "relevance": rate("relevance"),
        "category_accuracy": rate("category_correct"), "hallucination": rate("hallucination"),
        "completion": rate("completion"), "natural_japanese": rate("natural_japanese"),
        "actionable_score": statistics.fmean(row["actionable_score"] for row in rows),
        "human_score": None, "mean_response_seconds": statistics.fmean(times),
        "p95_response_seconds": times[int(count * .95) - 1], "peak_rss_mb": peak,
        "route_distribution": Counter(row["route"] for row in rows),
    }
    if variant == "d":
        for route in ("tool", "faq", "official", "safety", "model"):
            selected = [row["response_seconds"] for row in rows if row["route"] == route]
            metrics[f"{route}_mean_seconds"] = statistics.fmean(selected) if selected else None
            metrics[f"{route}_p95_seconds"] = sorted(selected)[int(len(selected) * .95) - 1] if selected else None
    return {"variant": variant, "checkpoint": checkpoint, "model": model.config.model_name,
            "parameters": model.parameter_count(), "vocab": tokenizer.vocab_size, "context": model.config.context_length,
            "step": payload.get("step"), "blind": "data/campus_v1/blind/evaluation.json", "metrics": metrics,
            "automatic_evaluation_limit": "Transparent lexical/tool/source proxy. Human score remains null until manual review.",
            "generations": rows, "external_ai_api": "OFF"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("a", "b", "c", "d"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = ("checkpoints/v07-grounded/unipilot-mini-v07-inference.pt" if args.variant == "c"
                  else "checkpoints/v04-eos15/unipilot-mini-v04-inference.pt")
    blind = json.loads(Path("data/campus_v1/blind/evaluation.json").read_text(encoding="utf-8"))
    report = evaluate(args.variant, blind, checkpoint, "tokenizer/vocab-v02-512.json")
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**report, "generations": f"{len(report['generations'])} rows"}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
