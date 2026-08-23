from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics

import psutil

from inference.generate import load_model
from pipeline.v08 import V08Pipeline
from training.train_v08 import PROBE_STOP_BIGRAMS


TARGET_DOMAINS = {
    "exam", "credit", "gpa", "professor_email", "report", "study", "career", "campus_life",
}


def expected_hits(text: str, points: list[str]) -> int:
    fragments = {fragment for point in points for fragment in
                 (point[index:index + 2] for index in range(max(0, len(point) - 1)))
                 if fragment not in PROBE_STOP_BIGRAMS and re.search(r"[ぁ-んァ-ヶー一-龥々A-Za-z0-9]", fragment)}
    return sum(fragment in text for fragment in fragments)


def repetition_rate(text: str) -> float:
    grams = [text[index:index + 3] for index in range(max(0, len(text) - 2))]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/standard-v08-scratch/unipilot-standard-v08-a100-inference.pt")
    parser.add_argument("--tokenizer", default="tokenizer/vocab-standard-v08-1024.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", default="evaluation/results-standard-v08-a100-blind.json")
    args = parser.parse_args()
    blind = json.loads(Path("data/v08/blind/evaluation.json").read_text(encoding="utf-8"))
    if args.limit:
        blind = [blind[index * len(blind) // args.limit] for index in range(args.limit)]
    model, tokenizer, device, payload = load_model(args.checkpoint, args.tokenizer, "cpu")
    pipeline = V08Pipeline(model, tokenizer, retrieval_method="tfidf", top_k=3)
    rows = []
    process = psutil.Process()
    peak = process.memory_info().rss / 1024**2
    for item in blind:
        result = pipeline.answer(item["prompt"], args.max_new_tokens, temperature=0.0,
                                 top_k=40, top_p=0.9, repetition_penalty=1.1, response_mode="auto")
        raw = result["raw_text"]
        hits = expected_hits(raw, item["expected_key_points"])
        issues = result["validator"]["issues"]
        complete = len(raw.strip()) >= 12 and bool(result["generation_metrics"]["eos_reached"])
        relevant = hits >= 4
        hallucination = any(issue in issues for issue in ("university_specific_hallucination", "unsupported_date_or_fee"))
        category_correct = result["category"] == item["category"]
        correctness = relevant and not hallucination and category_correct
        natural = bool(raw.strip()) and "�" not in raw and not re.search(r"[\x00-\x1f]", raw)
        rows.append({
            "id": item["id"], "prompt": item["prompt"], "expected_category": item["category"],
            "predicted_category": result["category"], "difficulty": item["difficulty"],
            "expected_key_points": item["expected_key_points"], "model_answer": raw,
            "final_answer": result["text"], "response_mode": result["response_mode"],
            "relevant": relevant, "correct": correctness, "category_correct": category_correct,
            "natural": natural, "complete": complete, "eos": result["generation_metrics"]["eos_reached"],
            "hallucination": hallucination, "unsupported_claim": "unsupported_date_or_fee" in issues,
            "repetition_rate": repetition_rate(raw), "useful": correctness and complete,
            "fallback_used": result["fallback_used"], "retrieved_ids": [row["id"] for row in result["retrieval"]],
            "tokens_per_second": result["generation_metrics"]["tokens_per_sec"],
            "first_token_not_measured_in_nonstreaming_run": True,
        })
        peak = max(peak, process.memory_info().rss / 1024**2)
    count = len(rows)
    metric = lambda key: sum(bool(row[key]) for row in rows) / count
    per_domain = {}
    for category in TARGET_DOMAINS:
        selected = [row for row in rows if row["expected_category"] == category]
        per_domain[category] = {
            "questions": len(selected),
            "relevance": sum(row["relevant"] for row in selected) / max(1, len(selected)),
            "correctness": sum(row["correct"] for row in selected) / max(1, len(selected)),
            "usefulness": sum(row["useful"] for row in selected) / max(1, len(selected)),
            "human_mean": None,
        }
    metrics = {
        "questions": count, "natural_japanese_rate": metric("natural"), "relevance_rate": metric("relevant"),
        "correctness_rate": metric("correct"), "category_accuracy": metric("category_correct"),
        "completion_rate": metric("complete"), "eos_rate": metric("eos"),
        "hallucination_rate": metric("hallucination"), "unsupported_claim_rate": metric("unsupported_claim"),
        "mean_repetition_rate": statistics.fmean(row["repetition_rate"] for row in rows),
        "answer_usefulness_rate": metric("useful"), "fallback_rate": metric("fallback_used"),
        "mean_tokens_per_second": statistics.fmean(row["tokens_per_second"] for row in rows),
        "peak_rss_mb": peak,
    }
    report = {
        "model": model.config.model_name, "checkpoint": args.checkpoint, "parameters": model.parameter_count(),
        "vocab": tokenizer.vocab_size, "context": model.config.context_length, "step": payload.get("step"),
        "blind_dataset": "data/v08/blind/evaluation.json", "metrics": metrics,
        "difficulty_distribution": Counter(row["difficulty"] for row in rows),
        "target_domain_metrics": per_domain, "generations": rows,
        "automatic_evaluation_limit": "Lexical transparent proxy. Human fields remain unscored; no human mean is claimed.",
        "external_ai_api": "OFF",
    }
    output = Path(args.output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    human = [{
        "id": row["id"], "question": row["prompt"], "category": row["expected_category"],
        "expected_key_points": row["expected_key_points"], "model_answer": row["model_answer"],
        "score_0_to_5": None, "notes": "", "blind": True,
    } for row in rows[:100]]
    output.with_name(output.stem + "-human-100.json").write_text(json.dumps(human, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**report, "generations": f"{len(rows)} rows written"}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
