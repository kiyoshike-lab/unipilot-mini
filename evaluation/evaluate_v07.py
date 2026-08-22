from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import psutil
import torch

from inference.generate import load_model
from pipeline.categories import CATEGORY_KEYWORDS
from pipeline.v07 import V07Pipeline


def japanese_ratio(text: str) -> float:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 0.0
    return sum(bool(re.match(r"[ぁ-んァ-ヶー一-龥々、。！？「」『』]", char)) for char in visible) / len(visible)


def complete(text: str) -> bool:
    return bool(text.strip()) and text.rstrip().endswith(("。", "！", "？", "ます", "です"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/vocab-v02-512.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--prompts", default="evaluation/fixed_prompts_v07.json")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--candidates", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--no-validator", action="store_true")
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument("--force-model", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(7072026)
    model, tokenizer, device, payload = load_model(args.checkpoint, args.tokenizer, "cpu")
    pipeline = V07Pipeline(model, tokenizer)
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    if args.limit:
        prompts = prompts[:args.limit]
    process = psutil.Process()
    peak = process.memory_info().rss / 1024**2
    rows = []
    for item in prompts:
        result = pipeline.answer(item["prompt"], args.max_new_tokens, candidates=args.candidates,
                                 use_validator=not args.no_validator, use_retrieval=not args.no_retrieval,
                                 force_model=args.force_model)
        answer = result["text"]
        expected = item.get("expected_keywords", [])
        expected_hit = not expected or any(word in answer for word in expected)
        category_hit = any(word in answer.lower() for word in CATEGORY_KEYWORDS.get(item["category"], ()))
        category_correct = result["category"] == item["category"]
        hallucination = any(issue.startswith(("invented_subject", "university_specific_hallucination"))
                            for issue in result["validator"]["issues"])
        relevance = category_correct and (expected_hit or category_hit) and not hallucination
        is_complete = complete(answer)
        natural = japanese_ratio(answer) >= 0.75 and "�" not in answer
        peak = max(peak, process.memory_info().rss / 1024**2)
        rows.append({**item, "answer": answer, "raw_answer": result["raw_text"], "predicted_category": result["category"],
                     "category_correct": category_correct, "expected_keyword_hit": expected_hit,
                     "category_keyword_hit": category_hit, "relevance_pass": relevance,
                     "correctness_pass": relevance and result["validator"]["valid"], "hallucination": hallucination,
                     "complete": is_complete, "natural_japanese": natural, "effective_eos": is_complete,
                     "model_eos": bool(result["generation_metrics"].get("eos_reached")),
                     "fallback_used": result["fallback_used"], "grounded_selected": result["grounded_selected"],
                     "selected_source": result["selected_source"], "validator": result["validator"],
                     "raw_validator": result["raw_validator"], "retrieval": result["retrieval"],
                     "context_tokens": result["context_tokens"], "length_policy": result["length_policy"],
                     "model_generation_skipped": result["model_generation_skipped"],
                     "timing": result["timing"], "generation_metrics": result["generation_metrics"]})
    count = len(rows)
    metrics = {
        "questions": count, "category_accuracy": sum(row["category_correct"] for row in rows) / count,
        "relevance_rate": sum(row["relevance_pass"] for row in rows) / count,
        "correctness_rate": sum(row["correctness_pass"] for row in rows) / count,
        "hallucination_rate": sum(row["hallucination"] for row in rows) / count,
        "completion_rate": sum(row["complete"] for row in rows) / count,
        "eos_rate": sum(row["effective_eos"] for row in rows) / count,
        "model_eos_rate": sum(row["model_eos"] for row in rows) / count,
        "natural_japanese_rate": sum(row["natural_japanese"] for row in rows) / count,
        "fallback_rate": sum(row["fallback_used"] for row in rows) / count,
        "grounded_selection_rate": sum(row["grounded_selected"] for row in rows) / count,
        "model_generation_skip_rate": sum(row["model_generation_skipped"] for row in rows) / count,
        "mean_total_seconds": sum(row["timing"]["total_seconds"] for row in rows) / count,
        "mean_generation_tokens_per_second": sum(row["generation_metrics"].get("tokens_per_sec", 0) for row in rows) / count,
        "peak_rss_mb": peak,
    }
    categories = defaultdict(list)
    for row in rows:
        categories[row["category"]].append(row)
    result = {"label": args.label, "checkpoint": args.checkpoint, "model": model.config.model_name,
              "parameters": model.parameter_count(), "vocab": model.config.vocab_size, "context": model.config.context_length,
              "step": payload.get("step"), "retrieval_enabled": not args.no_retrieval,
              "validator_enabled": not args.no_validator, "candidate_count": args.candidates, "metrics": metrics,
              "force_model": args.force_model,
              "per_category": {category: {"questions": len(items),
                               "relevance_rate": sum(row["relevance_pass"] for row in items) / len(items)}
                               for category, items in sorted(categories.items())},
              "automated_evaluation_limit": "FAQ-grounded proxy; final human review is still required.", "generations": rows}
    output = Path(args.output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    human = {"label": args.label, "completed": False, "target": "mean >= 3.5/5",
             "instructions": "Score directness, correctness, category fit, naturalness, and usefulness from 0 to 5.",
             "items": [{"id": row["id"], "prompt": row["prompt"], "answer": row["answer"], "score": None, "notes": ""}
                       for row in rows]}
    output.with_name(output.stem + "-human.json").write_text(json.dumps(human, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("generations", "per_category")}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
