from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from pipeline.classifier import (BM25CategoryClassifier, HybridCategoryClassifier, RuleCategoryClassifier,
                                 TfidfCategoryClassifier, benchmark_classifier)


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/v07/classifier/train.jsonl")
    parser.add_argument("--prompts", default="evaluation/fixed_prompts_v07.json")
    parser.add_argument("--output", default="evaluation/classifier-benchmark-v07.json")
    args = parser.parse_args()
    examples = load_jsonl(args.train)
    items = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    classifiers = (RuleCategoryClassifier(), BM25CategoryClassifier(examples), TfidfCategoryClassifier(examples),
                   HybridCategoryClassifier(examples))
    results = []
    for classifier in classifiers:
        result = benchmark_classifier(classifier, items, repeats=5)
        confusion = Counter()
        per_category = defaultdict(lambda: [0, 0])
        mistakes = []
        for item, prediction in zip(items, result.pop("predictions")):
            per_category[item["category"]][1] += 1
            per_category[item["category"]][0] += prediction == item["category"]
            if prediction != item["category"]:
                confusion[(item["category"], prediction)] += 1
                mistakes.append({"id": item["id"], "prompt": item["prompt"], "expected": item["category"],
                                 "predicted": prediction})
        result["per_category_accuracy"] = {category: correct / total for category, (correct, total) in sorted(per_category.items())}
        result["top_confusions"] = [{"expected": pair[0], "predicted": pair[1], "count": count}
                                     for pair, count in confusion.most_common(12)]
        result["mistakes"] = mistakes
        results.append(result)
    best = max(results, key=lambda item: (item["accuracy"], -item["mean_latency_ms"]))
    report = {"training_examples": len(examples), "evaluation_questions": len(items), "results": results,
              "selected_method": best["method"], "selected_accuracy": best["accuracy"],
              "selected_mean_latency_ms": best["mean_latency_ms"], "target_accuracy": 0.90,
              "target_passed": best["accuracy"] >= 0.90,
              "limitation": "Evaluation questions are held-out phrasings but share semantic seed families with part of the FAQ corpus."}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=True, indent=2))
    for result in results:
        print(json.dumps({key: value for key, value in result.items() if key not in ("mistakes", "per_category_accuracy")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
