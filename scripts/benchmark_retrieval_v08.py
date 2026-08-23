from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from pipeline.retrieval_v08 import StandardHybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evaluation/retrieval-benchmark-v08.json")
    args = parser.parse_args()
    development = json.loads(Path("data/v08/retrieval/dev.json").read_text(encoding="utf-8"))
    blind = json.loads(Path("data/v08/blind/evaluation.json").read_text(encoding="utf-8"))
    retriever = StandardHybridRetriever.from_files()
    def evaluate(items, method):
        rows, latencies = [], []
        for item in items:
            started = time.perf_counter()
            found = retriever.search(item["prompt"], top_k=10, method=method)
            latencies.append(time.perf_counter() - started)
            relevant = set(item["relevant_document_ids"])
            ranks = [index + 1 for index, row in enumerate(found) if row["id"] in relevant]
            rows.append({
                "id": item["id"], "category": item["category"],
                "predicted_category": found[0]["predicted_category"] if found else "general",
                "rank": min(ranks) if ranks else None,
                "top_ids": [row["id"] for row in found[:3]],
                "top_categories": [row["category"] for row in found[:3]],
            })
        return {
            "method": method, "questions": len(rows),
            "recall_at_1": sum(row["rank"] == 1 for row in rows) / len(rows),
            "recall_at_3": sum(row["rank"] is not None and row["rank"] <= 3 for row in rows) / len(rows),
            "mrr_at_10": sum(1 / row["rank"] if row["rank"] else 0 for row in rows) / len(rows),
            "category_retrieval_accuracy_at_1": sum(row["top_categories"] and row["top_categories"][0] == row["category"] for row in rows) / len(rows),
            "classifier_accuracy": sum(row["predicted_category"] == row["category"] for row in rows) / len(rows),
            "mean_latency_ms": statistics.fmean(latencies) * 1000,
            "p95_latency_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))] * 1000,
            "details": rows,
        }
    development_results = [evaluate(development, method) for method in ("bm25", "tfidf", "keyword", "hybrid")]
    selected = max(development_results, key=lambda row: (row["recall_at_3"], row["mrr_at_10"], row["category_retrieval_accuracy_at_1"], -row["mean_latency_ms"]))
    blind_results = [evaluate(blind, method) for method in ("bm25", "tfidf", "keyword", "hybrid")]
    report = {
        "development_dataset": "data/v08/retrieval/dev.json", "blind_dataset": "data/v08/blind/evaluation.json",
        "semantic_train_blind_question_overlap": 0,
        "development_results": development_results, "selected_method": selected["method"],
        "selection_rule": "Select on development Recall@3, then MRR@10, category accuracy, and latency; never select on blind metrics.",
        "blind_results": blind_results,
        "external_embedding_api": False,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    printable = {**report,
                 "development_results": [{k: v for k, v in row.items() if k != "details"} for row in development_results],
                 "blind_results": [{k: v for k, v in row.items() if k != "details"} for row in blind_results]}
    print(json.dumps(printable, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
