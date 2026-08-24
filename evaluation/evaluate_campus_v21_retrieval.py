from __future__ import annotations

import json
from pathlib import Path
import statistics
from time import perf_counter

from pipeline.campus_retrieval_v21 import CampusFAQRetrieverV21


ROOT = Path(__file__).resolve().parents[1]


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * fraction))]


def evaluate(retriever: CampusFAQRetrieverV21, rows: list[dict], method: str, threshold: float) -> dict:
    ranks, latencies = [], []
    false_matches = predicted_no_match = correct_no_match = 0
    match_rows = no_match_rows = 0
    failures = []
    for item in rows:
        started = perf_counter()
        found = retriever.search_method(item["query"], item.get("category"), method, 5, "high", threshold)
        latencies.append((perf_counter() - started) * 1000)
        identifiers = [row["id"] for row in found]
        if not item["has_match"]:
            no_match_rows += 1
            if found:
                false_matches += 1
                if len(failures) < 50:
                    failures.append({"id": item["id"], "query": item["query"], "reason": "FALSE_FAQ_MATCH",
                                     "returned": identifiers, "score": found[0]["retrieval_score"]})
            else:
                predicted_no_match += 1; correct_no_match += 1
            continue
        match_rows += 1
        if not found:
            predicted_no_match += 1
            ranks.append(0)
            if len(failures) < 50:
                failures.append({"id": item["id"], "query": item["query"], "reason": "FALSE_NO_MATCH",
                                 "relevant": item["relevant_ids"]})
            continue
        relevant = set(item["relevant_ids"])
        rank = next((index + 1 for index, identifier in enumerate(identifiers) if identifier in relevant), 0)
        ranks.append(rank)
        if rank == 0 and len(failures) < 50:
            failures.append({"id": item["id"], "query": item["query"], "reason": "WRONG_FAQ",
                             "returned": identifiers, "relevant": item["relevant_ids"],
                             "score": found[0]["retrieval_score"]})
    no_match_precision = correct_no_match / predicted_no_match if predicted_no_match else 0.0
    return {
        "method": method, "threshold": threshold, "questions": len(rows), "match_questions": match_rows,
        "no_match_questions": no_match_rows, "recall_at_1": sum(rank == 1 for rank in ranks) / match_rows,
        "recall_at_3": sum(0 < rank <= 3 for rank in ranks) / match_rows,
        "recall_at_5": sum(0 < rank <= 5 for rank in ranks) / match_rows,
        "mrr": statistics.fmean(1 / rank if rank else 0 for rank in ranks),
        "no_match_precision": no_match_precision, "false_faq_match": false_matches / no_match_rows,
        "mean_latency_ms": statistics.fmean(latencies), "p95_latency_ms": percentile(latencies, .95),
        "failures": failures,
    }


def main() -> None:
    validation = json.loads((ROOT / "data" / "campus_v21" / "retrieval" / "validation.json").read_text(encoding="utf-8"))
    test = json.loads((ROOT / "data" / "campus_v21" / "retrieval" / "test.json").read_text(encoding="utf-8"))
    retriever = CampusFAQRetrieverV21.from_jsonl(config_path=ROOT / "data" / "campus_v21" / "retrieval" / "missing.json")
    comparisons = []
    for method in retriever.methods:
        best = None
        for threshold in (0.0, .12, .14, .145, .15, .16, .17, .18, .22, .26, .30, .38, .48, .55):
            result = evaluate(retriever, validation, method, threshold)
            eligible = result["false_faq_match"] <= .02 and result["no_match_precision"] >= .90
            key = (eligible, result["recall_at_1"], result["recall_at_3"], result["mrr"],
                   -result["false_faq_match"])
            if best is None or key > best[0]:
                best = (key, result)
        comparisons.append(best[1])
        print(method, round(best[1]["recall_at_1"], 4), round(best[1]["false_faq_match"], 4), best[1]["threshold"])
    eligible = [row for row in comparisons if row["false_faq_match"] <= .02 and row["no_match_precision"] >= .90]
    selected = max(eligible or comparisons, key=lambda row: (row["recall_at_1"], row["recall_at_3"], row["mrr"],
                                                               row["method"] == "router_aware"))
    config = {"selected_method": selected["method"], "selected_threshold": selected["threshold"],
              "selection_split": "data/campus_v21/retrieval/validation.json",
              "selection_policy": "false FAQ match <=2%, then maximize Recall@1/Recall@3/MRR",
              "validation_metrics": {key: selected[key] for key in ("recall_at_1", "recall_at_3", "recall_at_5",
                                                                       "mrr", "no_match_precision", "false_faq_match")},
              "external_ai_api": "OFF"}
    config_path = ROOT / "data" / "campus_v21" / "retrieval" / "retrieval-config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    final_retriever = CampusFAQRetrieverV21.from_jsonl(config_path=config_path)
    test_result = evaluate(final_retriever, test, config["selected_method"], config["selected_threshold"])
    output = {"validation_questions": len(validation), "test_questions": len(test),
              "validation_comparison": comparisons, "selected": config, "test": test_result,
              "test_is_independent_of_threshold_selection": True, "external_ai_api": "OFF"}
    (ROOT / "evaluation" / "campus-v21-retrieval.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("selected", config["selected_method"], config["selected_threshold"],
          json.dumps({key: test_result[key] for key in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr",
                                                        "no_match_precision", "false_faq_match")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
