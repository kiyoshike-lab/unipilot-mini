from __future__ import annotations

import json
from pathlib import Path
import statistics
import time

from pipeline.campus_retrieval import load_jsonl
from pipeline.campus_router import CampusBM25Router, CampusHybridRouter, CampusRuleRouter, CampusTfidfRouter


def evaluate(router, blind: list[dict]) -> dict:
    rows = []
    for item in blind:
        started = time.perf_counter()
        category, confidence, _ = router.predict(item["prompt"])
        rows.append({"id": item["id"], "expected": item["category"], "predicted": category,
                     "correct": category == item["category"], "confidence": confidence,
                     "latency_ms": (time.perf_counter() - started) * 1000})
    return {"method": router.name, "questions": len(rows),
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "mean_latency_ms": statistics.fmean(row["latency_ms"] for row in rows),
            "p95_latency_ms": sorted(row["latency_ms"] for row in rows)[int(len(rows) * .95) - 1],
            "predictions": rows}


def main() -> None:
    examples = load_jsonl("data/campus_v1/router/train.jsonl")
    development = json.loads(Path("data/campus_v1/router/dev.json").read_text(encoding="utf-8"))
    blind = json.loads(Path("data/campus_v1/blind/evaluation.json").read_text(encoding="utf-8"))
    routers = (CampusRuleRouter(), CampusBM25Router(examples), CampusTfidfRouter(examples), CampusHybridRouter(examples))
    development_results = [evaluate(router, development) for router in routers]
    selected = max(development_results, key=lambda row: (row["accuracy"], -row["mean_latency_ms"]))
    blind_results = [evaluate(router, blind) for router in routers]
    selected_blind = next(row for row in blind_results if row["method"] == selected["method"])
    report = {"development_dataset": "data/campus_v1/router/dev.json", "blind_dataset": "data/campus_v1/blind/evaluation.json",
              "selection_policy": "method is frozen on the separate development set before final blind scores are inspected",
              "development_results": development_results, "blind_results": blind_results,
              "selected": selected["method"], "target_accuracy": .95,
              "target_passed": selected_blind["accuracy"] >= .95, "external_ai_api": "OFF"}
    Path("evaluation/router-benchmark-campus-v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    printable = {**report,
                 "development_results": [{key: value for key, value in row.items() if key != "predictions"} for row in development_results],
                 "blind_results": [{key: value for key, value in row.items() if key != "predictions"} for row in blind_results]}
    print(json.dumps(printable, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
