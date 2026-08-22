from __future__ import annotations

import argparse
import json
from pathlib import Path


KPI = {
    "natural_rate": 0.95, "completion_rate": 0.90, "relevance_rate": 0.85,
    "keyword_rate": 0.80, "category_accuracy": 0.80, "unnecessary_information_rate_max": 0.05,
    "runaway_rate_max": 0.01, "repetition_rate_max": 0.02, "eos_rate": 0.90,
}


def quality_score(metrics: dict) -> float:
    return (
        4 * metrics["relevance_rate"] + 3 * metrics["accuracy_rate"] + 2 * metrics["category_accuracy"]
        + metrics["keyword_rate"] + metrics["completion_rate"] + metrics["eos_rate"]
        + metrics["natural_rate"] - 3 * metrics["hallucination_rate"]
        - 2 * metrics["unnecessary_information_rate"] - metrics["runaway_rate"]
        - metrics["repetition_rate"]
    )


def kpi_pass(metrics: dict) -> dict:
    return {
        "natural": metrics["natural_rate"] >= KPI["natural_rate"],
        "completion": metrics["completion_rate"] >= KPI["completion_rate"],
        "relevance": metrics["relevance_rate"] >= KPI["relevance_rate"],
        "keyword": metrics["keyword_rate"] >= KPI["keyword_rate"],
        "category": metrics["category_accuracy"] >= KPI["category_accuracy"],
        "unnecessary": metrics["unnecessary_information_rate"] <= KPI["unnecessary_information_rate_max"],
        "runaway": metrics["runaway_rate"] <= KPI["runaway_rate_max"],
        "repetition": metrics["repetition_rate"] <= KPI["repetition_rate_max"],
        "eos": metrics["eos_rate"] >= KPI["eos_rate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="evaluation/results-v04-v06suite.json")
    parser.add_argument("--candidates", nargs="+", default=[
        "evaluation/results-v06-generic-500.json", "evaluation/results-v06-generic-1000.json",
        "evaluation/results-v06-generic-2000.json"])
    parser.add_argument("--output", default="evaluation/v06-selection.json")
    args = parser.parse_args()
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidates = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.candidates]
    assessed = []
    previous = baseline
    for path, candidate in zip(args.candidates, candidates):
        metrics = candidate["metrics"]
        previous_metrics = previous["metrics"]
        regressions = [name for name in ("natural_rate", "completion_rate", "relevance_rate", "category_accuracy", "eos_rate")
                       if metrics[name] + 0.02 < previous_metrics[name]]
        assessed.append({
            "result": path, "checkpoint": candidate["checkpoint"], "step": candidate["step"],
            "quality_score": quality_score(metrics), "kpi_pass": kpi_pass(metrics),
            "regressions_over_2pp_vs_previous": regressions,
        })
        previous = candidate
    best_index = max(range(len(candidates)), key=lambda index: quality_score(candidates[index]["metrics"]))
    best = candidates[best_index]
    best_kpi = kpi_pass(best["metrics"])
    promotion = all(best_kpi.values()) and best["metrics"]["rubric_mean_0_to_5"]["overall_quality"] >= 4.0
    report = {
        "baseline": args.baseline, "kpi": KPI, "stages": assessed,
        "automatic_best_result": args.candidates[best_index], "automatic_best_checkpoint": best["checkpoint"],
        "automatic_best_step": best["step"], "production_promotion": promotion,
        "production_model": "v0.6" if promotion else "v0.4",
        "stop_reason": "KPI gate failed; 5000/10000 would overfit the 798-row core without evidence of closing the relevance gap." if not promotion else None,
        "human_gate": "Not passed: representative manual audit is below 4/5 and full blinded human scoring is incomplete.",
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
