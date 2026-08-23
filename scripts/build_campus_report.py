from __future__ import annotations

import json
from pathlib import Path
import statistics


FILES = {
    "a": "evaluation/results-campus-v1-a.json",
    "b": "evaluation/results-campus-v1-b.json",
    "c": "evaluation/results-campus-v1-c.json",
    "d": "evaluation/results-campus-v1-d.json",
}
LABELS = {"a": "v0.4 model only", "b": "v0.4 Grounded", "c": "v0.7 Grounded", "d": "Campus v1"}
THRESHOLDS = {"relevance": .90, "correctness": .85, "category_accuracy": .95, "hallucination_max": .01,
              "completion": .98, "natural_japanese": .98, "actionable_score": 4.0, "human_score": 4.0,
              "faq_tool_p95_seconds": 1.0}


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def route_latency(report: dict, route: str) -> dict:
    values = sorted(row["response_seconds"] for row in report["generations"] if row["route"] == route)
    return {"questions": len(values), "mean_seconds": statistics.fmean(values) if values else None,
            "p95_seconds": values[int(len(values) * .95) - 1] if values else None}


def main() -> None:
    reports = {key: load(path) for key, path in FILES.items()}
    summaries = {}
    for key, report in reports.items():
        summaries[key] = {"label": LABELS[key], **report["metrics"]}
        summaries[key]["model_route_latency"] = route_latency(report, "model")
        summaries[key]["grounded_route_latency"] = route_latency(report, "grounded")

    campus = reports["d"]
    metrics = campus["metrics"]
    gate_checks = {
        "relevance": metrics["relevance"] >= THRESHOLDS["relevance"],
        "correctness": metrics["correctness"] >= THRESHOLDS["correctness"],
        "category_accuracy": metrics["category_accuracy"] >= THRESHOLDS["category_accuracy"],
        "hallucination": metrics["hallucination"] <= THRESHOLDS["hallucination_max"],
        "completion": metrics["completion"] >= THRESHOLDS["completion"],
        "natural_japanese": metrics["natural_japanese"] >= THRESHOLDS["natural_japanese"],
        "actionable_score": metrics["actionable_score"] >= THRESHOLDS["actionable_score"],
        "human_score": False,
        "faq_latency": metrics["faq_p95_seconds"] <= THRESHOLDS["faq_tool_p95_seconds"],
        "tool_latency": metrics["tool_p95_seconds"] <= THRESHOLDS["faq_tool_p95_seconds"],
    }
    gate = {"candidate": "UniPilot Campus v1", "production_before": "UniPilot Mini v0.4 step 2000",
            "production_after": "UniPilot Mini v0.4 step 2000", "thresholds": THRESHOLDS,
            "measured": metrics, "checks": gate_checks, "decision": "REJECT_PRODUCTION_PROMOTION",
            "production_promoted": False, "push_performed": False, "deployment_performed": False,
            "reason": "Category, relevance, correctness, actionable score, and human score gates are not met."}
    Path("evaluation/campus-v1-promotion-gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")

    benchmark = {"blind": "data/campus_v1/blind/evaluation.json", "questions": 1000,
                 "automatic_scoring_limit": "Lexical, deterministic-tool, source and safety proxy; no human score is fabricated.",
                 "variants": summaries, "promotion_gate": "evaluation/campus-v1-promotion-gate.json",
                 "production_changed": False, "external_ai_api": "OFF"}
    Path("evaluation/campus-v1-benchmark.json").write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")

    manual = load("data/campus_v1/benchmark/chatgpt-gemini-manual-100.json")
    campus_by_prompt = {row["id"]: row for row in campus["generations"]}
    blind = load("data/campus_v1/blind/evaluation.json")
    blind_id_by_prompt = {row["prompt"]: row["id"] for row in blind}
    for item in manual:
        result = campus_by_prompt[blind_id_by_prompt[item["question"]]]
        item["campus_answer"] = result["answer"]
        item["campus_latency_seconds"] = result["response_seconds"]
        item["campus_route"] = result["route"]
    Path("evaluation/human-comparison-campus-v1.json").write_text(json.dumps(manual, ensure_ascii=False, indent=2), encoding="utf-8")

    md = f"""# UniPilot Campus v1 evaluation

## Outcome

Campus v1 is implemented as an opt-in orchestration mode over the unchanged v0.4 model. Production v0.4, Render, Vercel, GitHub Release, and Standard v0.8 are unchanged. No external LLM or teacher model was used.

| Metric | v0.4 model only | v0.4 Grounded | v0.7 Grounded | Campus v1 |
|---|---:|---:|---:|---:|
| Correctness | {summaries['a']['correctness']:.1%} | {summaries['b']['correctness']:.1%} | {summaries['c']['correctness']:.1%} | {summaries['d']['correctness']:.1%} |
| Relevance | {summaries['a']['relevance']:.1%} | {summaries['b']['relevance']:.1%} | {summaries['c']['relevance']:.1%} | {summaries['d']['relevance']:.1%} |
| Category | {summaries['a']['category_accuracy']:.1%} mapped | {summaries['b']['category_accuracy']:.1%} mapped | {summaries['c']['category_accuracy']:.1%} mapped | {summaries['d']['category_accuracy']:.1%} |
| Hallucination proxy | {summaries['a']['hallucination']:.1%} | {summaries['b']['hallucination']:.1%} | {summaries['c']['hallucination']:.1%} | {summaries['d']['hallucination']:.1%} |
| Completion | {summaries['a']['completion']:.1%} | {summaries['b']['completion']:.1%} | {summaries['c']['completion']:.1%} | {summaries['d']['completion']:.1%} |
| Natural Japanese | {summaries['a']['natural_japanese']:.1%} | {summaries['b']['natural_japanese']:.1%} | {summaries['c']['natural_japanese']:.1%} | {summaries['d']['natural_japanese']:.1%} |
| Actionable Score | {summaries['a']['actionable_score']:.3f} | {summaries['b']['actionable_score']:.3f} | {summaries['c']['actionable_score']:.3f} | {summaries['d']['actionable_score']:.3f} |
| Human Score | unscored | unscored | unscored | unscored 100-item UI/JSON |
| Mean response | {summaries['a']['mean_response_seconds']*1000:.1f} ms | {summaries['b']['mean_response_seconds']*1000:.1f} ms | {summaries['c']['mean_response_seconds']*1000:.1f} ms | {summaries['d']['mean_response_seconds']*1000:.1f} ms |
| P95 response | {summaries['a']['p95_response_seconds']*1000:.1f} ms | {summaries['b']['p95_response_seconds']*1000:.1f} ms | {summaries['c']['p95_response_seconds']*1000:.1f} ms | {summaries['d']['p95_response_seconds']*1000:.1f} ms |
| Peak RSS | {summaries['a']['peak_rss_mb']:.2f} MiB | {summaries['b']['peak_rss_mb']:.2f} MiB | {summaries['c']['peak_rss_mb']:.2f} MiB | {summaries['d']['peak_rss_mb']:.2f} MiB |

## Campus routing and latency

- Routes: 465 tool, 410 FAQ, 104 model, and 21 safe university-policy answers.
- Tool latency: mean {metrics['tool_mean_seconds']*1000:.2f} ms, P95 {metrics['tool_p95_seconds']*1000:.2f} ms.
- FAQ latency: mean {metrics['faq_mean_seconds']*1000:.2f} ms, P95 {metrics['faq_p95_seconds']*1000:.2f} ms.
- Model-route total latency: mean {metrics['model_mean_seconds']*1000:.2f} ms, P95 {metrics['model_p95_seconds']*1000:.2f} ms for the 32-token bounded benchmark.
- Router comparison: the separate development set selects BM25 at 99.43%, but its untouched blind accuracy falls to 59.1%. Rules reach 77.9%, TF-IDF 54.9%, and hybrid 68.3% on blind. Campus uses hybrid for high-precision deterministic-tool overrides, and records 68.8% end-to-end category accuracy.

## Decision

Campus is materially better than model-only and older Grounded variants, especially for instant calculations, complete email drafts, plans, cards, and university-policy refusal. It still does not satisfy the stated promotion thresholds. The gap is primarily routing/coverage: when Campus selects the correct category, 686 of 688 answers pass the automatic correctness proxy. Long-form open-ended reasoning and general knowledge remain weaker than large external models.

Do not restart Standard 50M training yet. First improve the router on a newly collected, human-labeled colloquial dataset without reusing this consumed blind set; review the 1,000 programmatically composed FAQ rows; complete the 100-question human comparison; and raise Actionable Score above 4.0. Resume 50M only if generation quality remains the largest error after routing and tool coverage pass.

**Promotion decision: REJECT.** No push or deployment is performed.
"""
    Path("evaluation/comparison-campus-v1.md").write_text(md, encoding="utf-8")
    print(json.dumps({"variants": summaries, "gate": gate["decision"], "human_items": len(manual)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
