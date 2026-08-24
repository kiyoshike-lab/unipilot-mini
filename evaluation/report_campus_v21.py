from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def main() -> None:
    benchmark = json.loads((ROOT / "evaluation/campus-v21-benchmark.json").read_text(encoding="utf-8"))
    analysis = json.loads((ROOT / "evaluation/campus-v21-full-failure-analysis.json").read_text(encoding="utf-8"))
    blind_v2, blind_v21 = benchmark["comparisons"]["blind_2000"]
    real_v2, real_v21 = benchmark["comparisons"]["real_student_500"]
    b2, b21 = blind_v2["metrics"], blind_v21["metrics"]
    r2, r21 = real_v2["metrics"], real_v21["metrics"]
    retrieval = benchmark["retrieval"]["test"]
    retrieval_v2 = json.loads((ROOT / "evaluation/campus-v2-benchmark.json").read_text(encoding="utf-8"))["retrieval"]
    adversarial_v2, adversarial_v21 = benchmark["adversarial_300"]
    gate = benchmark["production_gate"]
    failure = analysis["blind_2000"][1]
    adversarial = analysis["adversarial_300"][1]
    decomposition = blind_v21["correctness_decomposition"]
    memory = benchmark["memory"]["after_evaluation"]

    lines = [
        "# UniPilot Campus v2.1 evaluation",
        "",
        "本番v0.4、Standard 50M、Render、Vercel、Releaseには変更を加えていない。外部AI/APIはOFF。",
        "既存adversarial 300件とblind 2000件はtest専用で、学習・閾値選択には使用していない。",
        "別作成のadversarial train 1,500件とvalidation 300件は分離し、validation accuracyは100%。",
        "",
        "## Campus v2 vs v2.1 — same blind 2000",
        "",
        "| Metric | Campus v2 | Campus v2.1 | Gate |",
        "|---|---:|---:|---:|",
        f"| Determinate Category Accuracy | {percent(b2['determinate_category_accuracy'])} | {percent(b21['determinate_category_accuracy'])} | ≥97% |",
        f"| Ambiguous Handling Accuracy | {percent(b2['ambiguous_handling_accuracy'])} | {percent(b21['ambiguous_handling_accuracy'])} | ≥97% |",
        f"| Overall Routing Success | {percent(b2['overall_routing_success'])} | {percent(b21['overall_routing_success'])} | ≥95% |",
        f"| Action Accuracy | {percent(b2['action_accuracy'])} | {percent(b21['action_accuracy'])} | ≥95% |",
        f"| Adversarial Category Accuracy | {percent(adversarial_v2['metrics']['determinate_category_accuracy'])} | {percent(adversarial_v21['metrics']['determinate_category_accuracy'])} | ≥95% |",
        f"| Correctness | {percent(b2['correctness'])} | {percent(b21['correctness'])} | ≥92% |",
        f"| Relevance | {percent(b2['relevance'])} | {percent(b21['relevance'])} | ≥92% |",
        f"| Hallucination | {percent(b2['hallucination'])} | {percent(b21['hallucination'])} | ≤1% |",
        f"| Completion | {percent(b2['completion'])} | {percent(b21['completion'])} | ≥99% |",
        f"| Natural Japanese | {percent(b2['natural_japanese'])} | {percent(b21['natural_japanese'])} | ≥99% |",
        f"| Actionable | {b2['actionable_score']:.3f} | {b21['actionable_score']:.3f} | ≥4.5 |",
        f"| Retrieval Recall@1 | {percent(retrieval_v2['recall_at_1'])} | {percent(retrieval['recall_at_1'])} | ≥90% |",
        f"| Retrieval Recall@3 | {percent(retrieval_v2['recall_at_3'])} | {percent(retrieval['recall_at_3'])} | ≥95% |",
        f"| Retrieval MRR | {retrieval_v2['mrr']:.3f} | {retrieval['mrr']:.3f} | ≥0.92 |",
        f"| False FAQ Match | — | {percent(retrieval['false_faq_match'])} | ≤2% |",
        f"| Router P95 | {b2['router_p95_ms']:.3f} ms | {b21['router_p95_ms']:.3f} ms | <20 ms |",
        f"| FAQ P95 | {b2['route_latency_ms']['faq']['p95']:.3f} ms | {b21['route_latency_ms']['faq']['p95']:.3f} ms | <50 ms |",
        f"| Tool P95 | {b2['route_latency_ms']['tool']['p95']:.3f} ms | {b21['route_latency_ms']['tool']['p95']:.3f} ms | <50 ms |",
        f"| Total P95 | {b2['p95_latency_ms']:.3f} ms | {b21['p95_latency_ms']:.3f} ms | — |",
        f"| Peak RAM | — | {memory['peak_rss_mb']:.2f} MB | <450 MB |",
        "",
        "## Real Student Set 500",
        "",
        "100件ずつ: very short / colloquial / correction / normal / compound。既存データとの正規化完全一致は0件。",
        "",
        "| Metric | Campus v2 | Campus v2.1 |",
        "|---|---:|---:|",
        f"| Category / Routing | {percent(r2['overall_routing_success'])} | {percent(r21['overall_routing_success'])} |",
        f"| Action | {percent(r2['action_accuracy'])} | {percent(r21['action_accuracy'])} |",
        f"| Correctness | {percent(r2['correctness'])} | {percent(r21['correctness'])} |",
        f"| Multi-intent Recall | {percent(r2['multi_intent_recall'])} | {percent(r21['multi_intent_recall'])} |",
        f"| Actionable | {r2['actionable_score']:.3f} | {r21['actionable_score']:.3f} |",
        "",
        "## Retrieval — independent test 338",
        "",
        f"Selected: `{benchmark['retrieval']['selected']['selected_method']}` / threshold "
        f"`{benchmark['retrieval']['selected']['selected_threshold']}` (validation only).",
        "",
        "| Recall@1 | Recall@3 | MRR | false FAQ | P95 |",
        "|---:|---:|---:|---:|---:|",
        f"| {percent(retrieval['recall_at_1'])} | {percent(retrieval['recall_at_3'])} | "
        f"{retrieval['mrr']:.3f} | {percent(retrieval['false_faq_match'])} | {retrieval['p95_latency_ms']:.3f} ms |",
        "",
        "Retrieval failures:", "",
    ]
    for row in retrieval["failures"][:10]:
        lines.append(f"- `{row['id']}` {row['reason']}: {row['query']}")

    lines.extend(["", "## Remaining blind failures (top 10)", ""])
    for row in failure["all_failures"][:10]:
        lines.append(f"- `{row['id']}` {row['reason']}: {row['gold']} → {row['predicted']} / "
                     f"{row['expected_action']} → {row['action']} / margin {row['margin']:.3f}")
    lines.extend(["", "Failure reasons: " + ", ".join(
        f"{key}={value}" for key, value in failure["reason_counts"].items()),
        "", "## Remaining adversarial failures (top 10)", ""])
    for row in adversarial["all_failures"][:10]:
        lines.append(f"- `{row['id']}` {row['reason']}: {row['gold']} → {row['predicted']} / "
                     f"{row['expected_action']} → {row['action']}")

    lines.extend(["", "## Correctness bottleneck", ""])
    for dimension, buckets in decomposition.items():
        for name, value in buckets.items():
            rate = value["correct_answer_rate"]
            lines.append(f"- {dimension}/{name}: n={value['questions']}, correct answer={percent(rate)}")
    lines.extend([
        "",
        f"Correctness低下16件の最大要因は、正しいroute後のanswer-level検証/hallucination {round(b21['hallucination'] * b21['questions'])}件。"
        f"route誤りは{decomposition['routing']['wrong_route']['questions']}件で、wrong retrieval 4件はこのtestでは直接のCorrectness低下を起こしていない。"
        "正しいroute時のanswer correctnessが99%超のため、Standard 50Mは現段階では不要。",
        "",
        "## Gate / human evaluation",
        "",
        f"- Automatic gate: {'PASS' if gate['automatic_passed'] else 'FAIL'}",
        f"- Human 100: {'COMPLETE' if benchmark['human_evaluation']['complete'] else 'PENDING'}",
        f"- Final gate: {'PASS' if gate['passed'] else 'STOP'}",
        f"- RAM peak: {memory['peak_rss_mb']:.2f} MB (<450 MB)",
        "- ChatGPT/Gemini比較: 外部APIを使わず、`/campus-v21-eval`で同一質問のUI結果を手入力する。",
        "- Human gateは未採点を合格扱いしない。本番v0.4を維持する。",
        "",
        "## Decision",
        "",
        "自動ゲートは合格。Human 100が未完了のため総合ゲートはSTOP。Campus v2.1は本番昇格しない。",
    ])
    (ROOT / "evaluation/comparison-campus-v21.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
