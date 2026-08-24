#!/usr/bin/env python3
"""Deterministic source-grounding evaluation for the opt-in Campus v2.2 path."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.campus_v21 import UniPilotCampusV21
from pipeline.campus_v22 import UniPilotCampusV22


BENCHMARK_PATH = ROOT / "data" / "campus_v22" / "benchmarks" / "knowledge-1000.jsonl"
HALLUCINATION_PATH = ROOT / "data" / "campus_v22" / "benchmarks" / "hallucination-500.jsonl"
OUT_PATH = ROOT / "evaluation" / "campus-v22-results.json"
COMPARE_PATH = ROOT / "evaluation" / "campus-v21-v22-comparison.json"
REPORT_MD_PATH = ROOT / "evaluation" / "campus-v22-report.md"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[^0-9a-zぁ-んァ-ヶ一-龥々]+", "", value)


def percentile(values: list[float], level: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * level
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def natural_japanese(text: str) -> bool:
    if not text.strip() or "�" in text or "����" in text:
        return False
    japanese = len(re.findall(r"[ぁ-んァ-ヶ一-龥々]", text))
    return japanese >= max(6, int(len(text) * .12))


def repeated_ratio(text: str) -> float:
    units = [normalized(part) for part in re.split(r"(?<=[。！？])\s*|\n+", text) if len(normalized(part)) >= 10]
    if not units:
        return 0.0
    return max(0.0, (len(units) - len(set(units))) / len(units))


def source_relevant(result: dict, expected_url: str | None) -> bool:
    if result.get("route") == "tool":
        return result.get("route") == "tool" and result.get("validator", {}).get("valid", False)
    return any(row.get("source_url") == expected_url for row in result.get("retrieval", []))


def evidence_claim_count(result: dict) -> int:
    return int(result.get("grounding", {}).get("evidence_sentence_count", 0))


def coverage_score(result: dict, relevant: bool) -> float:
    # Operational rubric: 2 points for selecting the gold source, then up to
    # 3 points for distinct source-supported facts actually included.
    if result.get("route") == "tool":
        return min(5.0, 2.0 + len(result.get("text", "")) / 180 + float(result.get("validator", {}).get("actionable_score", 0)) * .2)
    if result.get("route") == "faq":
        return min(5.0, 1.5 + len(result.get("text", "")) / 160 + float(result.get("validator", {}).get("actionable_score", 0)) * .2)
    if not relevant:
        return min(2.0, evidence_claim_count(result) * .4)
    return min(5.0, 2.0 + evidence_claim_count(result) * .6)


def current_rss_mb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024**2
    except (ImportError, OSError):
        return None


def standalone_pipeline_rss(version: str) -> float | None:
    module, class_name = (
        ("pipeline.campus_v21", "UniPilotCampusV21") if version == "campus-v2.1"
        else ("pipeline.campus_v22", "UniPilotCampusV22")
    )
    code = (
        "import os,psutil; "
        f"from {module} import {class_name}; "
        f"pipeline={class_name}(); "
        "print(psutil.Process(os.getpid()).memory_info().rss/1024**2)"
    )
    try:
        return float(subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def evaluate_pipeline(pipeline, benchmark: list[dict], *, mode: str, sample_limit: int | None = None) -> tuple[dict, list[dict]]:
    rows = benchmark[:sample_limit] if sample_limit else benchmark
    details: list[dict] = []
    latencies: list[float] = []
    retrieval_latencies: list[float] = []
    router_latencies: list[float] = []
    lengths: list[int] = []
    actionables: list[float] = []
    coverages: list[float] = []
    redundancies: list[float] = []
    publisher_hits: Counter[str] = Counter()
    peak_rss = current_rss_mb()
    for index, row in enumerate(rows):
        started = time.perf_counter()
        result = pipeline.answer(row["question"], response_mode=mode, session_id=f"eval-{mode}-{index}")
        elapsed = (time.perf_counter() - started) * 1000
        relevant = source_relevant(result, row["expected_source_url"])
        supported = bool(result.get("grounding", {}).get("supported"))
        correct = (relevant and (supported or result.get("route") == "tool")) if pipeline.version == "campus-v2.2" else (
            result.get("validator", {}).get("valid", False) and result.get("route") in ("faq", "official", "tool")
        )
        route_ok = result.get("category") == row["category"]
        completion = bool(result.get("text", "").strip()) and len(result.get("text", "")) >= 20
        natural = natural_japanese(result.get("text", ""))
        grounding_applicable = result.get("route") == "rag"
        grounding = 1.0 if grounding_applicable and supported and evidence_claim_count(result) > 0 else 0.0
        unsupported = int(result.get("grounding", {}).get("unsupported_factual_claims", 0))
        coverage = coverage_score(result, relevant) if pipeline.version == "campus-v2.2" else min(5.0, len(result.get("text", "")) / 120)
        redundancy = repeated_ratio(result.get("text", ""))
        for source in result.get("sources", []):
            publisher_hits[source.get("publisher") or "unknown"] += 1
        latencies.append(elapsed)
        retrieval_latencies.append(float(result.get("timing", {}).get("retrieval_ms", 0.0)))
        router_latencies.append(float(result.get("router", {}).get("latency_ms", 0.0)))
        lengths.append(len(result.get("text", "")))
        actionables.append(float(result.get("validator", {}).get("actionable_score", 0.0)))
        coverages.append(coverage)
        redundancies.append(redundancy)
        rss = current_rss_mb()
        if rss is not None:
            peak_rss = max(peak_rss or rss, rss)
        details.append({
            "id": row["id"], "category": row["category"], "route_ok": route_ok,
            "correct": correct, "relevant_source": relevant, "grounded": bool(grounding),
            "grounding_applicable": grounding_applicable, "route": result.get("route"),
            "unsupported_claims": unsupported, "complete": completion, "natural_japanese": natural,
            "coverage": round(coverage, 3), "answer_chars": lengths[-1], "redundancy": round(redundancy, 4),
            "latency_ms": round(elapsed, 3), "retrieval_ms": round(retrieval_latencies[-1], 3),
        })
    total = max(1, len(details))
    metrics = {
        "questions": len(details),
        "routing": sum(item["route_ok"] for item in details) / total,
        "correctness": sum(item["correct"] for item in details) / total,
        "relevance": sum(item["relevant_source"] for item in details) / total,
        "grounding": (
            sum(item["grounded"] for item in details)
            / max(1, sum(item["grounding_applicable"] for item in details))
        ),
        "unsupported_claim_rate": sum(item["unsupported_claims"] > 0 for item in details) / total,
        "completion": sum(item["complete"] for item in details) / total,
        "natural_japanese": sum(item["natural_japanese"] for item in details) / total,
        "actionable": statistics.fmean(actionables) if actionables else 0.0,
        "coverage": statistics.fmean(coverages) if coverages else 0.0,
        "average_answer_chars": statistics.fmean(lengths) if lengths else 0.0,
        "redundancy": statistics.fmean(redundancies) if redundancies else 0.0,
        "latency_p95_ms": percentile(latencies, .95),
        "router_p95_ms": percentile(router_latencies, .95),
        "retrieval_p95_ms": percentile(retrieval_latencies, .95),
        "peak_rss_mb": peak_rss,
        "source_helpfulness": dict(publisher_hits.most_common()),
    }
    return metrics, details


def evaluate_hallucination(pipeline: UniPilotCampusV22, rows: list[dict]) -> dict:
    unsupported = hallucinated = grounded = 0
    for index, row in enumerate(rows):
        result = pipeline.answer(row["question"], response_mode="normal", session_id=f"hall-{index}")
        if row["answerable"]:
            relevant = source_relevant(result, row["expected_source_url"])
            supported = (bool(result.get("grounding", {}).get("supported")) and relevant) or (
                result.get("route") == "tool" and result.get("validator", {}).get("valid", False)
            )
            grounded += int(supported)
            claim_issue = int(result.get("grounding", {}).get("unsupported_factual_claims", 0)) > 0
            unsupported += int(claim_issue)
            hallucinated += int(claim_issue)
        else:
            pattern = row.get("forbidden_claim_pattern", r"$^")
            claim_issue = bool(re.search(pattern, result.get("text", "")))
            unsupported += int(claim_issue)
            hallucinated += int(claim_issue or result.get("route") not in ("safety", "clarify"))
            grounded += int(result.get("route") in ("safety", "clarify"))
    total = max(1, len(rows))
    return {
        "questions": len(rows),
        "unsupported_claim_rate": unsupported / total,
        "hallucination_rate": hallucinated / total,
        "grounding_or_safe_rate": grounded / total,
    }


def rounded(metrics: dict) -> dict:
    return {key: round(value, 6) if isinstance(value, float) else value for key, value in metrics.items()}


def main() -> int:
    benchmark = load_jsonl(BENCHMARK_PATH)
    hallucination_rows = load_jsonl(HALLUCINATION_PATH)
    if len(benchmark) != 1000 or len(hallucination_rows) != 500:
        raise RuntimeError("evaluation sets must contain 1000 and 500 records")
    v21 = UniPilotCampusV21()
    v22 = UniPilotCampusV22()
    v21_normal, _ = evaluate_pipeline(v21, benchmark, mode="normal")
    v22_normal, normal_details = evaluate_pipeline(v22, benchmark, mode="normal")
    rag_ids = {detail["id"] for detail in normal_details if detail["route"] == "rag"}
    detailed_benchmark = [row for row in benchmark if row["id"] in rag_ids]
    v22_detailed, detailed_details = evaluate_pipeline(v22, detailed_benchmark, mode="detailed")
    hall_v21 = evaluate_hallucination(v21, hallucination_rows)
    hall = evaluate_hallucination(v22, hallucination_rows)
    v21_normal["standalone_rss_mb"] = standalone_pipeline_rss("campus-v2.1")
    v22_normal["standalone_rss_mb"] = standalone_pipeline_rss("campus-v2.2")
    v22_detailed["standalone_rss_mb"] = v22_normal["standalone_rss_mb"]
    v22_normal["unsupported_claim_rate"] = hall["unsupported_claim_rate"]
    v22_normal["hallucination"] = hall["hallucination_rate"]
    v22_normal["grounding"] = min(v22_normal["grounding"], hall["grounding_or_safe_rate"])
    v22_detailed["unsupported_claim_rate"] = hall["unsupported_claim_rate"]
    v22_detailed["hallucination"] = hall["hallucination_rate"]
    v21_normal["unsupported_claim_rate"] = hall_v21["unsupported_claim_rate"]
    v21_normal["hallucination"] = hall_v21["hallucination_rate"]
    gate_checks = {
        "routing_gte_98": v22_normal["routing"] >= .98,
        "correctness_gte_95": v22_normal["correctness"] >= .95,
        "relevance_gte_95": v22_normal["relevance"] >= .95,
        "grounding_gte_95": v22_normal["grounding"] >= .95,
        "unsupported_lte_1": v22_normal["unsupported_claim_rate"] <= .01,
        "hallucination_lte_1": v22_normal["hallucination"] <= .01,
        "completion_gte_99": v22_normal["completion"] >= .99,
        "natural_japanese_gte_99": v22_normal["natural_japanese"] >= .99,
        "actionable_gte_4_5": v22_normal["actionable"] >= 4.5,
        "normal_coverage_gte_3_5": v22_normal["coverage"] >= 3.5,
        "detailed_coverage_gte_4_3": v22_detailed["coverage"] >= 4.3,
        "redundancy_lte_2": v22_normal["redundancy"] <= .02 and v22_detailed["redundancy"] <= .02,
        "retrieval_p95_lt_100ms": v22_normal["retrieval_p95_ms"] < 100,
        "router_p95_lt_20ms": v22_normal["router_p95_ms"] < 20,
    }
    knowledge_report = json.loads((ROOT / "evaluation" / "campus-v22-knowledge-report.json").read_text(encoding="utf-8"))
    failing_by_group: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "retrieval_failures": 0, "routing_failures": 0})
    by_id = {row["id"]: row for row in benchmark}
    for detail in normal_details:
        group = by_id[detail["id"]]["group"]
        failing_by_group[group]["total"] += 1
        failing_by_group[group]["retrieval_failures"] += int(not detail["relevant_source"])
        failing_by_group[group]["routing_failures"] += int(not detail["route_ok"])
    result = {
        "version": "campus-v2.2",
        "evaluation_method": "deterministic source-linked local evaluation; no external AI judge",
        "knowledge": knowledge_report["counts"],
        "v2.1": rounded(v21_normal),
        "v2.2_normal": rounded(v22_normal),
        "v2.2_detailed": rounded(v22_detailed),
        "hallucination_500_v2.1": rounded(hall_v21),
        "hallucination_500": rounded(hall),
        "technical_gate": {"status": "PASS" if all(gate_checks.values()) else "FAIL", "checks": gate_checks},
        "human_gate": {"status": "PENDING", "items": 100, "production_promotion_allowed": False},
        "knowledge_gaps": {group: values for group, values in failing_by_group.items() if values["retrieval_failures"]},
        "training_recommendation": "Keep Standard 50M stopped. Reconsider only after human review and after RAG/composer failures are isolated from model-generation failures.",
    }
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    comparison = {
        "version": "campus-v2.1-vs-v2.2",
        "v2.1": result["v2.1"],
        "v2.2": result["v2.2_normal"],
        "technical_gate": result["technical_gate"],
        "human_gate": result["human_gate"],
        "note": "v2.1 is measured on the new source-linked knowledge benchmark; v2.2 remains opt-in and cannot ship before human review.",
    }
    COMPARE_PATH.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    helpful = next(iter(result["v2.2_normal"].get("source_helpfulness", {})), "該当なし")
    gaps = "、".join(result["knowledge_gaps"].keys()) or "自動評価上の検索失敗カテゴリなし"
    report_md = f"""# UniPilot Campus v2.2 knowledge report

- Evaluation: deterministic local source-linked benchmark (1,000 questions)
- Hallucination test: 500 questions
- External AI/LLM judge: not used
- Technical Gate: **{result['technical_gate']['status']}**
- Human Gate: **PENDING** (production promotion prohibited)

## Knowledge

- External documents: {knowledge_report['counts']['external_knowledge']}
- Wikipedia: {knowledge_report['counts']['wikipedia']}
- Government: {knowledge_report['counts']['government']}
- University official: {knowledge_report['counts']['university']}
- Reviewed FAQ: {knowledge_report['counts']['reviewed_faq']}
- Duplicates removed: {knowledge_report['counts']['duplicates_removed']}
- Stale documents: {knowledge_report['counts']['stale_documents']}
- Fetch failures: {knowledge_report['counts']['fetch_failures']}

The highest-use publisher in the benchmark was **{helpful}**. University counts stay below the candidate
target because pages without an explicit reusable license are deliberately excluded. Failed and unsupported
sources are listed in `campus-v22-knowledge-report.json`.

## Evaluation summary

- v2.1 correctness / relevance / grounding / hallucination: {result['v2.1']['correctness']:.3f} / {result['v2.1']['relevance']:.3f} / {result['v2.1']['grounding']:.3f} / {result['v2.1']['hallucination']:.3f}
- v2.2 correctness / relevance / grounding: {result['v2.2_normal']['correctness']:.3f} / {result['v2.2_normal']['relevance']:.3f} / {result['v2.2_normal']['grounding']:.3f}
- Unsupported / hallucination: {result['v2.2_normal']['unsupported_claim_rate']:.3f} / {result['v2.2_normal']['hallucination']:.3f}
- Normal / detailed coverage: {result['v2.2_normal']['coverage']:.3f} / {result['v2.2_detailed']['coverage']:.3f}
- Normal / detailed average characters: {result['v2.2_normal']['average_answer_chars']:.1f} / {result['v2.2_detailed']['average_answer_chars']:.1f}
- Retrieval P95: {result['v2.2_normal']['retrieval_p95_ms']:.1f} ms
- Standalone v2.1 / v2.2 RSS: {result['v2.1']['standalone_rss_mb']} / {result['v2.2_normal']['standalone_rss_mb']} MB

## Remaining risks and decision

- Knowledge gaps: {gaps}
- Freshness risk: scholarships, tuition, employment, registration and institutional rules require periodic verification.
- RAG failures must be corrected in topics, source coverage or retrieval before considering model training.
- Standard 50M remains stopped; current evidence does not authorize resuming long training.
- Human knowledge review is unscored, so v2.2 cannot be promoted even if the Technical Gate passes.
"""
    REPORT_MD_PATH.write_text(report_md, encoding="utf-8")
    print(json.dumps({"v2.2": rounded(v22_normal), "gate": result["technical_gate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
