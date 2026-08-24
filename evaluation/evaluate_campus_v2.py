from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
from time import perf_counter

from pipeline.campus_retrieval import load_jsonl
from pipeline.campus_tools_v2 import CampusToolEngineV2
from pipeline.campus_v1 import UniPilotCampusV1
from pipeline.campus_v2 import UniPilotCampusV2
from pipeline.campus_validator_v2 import CampusValidatorV2


ROOT = Path(__file__).resolve().parents[1]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def memory_mb() -> dict:
    try:
        import os
        import psutil
        info = psutil.Process(os.getpid()).memory_info()
        return {"rss_mb": info.rss / 1024**2,
                "peak_rss_mb": getattr(info, "peak_wset", info.rss) / 1024**2}
    except (ImportError, OSError):
        return {"rss_mb": None, "peak_rss_mb": None}


def hallucination(answer: str, forbidden: list[str]) -> bool:
    value = answer.replace("必ず認められるとは限りません", "認められるかは公式確認が必要です")
    return any(claim in value for claim in forbidden) or any(claim in value for claim in (
        "必ず追試", "必ず単位", "全国の大学で", "どの大学でも同じ",
    ))


def natural(answer: str) -> bool:
    if not answer.strip() or "�" in answer:
        return False
    japanese = sum("ぁ" <= char <= "ん" or "ァ" <= char <= "ヶ" or "一" <= char <= "龥" for char in answer)
    return japanese / max(1, len(answer)) >= .25


def v1_action(result: dict) -> str:
    if result["route"] == "tool": return "TOOL"
    if result["route"] == "faq": return "FAQ"
    if result["route"] in ("official", "safety"): return "RAG"
    if result["route"] == "model" and result.get("retrieval"): return "RAG+MODEL"
    return "MODEL"


def evaluate_pipeline(name: str, pipeline, rows: list[dict]) -> tuple[dict, dict[str, dict]]:
    records, outputs = [], {}
    route_latencies: dict[str, list[float]] = defaultdict(list)
    router_latencies = []
    for index, item in enumerate(rows):
        started = perf_counter()
        result = pipeline.answer(item["prompt"], max_new_tokens=32)
        elapsed = perf_counter() - started
        outputs[item["id"]] = result
        action = result.get("route_action") or v1_action(result)
        category_correct = result["category"] == item["category"]
        action_correct = action == item["expected_action"]
        invented = hallucination(result["text"], item["forbidden_claims"])
        valid = result.get("validator", {}).get("valid", True)
        correct = category_correct and not invented and valid
        relevance = category_correct and len(result["text"].strip()) >= 16
        completion = len(result["text"].strip()) >= 16 or bool(result.get("cards"))
        multi_expected = set(item["intent_labels"])
        multi_predicted = set(result.get("intents", [result["category"]]))
        multi_recall = len(multi_expected & multi_predicted) / len(multi_expected)
        router_ms = result.get("router", {}).get("latency_ms")
        if router_ms is not None:
            router_latencies.append(router_ms)
        route_latencies[result["route"]].append(elapsed * 1000)
        records.append({
            "id": item["id"], "surface_type": item["surface_type"], "expected_category": item["category"],
            "predicted_category": result["category"], "expected_action": item["expected_action"],
            "predicted_action": action, "category_correct": category_correct, "action_correct": action_correct,
            "correctness": correct, "relevance": relevance, "hallucination": invented,
            "completion": completion, "natural_japanese": natural(result["text"]),
            "actionable_score": result.get("validator", {}).get("actionable_score", 0),
            "clarification_correct": action == "CLARIFY" if item["surface_type"] == "ambiguous" else None,
            "multi_intent_recall": multi_recall if len(multi_expected) > 1 else None,
            "response_ms": elapsed * 1000, "router_ms": router_ms,
        })
        if (index + 1) % 500 == 0:
            print(name, index + 1)
    subset = lambda key: [record for record in records if record[key] is not None]
    per_surface = {}
    for surface in sorted({record["surface_type"] for record in records}):
        group = [record for record in records if record["surface_type"] == surface]
        per_surface[surface] = {"questions": len(group),
                                "category_accuracy": statistics.fmean(record["category_correct"] for record in group),
                                "action_accuracy": statistics.fmean(record["action_correct"] for record in group)}
    metrics = {
        "questions": len(records),
        "category_accuracy": statistics.fmean(record["category_correct"] for record in records),
        "category_accuracy_routable_only": statistics.fmean(record["category_correct"] for record in records if record["surface_type"] != "ambiguous"),
        "action_accuracy": statistics.fmean(record["action_correct"] for record in records),
        "correctness": statistics.fmean(record["correctness"] for record in records),
        "relevance": statistics.fmean(record["relevance"] for record in records),
        "hallucination": statistics.fmean(record["hallucination"] for record in records),
        "completion": statistics.fmean(record["completion"] for record in records),
        "natural_japanese": statistics.fmean(record["natural_japanese"] for record in records),
        "actionable_score": statistics.fmean(record["actionable_score"] for record in records),
        "clarification_accuracy": statistics.fmean(record["clarification_correct"] for record in subset("clarification_correct")),
        "multi_intent_recall": statistics.fmean(record["multi_intent_recall"] for record in subset("multi_intent_recall")),
        "mean_latency_ms": statistics.fmean(record["response_ms"] for record in records),
        "p95_latency_ms": percentile([record["response_ms"] for record in records], .95),
        "router_p95_ms": percentile(router_latencies, .95),
        "route_latency_ms": {route: {"count": len(values), "mean": statistics.fmean(values),
                                      "p95": percentile(values, .95)} for route, values in route_latencies.items()},
        "per_surface": per_surface,
    }
    return {"variant": name, "metrics": metrics, "records": records}, outputs


def retrieval_benchmark(pipeline: UniPilotCampusV2, rows: list[dict]) -> dict:
    reciprocal, recall1, recall3, latencies = [], 0, 0, []
    evaluated = 0
    for item in rows:
        if item["surface_type"] == "ambiguous":
            continue
        relevant = set(item["relevant_faq_ids"])
        started = perf_counter(); found = pipeline.faq.search(item["prompt"], item["category"], 3)
        latencies.append((perf_counter() - started) * 1000)
        identifiers = [row["id"] for row in found]
        ranks = [index + 1 for index, identifier in enumerate(identifiers) if identifier in relevant]
        recall1 += bool(ranks and ranks[0] <= 1); recall3 += bool(ranks and ranks[0] <= 3)
        reciprocal.append(1 / ranks[0] if ranks else 0.0); evaluated += 1
    return {"questions": evaluated, "relevance_granularity": "category-level relevant FAQ set; exact duplicates are excluded",
            "recall_at_1": recall1 / evaluated, "recall_at_3": recall3 / evaluated,
            "mrr": statistics.fmean(reciprocal), "mean_latency_ms": statistics.fmean(latencies),
            "p95_latency_ms": percentile(latencies, .95)}


def validator_benchmark() -> dict:
    validator = CampusValidatorV2()
    faq = load_jsonl(ROOT / "data" / "campus_v2" / "faq" / "reviewed.jsonl")
    cases = []
    for index, item in enumerate(faq[:250]):
        cases.append((True, item["question"], item["answer"], {"grounded": True, "source_urls": ["project-authored"]}))
    bad_answers = (
        "この大学では欠席3回で単位を落とすと決まっています。",
        "",
        "計算結果：残り42点です。",
        "結論：［ここに回答を入力］ 今やること：確認する。",
        "件名：相談 先生 序論 本論 結論 ES 面接 企業 計算結果 式：1+1=2",
    )
    for index in range(250):
        cases.append((False, "大学の相談", bad_answers[index % len(bad_answers)], {}))
    tp = fp = tn = fn = 0
    for expected_valid, question, answer, kwargs in cases:
        predicted_valid = validator.validate(question, answer, **kwargs).valid
        if not expected_valid and not predicted_valid: tp += 1
        elif expected_valid and not predicted_valid: fp += 1
        elif expected_valid and predicted_valid: tn += 1
        else: fn += 1
    return {"cases": len(cases), "invalid_detection_precision": tp / (tp + fp),
            "invalid_detection_recall": tp / (tp + fn), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def tool_audit() -> dict:
    engine = CampusToolEngineV2(); state = {}
    cases = {
        "gpa": ("GPA計算", {"courses": [{"grade": "A", "credits": 2}, {"grade": "B", "credits": 2}]}),
        "grade_simulator": ("必要点", {"earned_points": 40, "target_points": 60, "remaining_weight": 30}),
        "professor_email": ("教授メール", {}), "absence_email": ("欠席メール", {}),
        "lateness_email": ("遅刻メール", {}), "late_submission_email": ("提出遅延メール", {}),
        "registration": ("履修を整理", {}), "study_plan": ("勉強計画", {"subject": "数学", "days": 5, "hours_per_day": 2}),
        "assignment_priority": ("課題優先", {"assignments": [{"name": "A", "days_remaining": 1, "estimated_hours": 2}]}),
        "deadline_organizer": ("締切整理", {"deadlines": [{"name": "A", "deadline": "2026-09-01"}]}),
        "report_outline": ("構成", {"topic": "問い"}),
        "citation_check": ("引用", {"author": "著者", "title": "題", "year": "2026", "publisher": "掲載元"}),
        "presentation_outline": ("発表構成", {"topic": "テーマ"}), "career_schedule": ("就活計画", {}),
        "es_outline": ("ES構成", {}), "toeic_plan": ("TOEIC計画", {"days": 30, "hours_per_day": 1}),
        "gpa_target": ("目標GPA", {"current_gpa": 2.5, "current_credits": 60, "target_gpa": 3.0, "future_credits": 30}),
        "credit_progress": ("取得単位の進捗", {"earned_credits": 80, "required_credits": 124}),
        "exam_countdown": ("あと何日", {"current_date": "2026-08-24", "exam_date": "2026-09-10"}),
        "report_allocation": ("2000字の文字数配分", {"target_characters": 2000}),
        "presentation_allocation": ("10分の時間配分", {"total_minutes": 10}),
        "time_allocation": ("時間配分", {"available_hours": 10, "fixed_hours": 4}),
    }
    dispatch = {"gpa_target": "gpa", "credit_progress": "credit", "exam_countdown": "exam",
                "report_allocation": "report_outline", "presentation_allocation": "presentation_outline",
                "time_allocation": "schedule"}
    rows = []
    for name, (question, inputs) in cases.items():
        intent = dispatch.get(name, name)
        first = engine.execute(intent, question, state, inputs); second = engine.execute(intent, question, state, inputs)
        rows.append({"tool": name, "completed": first.completed, "deterministic": first.text == second.text and first.calculation == second.calculation,
                     "cards": len(first.cards), "external_ai": False,
                     "unprompted_subject": any(subject in first.text for subject in ("法学", "経済学", "心理学"))})
    return {"tools": len(rows), "passed": sum(row["completed"] and row["deterministic"] and not row["unprompted_subject"] for row in rows),
            "rows": rows}


def update_human(outputs: dict[str, dict]) -> None:
    source = json.loads((ROOT / "data" / "campus_v2" / "human" / "comparison-100.json").read_text(encoding="utf-8"))
    for item in source:
        result = outputs[item["blind_id"]]
        item["campus_answer"] = result["text"]
        item["campus_metadata"] = {"category": result["category"], "action": result.get("route_action"),
                                   "latency_ms": result["timing"]["total_seconds"] * 1000}
    (ROOT / "evaluation" / "human-comparison-campus-v2.json").write_text(
        json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    blind = json.loads((ROOT / "data" / "campus_v2" / "blind" / "evaluation-2000.json").read_text(encoding="utf-8"))
    initial = memory_mb(); v1 = UniPilotCampusV1(); after_v1 = memory_mb(); v2 = UniPilotCampusV2(); after_v2 = memory_mb()
    result_v1, _ = evaluate_pipeline("Campus v1", v1, blind)
    result_v2, outputs = evaluate_pipeline("Campus v2", v2, blind)
    retrieval = retrieval_benchmark(v2, blind)
    validator = validator_benchmark(); tools = tool_audit(); update_human(outputs)
    metrics = result_v2["metrics"]
    human_rows = json.loads((ROOT / "evaluation" / "human-comparison-campus-v2.json").read_text(encoding="utf-8"))
    human_complete = all(row["scores"]["correctness"] is not None for row in human_rows)
    gates = {
        "category_accuracy_gte_0_92": metrics["category_accuracy"] >= .92,
        "relevance_gte_0_90": metrics["relevance"] >= .90,
        "correctness_gte_0_88": metrics["correctness"] >= .88,
        "hallucination_lte_0_01": metrics["hallucination"] <= .01,
        "completion_gte_0_99": metrics["completion"] >= .99,
        "natural_japanese_gte_0_99": metrics["natural_japanese"] >= .99,
        "actionable_gte_4_2": metrics["actionable_score"] >= 4.2,
        "router_p95_lt_20ms": (metrics["router_p95_ms"] or 999) < 20,
        "faq_tool_p95_lt_50ms": all(value["p95"] < 50 for route, value in metrics["route_latency_ms"].items() if route in ("faq", "tool")),
        "human_100_complete": human_complete,
    }
    output = {
        "version": "unipilot-campus-v2", "blind": "data/campus_v2/blind/evaluation-2000.json",
        "memory": {"initial": initial, "after_v1": after_v1, "after_v2": after_v2},
        "variants": [result_v1, result_v2], "retrieval": retrieval, "validator": validator,
        "tool_audit": tools, "human_evaluation": {"questions": 100, "complete": human_complete,
            "distribution": dict(Counter(row["difficulty"] for row in human_rows)),
            "chatgpt_gemini_method": "manual UI entry only; no external API"},
        "production_gate": {"passed": all(gates.values()), "checks": gates,
                            "decision": "STOP; keep v0.4 production" if not all(gates.values()) else "eligible for explicit human approval"},
        "external_ai_api": "OFF",
    }
    (ROOT / "evaluation" / "campus-v2-benchmark.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"v1": result_v1["metrics"], "v2": result_v2["metrics"], "gate": output["production_gate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
