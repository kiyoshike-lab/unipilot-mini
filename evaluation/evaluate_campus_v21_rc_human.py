"""Aggregate Campus v2.1 RC human scores without inventing missing judgments."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
HUMAN = ROOT / "evaluation/human-comparison-campus-v21.json"
BENCHMARK = ROOT / "evaluation/campus-v21-benchmark.json"
MANIFEST = ROOT / "evaluation/campus-v21-rc-manifest.json"
AUDIT = ROOT / "evaluation/campus-v21-human-audit.json"
KNOWN = ROOT / "evaluation/campus-v21-rc-known-issues.json"
E2E = ROOT / "evaluation/campus-v21-rc-e2e.json"
ROUTES = ROOT / "evaluation/campus-v21-rc-route-speed.json"
OUTPUT = ROOT / "evaluation/campus-v21-rc-human-report.json"
MARKDOWN = ROOT / "evaluation/campus-v21-rc-human-report.md"
AXES = ("correctness", "relevance", "actionable", "naturalness", "would_use_again")
PAIR_AXES = ("correctness", "specificity", "actionability", "readability", "would_use")
COMPETITORS = ("chatgpt", "gemini")


def score_complete(row: dict) -> bool:
    return row.get("issues_reviewed", False) and all(row.get("scores", {}).get(axis) is not None for axis in AXES)


def pair_complete(row: dict, competitor: str) -> bool:
    return bool(row.get(f"{competitor}_answer", "").strip()) and all(
        row.get("pairwise", {}).get(competitor, {}).get(axis, "unscored") != "unscored" for axis in PAIR_AXES)


def pair_outcome(row: dict, competitor: str) -> str | None:
    if not pair_complete(row, competitor):
        return None
    choices = row["pairwise"][competitor]
    balance = sum(1 if choices[axis] == "unipilot" else -1 if choices[axis] == "competitor" else 0
                  for axis in PAIR_AXES)
    return "win" if balance > 0 else "loss" if balance < 0 else "tie"


def pair_summary(rows: list[dict], competitor: str) -> dict:
    outcomes = [pair_outcome(row, competitor) for row in rows]
    counts = Counter(outcome for outcome in outcomes if outcome is not None)
    return {"completed": sum(outcome is not None for outcome in outcomes), "total": len(rows),
            "win": counts["win"], "tie": counts["tie"], "loss": counts["loss"]}


def correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def display(value: object) -> str:
    if value is None:
        return "PENDING"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    rows = json.loads(HUMAN.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    known = json.loads(KNOWN.read_text(encoding="utf-8"))
    e2e = json.loads(E2E.read_text(encoding="utf-8"))
    routes = json.loads(ROUTES.read_text(encoding="utf-8"))
    automatic = next(item["metrics"] for item in benchmark["comparisons"]["blind_2000"]
                     if item["variant"] == "Campus v2.1")

    scored = [row for row in rows if score_complete(row)]
    score_averages = {axis: statistics.mean(row["scores"][axis] for row in scored) if scored else None for axis in AXES}
    comparison = {competitor: pair_summary(rows, competitor) for competitor in COMPETITORS}
    specialist = [row for row in rows if row.get("specialist_domain")]
    specialist_outcomes = [pair_outcome(row, competitor) for row in specialist for competitor in COMPETITORS]
    specialist_counts = Counter(outcome for outcome in specialist_outcomes if outcome is not None)
    specialist_completed = sum(outcome is not None for outcome in specialist_outcomes)
    specialist_total = len(specialist) * len(COMPETITORS)
    specialist_summary = {"questions": len(specialist), "completed_comparisons": specialist_completed,
                          "total_comparisons": specialist_total, "win": specialist_counts["win"],
                          "tie": specialist_counts["tie"], "loss": specialist_counts["loss"],
                          "win_plus_tie_rate": ((specialist_counts["win"] + specialist_counts["tie"]) / specialist_completed
                                                if specialist_completed else None)}

    reviewed = [row for row in rows if row.get("issues_reviewed")]
    issue_counts = ({key: sum(bool(row.get("issue_flags", {}).get(key)) for row in reviewed)
                     for key in next(iter(rows))["issue_flags"]} if reviewed else
                    {key: None for key in next(iter(rows))["issue_flags"]})
    critical_rate = issue_counts["critical_error"] / len(reviewed) if reviewed else None
    policy_rate = issue_counts["university_policy_assertion"] / len(reviewed) if reviewed else None

    auto_x = [float(row["automatic_evaluation"]["answer_correct"]) for row in scored]
    human_y = [float(row["scores"]["correctness"]) for row in scored]
    human_correctness_percent = score_averages["correctness"] / 5 * 100 if scored else None
    auto_vs_human = {"automatic_correctness_percent": automatic["correctness"] * 100,
                     "human_correctness_percent": human_correctness_percent,
                     "human_minus_automatic_percentage_points": (human_correctness_percent - automatic["correctness"] * 100
                                                                  if human_correctness_percent is not None else None),
                     "pearson_point_biserial": correlation(auto_x, human_y),
                     "correlation_n": len(scored), "interpretation": "PENDING" if len(scored) < len(rows) else "READY"}

    all_scores_complete = len(scored) == len(rows) == 100
    all_pairs_complete = all(pair_complete(row, competitor) for row in rows for competitor in COMPETITORS)
    gate_checks = {
        "human_100_complete": all_scores_complete,
        "chatgpt_gemini_pairwise_complete": all_pairs_complete,
        "correctness_gte_4_2": score_averages["correctness"] >= 4.2 if all_scores_complete else None,
        "relevance_gte_4_2": score_averages["relevance"] >= 4.2 if all_scores_complete else None,
        "actionable_gte_4_2": score_averages["actionable"] >= 4.2 if all_scores_complete else None,
        "naturalness_gte_4_2": score_averages["naturalness"] >= 4.2 if all_scores_complete else None,
        "would_use_again_gte_4_0": score_averages["would_use_again"] >= 4.0 if all_scores_complete else None,
        "critical_error_rate_lte_0_01": critical_rate <= .01 if all_scores_complete else None,
        "university_policy_assertion_rate_lte_0_01": policy_rate <= .01 if all_scores_complete else None,
        "specialist_win_plus_tie_gte_0_80": (specialist_summary["win_plus_tie_rate"] >= .80
                                             if specialist_completed == specialist_total else None),
    }
    gate_ready = all_scores_complete and all_pairs_complete and specialist_completed == specialist_total
    gate_passed = gate_ready and all(value is True for key, value in gate_checks.items()
                                     if key not in ("human_100_complete", "chatgpt_gemini_pairwise_complete"))
    gate_status = "PASS" if gate_passed else "FAIL" if gate_ready else "PENDING"

    model_rows = [row for row in scored if "MODEL" in (row.get("campus_metadata", {}).get("action") or "")]
    non_model_rows = [row for row in scored if row not in model_rows]
    model_mean = statistics.mean(row["scores"]["correctness"] for row in model_rows) if model_rows else None
    non_model_mean = statistics.mean(row["scores"]["correctness"] for row in non_model_rows) if non_model_rows else None
    standard_candidate = (gate_ready and len(model_rows) >= 5 and model_mean is not None and non_model_mean is not None
                          and model_mean < 4.2 <= non_model_mean)

    known_items = [item for group in known["groups"].values() for item in group]
    known_reviewed = sum(item["human_review"]["status"] != "pending" for item in known_items)
    report = {
        "release_candidate": manifest["release_candidate"], "rc_source_commit": manifest["rc_source_commit"],
        "automatic_gate": "PASS", "human_evaluation": {"status": "COMPLETE" if all_scores_complete else "PENDING",
            "completed": len(scored), "total": len(rows), "averages_0_to_5": score_averages},
        "chatgpt_comparison": comparison["chatgpt"], "gemini_comparison": comparison["gemini"],
        "specialist_comparison": specialist_summary, "automatic_vs_human": auto_vs_human,
        "human_issue_counts": {"reviewed_questions": len(reviewed), "total_questions": len(rows), **issue_counts,
                               "critical_error_rate": critical_rate, "university_policy_assertion_rate": policy_rate},
        "known_issue_review": {"status": "COMPLETE" if known_reviewed == len(known_items) else "PENDING",
            "reviewed": known_reviewed, "total": len(known_items), "automatic_candidates": known["counts"]},
        "question_audit": audit, "end_to_end": {"passed": e2e["passed"], "scenarios": e2e["scenarios"],
                                                  "success_rate": e2e["success_rate"]},
        "route_mix": routes["canonical_route_mix"],
        "non_model_generation_share": routes["actual_model_generation"]["non_model_share"],
        "planned_model_assisted_action_share": routes["planned_model_assisted_actions"]["share"],
        "under_one_second_share": routes["local_latency"]["under_one_second_share"],
        "latency_scope": routes["measurement_scope"], "human_production_gate": {"status": gate_status, "checks": gate_checks},
        "production_promotion_recommended": "YES" if gate_passed else "NO",
        "beta_start_recommended": "YES" if gate_passed else "NO",
        "campus_v22_needed": "PENDING_HUMAN_REVIEW; router/retrieval candidates recorded",
        "standard_50m_needed": "CANDIDATE" if standard_candidate else "NO; remains stopped",
        "model_route_human_correctness": {"questions": len(model_rows), "mean_0_to_5": model_mean,
                                           "non_model_mean_0_to_5": non_model_mean},
        "decision": ("PRODUCTION_CANDIDATE; deployment still requires a separate explicit action" if gate_passed else
                     "STOP; keep production v0.4 while Human Gate is pending or failed"),
        "external_ai_api": "OFF", "production_changed": False, "push_or_deploy_performed": False,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# UniPilot Campus v2.1 RC Human Gate", "",
        f"- RC: `{manifest['rc_source_commit']}`", f"- Human evaluation: {len(scored)}/100 ({report['human_evaluation']['status']})",
        f"- Correctness: {display(score_averages['correctness'])}", f"- Relevance: {display(score_averages['relevance'])}",
        f"- Actionable: {display(score_averages['actionable'])}", f"- Naturalness: {display(score_averages['naturalness'])}",
        f"- Would use again: {display(score_averages['would_use_again'])}",
        f"- ChatGPT Win/Tie/Loss: {comparison['chatgpt']['win']}/{comparison['chatgpt']['tie']}/{comparison['chatgpt']['loss']} (completed {comparison['chatgpt']['completed']}/100)",
        f"- Gemini Win/Tie/Loss: {comparison['gemini']['win']}/{comparison['gemini']['tie']}/{comparison['gemini']['loss']} (completed {comparison['gemini']['completed']}/100)",
        f"- Specialist Win/Tie/Loss: {specialist_summary['win']}/{specialist_summary['tie']}/{specialist_summary['loss']} (completed {specialist_completed}/{specialist_total})",
        f"- Automatic correctness: {automatic['correctness'] * 100:.2f}% / Human: {display(human_correctness_percent)}% / difference: {display(auto_vs_human['human_minus_automatic_percentage_points'])}pp",
        f"- Critical/router/retrieval/tool/model human errors: {display(issue_counts['critical_error'])}/{display(issue_counts['router_error'])}/{display(issue_counts['retrieval_error'])}/{display(issue_counts['tool_error'])}/{display(issue_counts['model_error'])}",
        f"- Known automatic candidates: hallucination {known['counts']['hallucination']}, router {known['counts']['router']}, retrieval {known['counts']['retrieval']} (human reviewed {known_reviewed}/23)",
        f"- E2E: {e2e['passed']}/{e2e['scenarios']} ({e2e['success_rate'] * 100:.1f}%)",
        f"- Actual non-model path: {routes['actual_model_generation']['non_model_share'] * 100:.1f}% (planned action labels containing MODEL: {routes['planned_model_assisted_actions']['share'] * 100:.1f}%)",
        f"- Local <1 second: {routes['local_latency']['under_one_second_share'] * 100:.1f}% ({routes['measurement_scope']})",
        f"- Human Production Gate: {gate_status}", f"- Production promotion: {report['production_promotion_recommended']}",
        f"- Beta start: {report['beta_start_recommended']}", f"- Campus v2.2: {report['campus_v22_needed']}",
        f"- Standard 50M: {report['standard_50m_needed']}", "",
        "Human scores and competitor answers are not present yet. Missing judgments are reported as PENDING, never as zero or synthetic scores.",
    ]
    MARKDOWN.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"human_gate": gate_status, "human_completed": len(scored), "production": report["production_promotion_recommended"],
                      "beta": report["beta_start_recommended"], "e2e": report["end_to_end"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
