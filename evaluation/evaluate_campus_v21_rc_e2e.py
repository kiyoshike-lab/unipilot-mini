"""Run 25 deterministic end-to-end Campus v2.1 RC user flows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.campus_v21 import UniPilotCampusV21


MANIFEST = ROOT / "evaluation/campus-v21-rc-manifest.json"
OUTPUT = ROOT / "evaluation/campus-v21-rc-e2e.json"


def verify_frozen() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for relative, expected in manifest["logic_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"RC answer logic changed: {relative}")
    return manifest


def contains(text: str) -> Callable[[dict], bool]:
    return lambda result: text in result["text"]


def main() -> None:
    manifest = verify_frozen()
    pipeline = UniPilotCampusV21(model=None, tokenizer=None)
    scenarios: list[dict] = [
        {"id": "E2E-01", "name": "GPA input card", "question": "GPAを計算したい", "expected": "tool=gpa, missing=courses",
         "check": lambda r: r["tool"] == "gpa" and "courses" in r["missing_fields"] and bool(r["cards"])},
        {"id": "E2E-02", "name": "GPA calculation", "question": "GPAを計算して", "inputs": {"courses": [{"name": "A", "grade": "A", "credits": 2}, {"name": "B", "grade": "B", "credits": 2}]},
         "expected": "GPA=2.5", "check": lambda r: r["calculation"]["gpa"] == 2.5 and bool(r["cards"][0]["copy_text"])},
        {"id": "E2E-03", "name": "Target GPA", "question": "目標GPAに必要な成績を計算したい", "inputs": {"current_gpa": 2.5, "current_credits": 60, "target_gpa": 3.0, "future_credits": 30},
         "expected": "required future GPA=4.0", "check": lambda r: r["tool"] == "gpa_target" and r["calculation"]["required_future_gpa"] == 4.0},
        {"id": "E2E-04", "name": "Required score", "question": "単位を取るための必要点数を計算して", "inputs": {"earned_points": 40, "target_points": 60, "remaining_weight": 30},
         "expected": "required=66.67%", "check": lambda r: r["tool"] == "grade_simulator" and 66.6 < r["calculation"]["required_average_percent"] < 66.7},
        {"id": "E2E-05", "name": "Professor email", "question": "教授に相談メールを書きたい", "expected": "editable email card",
         "check": lambda r: r["tool"] == "professor_email" and r["cards"][0]["kind"] == "email" and "［先生名］" in r["text"]},
        {"id": "E2E-06", "name": "Absence email", "question": "授業を欠席するメールを先生に送りたい", "expected": "absence email",
         "check": lambda r: r["tool"] == "absence_email" and "欠席" in r["text"]},
        {"id": "E2E-07", "name": "Lateness email", "question": "授業に遅刻するので教授への連絡文を作って", "expected": "lateness email",
         "check": lambda r: r["tool"] == "lateness_email" and "到着予定時刻" in r["text"]},
        {"id": "E2E-08", "name": "Late submission email", "question": "課題が締切に遅れるので先生へのメールを作って", "expected": "late submission email",
         "check": lambda r: r["tool"] == "late_submission_email" and "提出可能日時" in r["text"]},
        {"id": "E2E-09", "name": "Study plan input", "question": "テスト勉強の計画を立てたい", "expected": "three missing fields",
         "check": lambda r: r["tool"] == "study_plan" and set(r["missing_fields"]) == {"subject", "days", "hours_per_day"}},
        {"id": "E2E-10", "name": "Study plan complete", "question": "統計の試験勉強計画を作って", "inputs": {"subject": "統計", "days": 5, "hours_per_day": 2},
         "expected": "5-day plan", "check": lambda r: r["tool"] == "study_plan" and len(r["calculation"]["plan"]) == 5 and "5日間" in r["text"]},
        {"id": "E2E-11", "name": "Assignment priority", "question": "複数の課題の優先順位を決めたい", "inputs": {"assignments": [{"name": "レポート", "days_remaining": 1, "estimated_hours": 3, "impact": 5}, {"name": "小テスト", "days_remaining": 3, "estimated_hours": 1, "impact": 2}]},
         "expected": "ordered assignments", "check": lambda r: r["tool"] == "assignment_priority" and r["calculation"]["assignments"][0]["name"] == "レポート"},
        {"id": "E2E-12", "name": "Deadline organization", "question": "課題の締切を整理して", "inputs": {"deadlines": [{"name": "B", "deadline": "2026-09-10", "estimated_hours": 2}, {"name": "A", "deadline": "2026-09-01", "estimated_hours": 4}]},
         "expected": "deadline sort", "check": lambda r: r["tool"] == "deadline_organizer" and r["calculation"]["deadlines"][0]["name"] == "A"},
        {"id": "E2E-13", "name": "Credit progress", "question": "卒業まであと何単位か計算して", "inputs": {"earned_credits": 90, "required_credits": 124},
         "expected": "remaining=34", "check": lambda r: r["tool"] == "credit_progress" and r["calculation"]["remaining_credits"] == 34},
        {"id": "E2E-14", "name": "Exam countdown", "question": "試験日まであと何日か知りたい。解答時間の配分も決めたい", "inputs": {"current_date": "2026-08-24", "exam_date": "2026-09-03"},
         "expected": "10 days", "check": lambda r: r["tool"] == "exam_countdown" and r["calculation"]["days_remaining"] == 10},
        {"id": "E2E-15", "name": "Report allocation", "question": "4000字レポートの文字数を節ごとに配分して", "inputs": {"target_characters": 4000},
         "expected": "4000-character allocation", "check": lambda r: r["tool"] == "report_allocation" and sum(r["calculation"].values()) == 4000},
        {"id": "E2E-16", "name": "Report outline", "question": "生成AIと大学教育についてレポート構成を作って", "inputs": {"topic": "生成AIと大学教育"},
         "expected": "report outline", "check": lambda r: r["tool"] == "report_outline" and "生成AIと大学教育" in r["text"]},
        {"id": "E2E-17", "name": "Citation check", "question": "著者・資料名・発行年・掲載元を入力して引用チェックしたい", "inputs": {"author": "山田太郎", "title": "大学教育", "year": "2025", "publisher": "大学出版"},
         "expected": "bibliographic fields complete", "check": lambda r: r["tool"] == "citation_check" and r["missing_fields"] == []},
        {"id": "E2E-18", "name": "Presentation allocation", "question": "10分の発表時間を配分して", "inputs": {"total_minutes": 10},
         "expected": "10-minute allocation", "check": lambda r: r["tool"] == "presentation_allocation" and abs(sum(r["calculation"].values()) - 10) < .01},
        {"id": "E2E-19", "name": "Time allocation", "question": "今週の勉強と課題の時間配分を作って", "inputs": {"available_hours": 20, "fixed_hours": 5},
         "expected": "time allocation", "check": lambda r: r["tool"] == "time_allocation" and sum(r["calculation"].values()) == 20},
        {"id": "E2E-20", "name": "Registration checklist", "question": "履修登録の確認リストを作りたい", "expected": "registration checklist",
         "check": lambda r: r["tool"] == "registration" and "履修要項" in r["text"]},
        {"id": "E2E-21", "name": "University-safe guidance", "question": "私の大学では欠席何回で単位を落としますか", "expected": "no unsupported policy assertion",
         "check": lambda r: r["category"] == "university_policy" and r["route"] == "safety" and "断定できません" in r["text"]},
        {"id": "E2E-22", "name": "Reviewed FAQ", "question": "GPAって何？", "expected": "reviewed FAQ route",
         "check": lambda r: r["route"] == "faq" and r["cards"][0]["kind"] == "faq" and r["validator"]["valid"]},
        {"id": "E2E-23", "name": "Ambiguous clarification", "question": "相談したい", "expected": "clarification card with options",
         "check": lambda r: r["route"] == "clarify" and len(r["cards"][0]["data"]["options"]) >= 2},
    ]
    records = []
    for scenario in scenarios:
        started = time.perf_counter()
        result = pipeline.answer(scenario["question"], session_id=scenario["id"], tool_inputs=scenario.get("inputs"))
        elapsed_ms = (time.perf_counter() - started) * 1000
        passed = bool(scenario["check"](result))
        records.append({key: scenario[key] for key in ("id", "name", "question", "expected")} | {
            "passed": passed, "elapsed_ms": elapsed_ms, "category": result["category"], "route": result["route"],
            "route_action": result.get("route_action"), "executed_action": result.get("executed_action"),
            "tool": result["tool"], "cards": len(result["cards"]), "missing_fields": result["missing_fields"],
            "validator_valid": result["validator"]["valid"], "answer": result["text"],
        })

    session_id = "E2E-24-session"
    first = pipeline.answer("テスト勉強の計画を立てたい", session_id=session_id)
    second = pipeline.answer("相談したい", session_id=session_id,
                             tool_inputs={"subject": "英語", "days": 4, "hours_per_day": 1.5})
    records.append({"id": "E2E-24", "name": "Two-turn input flow", "question": "study plan → input follow-up",
                    "expected": "pending intent continues and completes", "passed": first["missing_fields"] != [] and second["tool"] == "study_plan" and second["missing_fields"] == [],
                    "elapsed_ms": (first["timing"]["total_seconds"] + second["timing"]["total_seconds"]) * 1000,
                    "category": second["category"], "route": second["route"], "route_action": second.get("route_action"),
                    "executed_action": second.get("executed_action"), "tool": second["tool"], "cards": len(second["cards"]),
                    "missing_fields": second["missing_fields"], "validator_valid": second["validator"]["valid"], "answer": second["text"]})

    events = list(pipeline.iter_answer("教授に欠席メールを送りたい", session_id="E2E-25-stream"))
    stream_pass = len(events) == 1 and events[0]["phase"] == "complete" and events[0]["cards"] and events[0]["validator"]["valid"]
    records.append({"id": "E2E-25", "name": "Streaming terminal event", "question": "教授に欠席メールを送りたい",
                    "expected": "single complete NDJSON-compatible event", "passed": bool(stream_pass),
                    "elapsed_ms": events[0].get("seconds", 0) * 1000 if events else None,
                    "category": events[0].get("category") if events else None, "route": events[0].get("route") if events else None,
                    "route_action": None, "executed_action": None, "tool": None,
                    "cards": len(events[0].get("cards", [])) if events else 0, "missing_fields": [],
                    "validator_valid": events[0].get("validator", {}).get("valid") if events else False,
                    "answer": events[0].get("text", "") if events else ""})

    passed = sum(record["passed"] for record in records)
    payload = {"release_candidate": manifest["release_candidate"], "rc_source_commit": manifest["rc_source_commit"],
               "scenarios": len(records), "passed": passed, "failed": len(records) - passed,
               "success_rate": passed / len(records), "records": records, "external_ai_api": "OFF",
               "production_changed": False}
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"scenarios": len(records), "passed": passed,
                      "failures": [record["id"] for record in records if not record["passed"]]}, ensure_ascii=False))
    if passed != len(records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
