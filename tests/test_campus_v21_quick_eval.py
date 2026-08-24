from collections import Counter
import json
from pathlib import Path

from fastapi.testclient import TestClient

import api.main as api_main


def read(path: str | Path) -> dict | list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def quick_item(index: int, rating: str | None = None) -> dict:
    return {
        "id": f"quick-{index:02d}", "source_id": f"source-{index:02d}", "question": f"question {index}",
        "category": f"category-{index}", "difficulty": "easy", "evaluation_bucket": "easy",
        "campus_answer": f"answer {index}", "campus_metadata": {"action": "FAQ", "route": "faq"},
        "focus": f"focus {index}", "quick_rating": rating, "quick_reason": None, "quick_scored_at": None,
    }


def configure_quick_files(tmp_path, monkeypatch):
    source_path = tmp_path / "human-100.json"
    selection_path = tmp_path / "selection.json"
    results_path = tmp_path / "quick-results.json"
    report_path = tmp_path / "quick-report.md"
    source = [quick_item(index) for index in range(20)]
    selection = {"items": [{"item_id": item["id"], "focus": item["focus"]} for item in source]}
    source_path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    selection_path.write_text(json.dumps(selection, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(api_main, "HUMAN_CAMPUS_V21", source_path)
    monkeypatch.setattr(api_main, "HUMAN_CAMPUS_V21_QUICK_SELECTION", selection_path)
    monkeypatch.setattr(api_main, "HUMAN_CAMPUS_V21_QUICK_RESULTS", results_path)
    monkeypatch.setattr(api_main, "HUMAN_CAMPUS_V21_QUICK_REPORT", report_path)
    monkeypatch.setattr(api_main, "load_runtime", lambda checkpoint=None: None)
    return results_path, report_path


def test_fixed_quick_selection_has_20_balanced_representative_questions():
    selection = read("evaluation/campus-v21-quick-selection.json")
    source = {item["id"]: item for item in read("evaluation/human-comparison-campus-v21.json")}
    selected = [source[entry["item_id"]] for entry in selection["items"]]
    assert len(selected) == len({item["id"] for item in selected}) == 20
    assert Counter(item["evaluation_bucket"] for item in selected) == {
        "easy": 5, "medium": 5, "hard": 5, "compound_ambiguous": 5,
    }
    assert len({item["category"] for item in selected}) >= 18
    actions = {item["campus_metadata"]["action"] for item in selected}
    assert {"FAQ", "TOOL", "RAG", "MODEL", "CLARIFY"} <= actions
    assert all(item["rc_source_commit"].startswith("0dc1878") for item in selected)


def test_quick_gate_rules_require_20_completed_items():
    pending = api_main.build_campus_v21_quick_summary([quick_item(index, "good") for index in range(19)])
    passed = api_main.build_campus_v21_quick_summary([quick_item(index, "good") for index in range(20)])
    improvement = api_main.build_campus_v21_quick_summary(
        [quick_item(index, "good" if index < 15 else "close" if index < 19 else "bad") for index in range(20)])
    failed = api_main.build_campus_v21_quick_summary(
        [quick_item(index, "good" if index < 18 else "bad") for index in range(20)])
    assert pending["quick_human_gate"]["status"] == "PENDING"
    assert passed["quick_human_gate"]["status"] == "PASS_CANDIDATE"
    assert improvement["rates_percent"] == {"good": 75.0, "close": 20.0, "bad": 5.0}
    assert improvement["quick_human_gate"]["status"] == "NEEDS_IMPROVEMENT"
    assert failed["quick_human_gate"]["status"] == "FAIL"


def test_three_choice_save_and_reload_restores_progress(tmp_path, monkeypatch):
    results_path, _ = configure_quick_files(tmp_path, monkeypatch)
    with TestClient(api_main.app) as client:
        first = client.get("/human-eval/campus-v21/quick")
        saved = client.post("/human-eval/campus-v21/quick", json={"item_id": "quick-00", "rating": "good"})
        restored = client.get("/human-eval/campus-v21/quick")
    assert first.status_code == saved.status_code == restored.status_code == 200
    assert saved.json()["summary"]["completed"] == 1
    assert restored.json()["completed"] == 1 and restored.json()["pending"] == 19
    assert restored.json()["items"][0]["quick_rating"] == "good"
    assert read(results_path)["items"][0]["rating"] == "good"


def test_bad_reason_is_optional_and_can_be_added_later(tmp_path, monkeypatch):
    configure_quick_files(tmp_path, monkeypatch)
    with TestClient(api_main.app) as client:
        initial = client.post("/human-eval/campus-v21/quick", json={"item_id": "quick-00", "rating": "bad"})
        updated = client.post("/human-eval/campus-v21/quick", json={
            "item_id": "quick-00", "rating": "bad", "reason": "router",
        })
        restored = client.get("/human-eval/campus-v21/quick")
    assert initial.status_code == updated.status_code == 200
    assert initial.json()["reason"] is None and updated.json()["reason"] == "router"
    assert restored.json()["items"][0]["quick_reason"] == "router"


def test_20_of_20_completion_and_export(tmp_path, monkeypatch):
    results_path, report_path = configure_quick_files(tmp_path, monkeypatch)
    with TestClient(api_main.app) as client:
        for index in range(20):
            response = client.post("/human-eval/campus-v21/quick", json={
                "item_id": f"quick-{index:02d}", "rating": "good" if index < 16 else "close",
            })
            assert response.status_code == 200
        exported = client.post("/human-eval/campus-v21/quick/export")
    assert exported.status_code == 200
    assert exported.json()["summary"]["status"] == "COMPLETE"
    assert exported.json()["summary"]["quick_human_gate"]["status"] == "PASS_CANDIDATE"
    assert results_path.exists() and report_path.exists()
    assert read(results_path)["completed"] == 20
    assert "Progress: 20 / 20" in report_path.read_text(encoding="utf-8")


def test_quick_ui_has_keyboard_auto_advance_and_position_restore_contract():
    page = Path("web/app/campus-v21-quick-eval/page.tsx").read_text(encoding="utf-8")
    assert 'event.key === "1"' in page and 'event.key === "2"' in page and 'event.key === "3"' in page
    assert "moveToNextPending(updated, item.id)" in page
    assert "localStorage.getItem(POSITION_KEY)" in page and "localStorage.setItem(POSITION_KEY" in page
