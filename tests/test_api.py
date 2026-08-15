from fastapi.testclient import TestClient
from api.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["external_ai_api"] == "OFF"


def test_v03_evaluation_and_training_endpoints():
    with TestClient(app) as client:
        evaluation = client.get("/evaluation/latest")
        training = client.get("/training/latest")
        comparison = client.get("/evaluation/comparison")
    assert evaluation.status_code == 200 and len(evaluation.json()["result"]["generations"]) == 300
    assert training.status_code == 200 and training.json()["step"] == 2000
    assert comparison.status_code == 200 and len(comparison.json()["comparisons"]) == 30


def test_v04_comparison_and_human_persistence(tmp_path, monkeypatch):
    import json
    import api.main as api_main
    human = tmp_path / "human.json"
    human.write_text(json.dumps([{"id": "x", "prompt": "質問", "model_answer": "回答", "score": None, "notes": ""}], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(api_main, "HUMAN_V04", human)
    with TestClient(app) as client:
        comparison = client.get("/evaluation/v03-v04")
        saved = client.post("/human-eval/v04", json={"item_id": "x", "score": 3, "notes": "確認済み"})
    assert comparison.status_code == 200 and len(comparison.json()["comparisons"]) == 30
    assert saved.status_code == 200
    assert json.loads(human.read_text(encoding="utf-8"))[0]["score"] == 3
