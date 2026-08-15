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
    assert training.status_code == 200 and training.json()["step"] == 5000
    assert comparison.status_code == 200 and len(comparison.json()["comparisons"]) == 30
