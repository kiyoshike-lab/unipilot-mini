from fastapi.testclient import TestClient
from api.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["external_ai_api"] == "OFF"
