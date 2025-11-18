from fastapi.testclient import TestClient
from services.api.app.main import app

client = TestClient(app=app)


def test_health_live():
    response = client.get("/health/live")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    
def test_health_ready():
    response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"