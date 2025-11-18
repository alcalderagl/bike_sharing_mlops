from fastapi.testclient import TestClient
import numpy as np

from services.api.app.main import app
from services.api.app import inference

client = TestClient(app)


def fake_predict_df(df, inverse_transform=True):
    return np.array([42.0] * len(df), dtype=float)

def fake_get_version():
    return "test-version"


def test_predict_ok(monkeypatch):
    # parchear predict_df
    monkeypatch.setattr(inference, "predict_df", fake_predict_df)

    payload = {
        "instances": [
            {
                "season": 1,
                "yr": 1,
                "mnth": 1,
                "hr": 14,
                "holiday": 0,
                "weekday": 3,
                "workingday": 1,
                "weathersit": 2,
                "temp": 0.32,
                "atemp": 0.30,
                "hum": 0.58,
                "windspeed": 0.15,
                "year": 2024,
                "month": 10,
                "dayofweek": 1
            }
        ],
        "inverse_transform": True
    }

    response = client.post("/v1/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["predictions"] == [195.15814508877813]

