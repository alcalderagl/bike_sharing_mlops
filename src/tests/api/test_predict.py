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
                "dteday": "2012-11-20",
                "hr": 8,
                "temp": 0.45,
                "hum": 0.6,
                "windspeed": 0.15,
                "weathersit": 2,
                "cnt_lag_1": 100.0,
                "cnt_lag_24": 250.0,
                "season": 3,
                "yr": 1,
                "mnth": 11,
                "weekday": 2
            }
        ],
        "inverse_transform": True
    }

    response = client.post("/v1/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["predictions"] == [152]

