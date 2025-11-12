from .schemas.schemas import RideFeatures
from .config import settings
import joblib
import numpy as np

_model = None

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(settings.MODEL_PATH)
    return _model

def predict_one(features: RideFeatures) -> float:
    model = load_model()
    x = np.array([[
        features.season,
        features.yr,
        features.mnth,
        features.hr,
        features.holiday,
        features.weekday,
        features.workingday,
        features.weathersit,
        features.temp,
        features.atemp,
        features.hum,
        features.windspeed,
        features.year,
        features.month,
        features.dayofweek
    ]], dtype=float)
    
    y = model.predict(x)
    return float(y[0])
