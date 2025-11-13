from pydantic import BaseModel, Field
from typing import List, Optional

class RideFeatures(BaseModel):
    season: int
    yr: int
    mnth: int
    hr: int
    holiday: int
    weekday: int
    workingday: int
    weathersit: int
    temp: float
    atemp: float
    hum: float
    windspeed: float
    year: int
    month: int
    dayofweek: int

class PredictionRequest(BaseModel):
    instances: List[RideFeatures]
    inverse_transform: bool = True

class PredictionResponse(BaseModel):
    predictions: List[float]
    model_version: Optional[str] = None