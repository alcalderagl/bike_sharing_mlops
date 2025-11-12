from pydantic import BaseModel, Field
from typing import List, Optional

class RideFeatures(BaseModel):
    season: int = Field(..., ge=1, le=4)       # 1: invierno ... 4: otoño
    yr: int = Field(..., ge=0, le=1)           # 0/1
    mnth: int = Field(..., ge=1, le=12)        # 1-12
    hr: int = Field(..., ge=0, le=23)          # 0-23
    holiday: int = Field(..., ge=0, le=1)      # 0/1
    weekday: int = Field(..., ge=0, le=6)      # 0=Dom ... 6=Sáb
    workingday: int = Field(..., ge=0, le=1)   # 0/1
    weathersit: int = Field(..., ge=1, le=4)   # 1-4

    # continuas
    temp: float
    atemp: float
    hum: float
    windspeed: float

    # derivados
    year: int                                   
    month: int = Field(..., ge=1, le=12)
    dayofweek: int = Field(..., ge=0, le=6)

class PredictionRequest(BaseModel):
    instances: List[RideFeatures]

class PredictionResponse(BaseModel):
    predictions: List[float]
    model_version: Optional[str] = None