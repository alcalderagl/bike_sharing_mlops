from pydantic import BaseModel, Field
from typing import List, Optional

class RideFeatures(BaseModel):
    season: int = Field(..., description="Season of the year (1=Spring, 2=Summer, 3=Fall, 4=Winter)", examples=[1])
    yr: int = Field(..., description="Binary year indicator", examples=[1])
    mnth: int = Field(..., description="Month of the year (1-12)", examples=[4])
    hr: int = Field(..., description="Hour of the day (0-23)", examples=[18])
    holiday: int = Field(..., description="Holiday indicator (1=Holiday, 0=Non-holiday)", examples=[0])
    weekday: int = Field(..., description="Day of the week (0=Sunday to 6=Saturday)", examples=[3])
    workingday: int = Field(..., description="Working day indicator (1=Working day, 0=Weekend or holiday)", examples=[1])
    weathersit: int = Field(..., description="Weather situation (1=Clear, 2=Mist/Cloudy, 3=Light snow/rain, 4=Heavy snow/rain)", examples=[0])
    temp: float = Field(..., description="Normalized temperature (0 to 1)", examples=[0.32])
    atemp: float = Field(..., description="Normalized 'feels-like' temperature (0 to 1)", examples=[0.30])
    hum: float = Field(..., description="Normalized humidity (0 to 1)", examples=[0.58])
    windspeed: float = Field(..., description="Normalized wind speed (0 to 1)", examples=[0.15])
    year: int = Field(..., description="Calendar year (e.g., 2011, 2012)", examples=[2024])
    month: int = Field(..., description="Calendar month (1-12)", examples=[1])
    dayofweek: int = Field(..., description="Day of the week (0=Monday to 6=Sunday)", examples=[4])

class PredictionRequest(BaseModel):
    instances: List[RideFeatures] = Field(
        ..., 
        description="List of input feature records to predict."
    )
    inverse_transform: bool = Field(
        True,
        description="Apply inverse transformation (expm1) to predictions if the model was trained in log-scale."
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }

class PredictionResponse(BaseModel):
    predictions: List[float] = Field(
        ..., 
        description="List of predicted bike rental counts."
    )
    model_version: Optional[str] = Field(
        None,
        description="Version of the model used for prediction (if provided by the service)."
    )