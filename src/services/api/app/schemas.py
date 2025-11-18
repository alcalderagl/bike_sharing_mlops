from pydantic import BaseModel, Field
from typing import List, Optional

class RideFeatures(BaseModel):
    # Features de Tiempo/Contexto para el Feature Builder
    dteday: str = Field(..., example="2012-11-20", description="Fecha en formato YYYY-MM-DD")
    hr: int = Field(..., example=8, ge=0, le=23, description="Hora del día (0-23)")
    
    # Features de Clima/Input
    temp: float = Field(..., example=0.45, ge=0.0, le=1.0, description="Temperatura normalizada")
    hum: float = Field(..., example=0.6, ge=0.0, le=1.0, description="Humedad normalizada")
    windspeed: float = Field(..., example=0.15, ge=0.0, description="Velocidad del viento normalizada")
    weathersit: int = Field(..., example=2, ge=1, le=4, description="Situación climática (1-4)")
    
    # Features Lag: CRÍTICO - El cliente debe proveer estos valores
    cnt_lag_1: float = Field(..., example=100.0, description="Conteo de bicicletas de la hora anterior (escala original)")
    cnt_lag_24: float = Field(..., example=250.0, description="Conteo de bicicletas de hace 24 horas (escala original)")
    
    # Features de Días/Contexto (pueden ser calculadas, pero es más fácil pasarlas)
    season: int = Field(..., example=3, ge=1, le=4)
    yr: int = Field(..., example=1, ge=0, le=1) # Asumimos yr=0 para 2011, yr=1 para 2012
    mnth: int = Field(..., example=11, ge=1, le=12)
    weekday: int = Field(..., example=2, ge=0, le=6)

class Config:
        # Ejemplo para Swagger
    json_schema_extra = {
        "example": {
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
    }

class PredictionRequest(BaseModel):
    instances: List[RideFeatures] = Field(
        ..., 
        description="List of input feature records to predict."
    )
    inverse_transform: bool = Field(
        True,
        description="Apply inverse transformation (expm1) to predictions if the model was trained in log-scale."
    )

class PredictionResponse(BaseModel):
    predictions: List[float] = Field(
        ..., 
        description="List of predicted bike rental counts."
    )
    model_version: Optional[str] = Field(
        None,
        description="Version of the model used for prediction (if provided by the service)."
    )