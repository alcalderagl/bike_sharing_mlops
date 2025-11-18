from fastapi import APIRouter, Depends, HTTPException
from services.api.app.schemas import PredictionRequest, PredictionResponse
from services.api.app.deps import load_model
from services.api.app.inference import predict_df
from services.api.app.version import get_version
from typing import List
import pandas as pd
from src.services.api.app.inference import load_model # Importar load_model para usarlo en Depends
router = APIRouter()

@router.post('/predict', response_model=PredictionResponse, summary="On demand prediction", description="")
def predict(payload: PredictionRequest, _=Depends(load_model)):
    """
    Endpoint para realizar predicciones. Toma una lista de instancias
    RAW y devuelve la predicción de demanda.
    """
    
    if not payload.instances:
        raise HTTPException(
            status_code=400,
            detail="Debes enviar al menos un registro en 'instances'.",
        )
        
    # convertir a dataframe
    df = pd.DataFrame([inst.model_dump() for inst in payload.instances])
    
    # Validación de columnas esperadas
    expected_cols = ["dteday","hr","temp","hum","windspeed","weathersit","cnt_lag_1","cnt_lag_24","season","yr","mnth","weekday"]
    
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Faltan columnas en la entrada: {missing}",
        )
    
    try:
        # Llamar a la función de inferencia, que ahora incluye Feature Engineering
        preds = predict_df(df, inverse_transform=payload.inverse_transform)
    except ValueError as e:
        # Errores de scikit-learn
        raise HTTPException(
            status_code=400,
            detail=f"Error al generar predicciones: {str(e)}",
        )
    except Exception as e:
        # Deja que el handler global lo convierta en 500
        raise e
    
    return {
        "predictions": [float(pred) for pred in preds],
        "model_version": get_version()
    }