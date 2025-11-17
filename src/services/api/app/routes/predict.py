from fastapi import APIRouter, Depends
from src.services.api.app.schemas import PredictionRequest, PredictionResponse
from src.services.api.app.inference import predict_df
from src.services.api.app.version import get_version
from typing import List
import pandas as pd
from src.services.api.app.inference import load_model # Importar load_model para usarlo en Depends
router = APIRouter()

@router.post('/predict', response_model=PredictionResponse, summary="Predicción de demanda de bicicletas")
def predict(req: PredictionRequest, _=Depends(load_model)):
    """
    Endpoint para realizar predicciones. Toma una lista de instancias
    RAW y devuelve la predicción de demanda.
    """
    # Convertir la lista de modelos Pydantic a DataFrame
    df = pd.DataFrame([inst.model_dump() for inst in req.instances])

    # Llamar a la función de inferencia, que ahora incluye Feature Engineering
    y = predict_df(df, inverse_transform=req.inverse_transform)
    
    return {
        "predictions": [float(v) for v in y],
        "model_version": get_version() # Añadir la versión del modelo
    }