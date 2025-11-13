from fastapi import APIRouter, Depends
from services.api.app.schemas import PredictionRequest, PredictionResponse
from services.api.app.deps import load_model
from services.api.app.inference import predict_df
from services.api.app.version import get_version
from typing import List
import pandas as pd
router = APIRouter()

@router.post('/predict', response_model=PredictionResponse, summary="On demand prediction")
def predict(req: PredictionRequest, _=Depends(load_model)):
    df = pd.DataFrame([inst.model_dump() for inst in req.instances])

    y = predict_df(df, inverse_transform=req.inverse_transform)
    
    return {
        "predictions": [float(v) for v in y],
        "rows": len(y)
    }