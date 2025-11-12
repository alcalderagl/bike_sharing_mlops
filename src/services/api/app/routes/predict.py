from fastapi import APIRouter, Depends
from services.api.app.schemas import PredictionRequest, PredictionResponse
from services.api.app.deps import load_model
from services.api.app.inference import predict_one
from services.api.app.version import get_version
from typing import List
router = APIRouter()

@router.post('/predict', response_model=PredictionResponse, summary="On demand prediction")
def predict(req: PredictionRequest, _=Depends(load_model)):
    preds: List[float] = [predict_one(inst) for inst in req.instances]
    return PredictionResponse(predictions=preds, model_version=get_version())