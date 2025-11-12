from fastapi import APIRouter, Depends
from ..schemas.schemas import PredictionRequest, PredictionResponse
from ..deps import load_model
from ..inference import predict_one
from ..version import get_version
from typing import List
router = APIRouter()

@router.post('/predict', response_model=PredictionResponse, summary="On demand prediction")
def predict(req: PredictionRequest, _=Depends(load_model)):
    preds: List[float] = [predict_one(inst) for inst in req.instances]
    return PredictionResponse(predictions=preds, model_version=get_version())