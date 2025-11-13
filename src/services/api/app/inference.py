from services.api.app.schemas import RideFeatures
from services.api.app.config import settings
import joblib
import numpy as np
import pandas as pd

_model = None

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(settings.MODEL_PATH)
        # print("Tipo:", type(_model))
        # print("n_features_in_:", getattr(_model, "n_features_in_", None))
        # print(_model.named_steps)
        try:
            print(_model.feature_names_in_)
        except:
            print("No trae feature_names_in_")
    return _model

def predict_df(df: pd.DataFrame, inverse_transform: bool = True) -> np.ndarray:
    model = load_model()
    
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
       
        for c in cols:
            if c not in df.columns:
                df[c] = 0
        df = df[cols]

    y = model.predict(df)


    if inverse_transform:
        try:
            y = np.expm1(y)
        except Exception:
            pass

    return y
