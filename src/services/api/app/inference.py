from src.services.api.app.schemas import RideFeatures
from src.services.api.app.config import settings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.models.predict_model import predict_features_builder
from src.models.predict_model import FINAL_FEATURES

_model = None
_historical_df = None


def load_model():
    """Carga el modelo y los datos históricos (para FE) al iniciar."""
    global _model
    global _historical_df
    
    if _model is None:
        model_path = Path(settings.MODEL_PATH)
        if not model_path.exists():
             raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
             
        _model = joblib.load(model_path)
        print(f"✅ Modelo cargado desde: {settings.MODEL_PATH}")
        
    # Cargar datos históricos solo si no se han cargado (para el Feature Builder)
    if _historical_df is None:
        try:
            # En tu pipeline de 'build_features' o 'predict_model', debiste guardar un
            # archivo con el historial necesario para calcular el rolling mean y los lags.
            # Ajusta esta ruta si es diferente.
            historical_file = Path("data/interim/historical_for_prediction.csv") 
            _historical_df = pd.read_csv(historical_file)
            _historical_df['dteday'] = pd.to_datetime(_historical_df['dteday'])
            print(f"✅ Datos históricos para FE cargados desde: {historical_file}")
        except FileNotFoundError:
            print("⚠️ Advertencia: No se encontraron datos históricos para Feature Engineering. El modelo fallará si requiere Lags complejos o Rolling Means.")
            _historical_df = pd.DataFrame() # Crear DF vacío para evitar fallos.

    return _model

def predict_df(df_raw: pd.DataFrame, inverse_transform: bool = True) -> np.ndarray:
    """
    Aplica FE, alinea features y realiza la predicción.
    """
    model = load_model()
    global _historical_df

    # 1. Preparar datos RAW para el Feature Builder
    df_raw['dteday'] = pd.to_datetime(df_raw['dteday'])
    # Se añade una columna 'cnt' temporal con cnt_lag_1 para que el FE funcione si lo necesita
    df_raw['cnt'] = df_raw['cnt_lag_1'] 
    
    # 2. Aplicar Feature Engineering
    # predict_features_builder necesita el df_raw y el historial
    df_fe = predict_features_builder(df_raw, _historical_df)

    # 3. Asegurar alineación y solo usar las FINAL_FEATURES
    try:
        df_final = df_fe[FINAL_FEATURES]
    except KeyError as e:
        raise ValueError(f"Feature Mismatch: Falta una columna final ({e}) después del FE. Revise FINAL_FEATURES.")
        
    # 4. Generar Predicciones (Logarítmicas)
    y_log = model.predict(df_final)

    # 5. Transformación Inversa (Escala Original)
    if inverse_transform:
        y = np.expm1(y_log)
        y[y < 0] = 0.0 # Asegurar no negativos
    else:
        y = y_log

    return y.round(0) # Retorna el conteo como enteros
