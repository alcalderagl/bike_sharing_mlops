import pytest
import numpy as np
import pandas as pd
from src.models.train_model import evaluate_metrics
from pathlib import Path
from src.features.build_features import build_features

def test_pipeline_integration_predict_format(trained_pipeline, X_test_data):
    """
    Prueba de integración E2E: Verifica la predicción del pipeline cargado 
    y el formato de la salida.
    """

    if 'dteday' not in X_test_data.columns:
        start_date = pd.to_datetime('2012-10-01')
        dates = pd.date_range(start=start_date, periods=len(X_test_data), freq='h')
        X_test_data['dteday'] = dates.date # Asigna solo la parte de la fecha

    if 'cnt' not in X_test_data.columns:
        # Usamos un valor fijo razonable (ej. 50.0) para que el cálculo de lag proceda
        X_test_data['cnt'] = np.full(len(X_test_data), 50.0)
    
    df_clean = X_test_data.drop(columns=['atemp', 'cnt_lag_1', 'cnt_lag_2', 'cnt_lag_24', 'cnt_rolling_3h', 'cnt_rolling_24h'], errors='ignore')
    if 'dteday' in df_clean.columns:
        df_clean['dteday'] = pd.to_datetime(df_clean['dteday'])


    X_fe = build_features(df_clean)
    FINAL_MODEL_FEATURES = ['yr', 'holiday', 'workingday', 'temp', 
                            'hum', 'windspeed', 'temp_x_hum', 'wind_sq', 'is_weekend', 'hr_sin', 'hr_cos',
                              'mnth_sin', 'mnth_cos', 'doy_sin', 'doy_cos', 'cnt_lag1_log', 'cnt_lag24_log', 
                              'cnt_rolling_mean_log', 'season_2', 'season_3', 'season_4', 'weekday_1', 'weekday_2', 
                              'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6', 'weathersit_2', 'weathersit_3', 'weathersit_4']
    
    X_predict = X_fe[FINAL_MODEL_FEATURES]
    
    # Hacer la predicción (debe retornar el logaritmo)
    y_pred_log = trained_pipeline.predict(X_predict)
    
    # 1. Validar la forma y el tipo
    assert isinstance(y_pred_log, np.ndarray)
    assert len(y_pred_log) == len(X_predict)
    
    # 2. Validar que la predicción está en la escala logarítmica (rango pequeño)
    assert y_pred_log.min() > 0 
    assert y_pred_log.max() < 10 
    
    # 3. Validar la transformación inversa a la escala original
    y_pred_real_scale = np.expm1(y_pred_log)
    assert y_pred_real_scale.min() >= 0
    assert y_pred_real_scale.max() > 100 
    
    # 4. Validar la función de métricas
    # Requiere que y_test.csv exista localmente
    try:
        project_root = Path(__file__).parent.parent
        y_test_path = project_root / 'data' / 'processed' / 'y_test.csv'
        y_true_log = pd.read_csv(y_test_path, header=None, names=['cnt'])['cnt']
    except FileNotFoundError:
        pytest.skip("No se encontró y_test.csv para calcular métricas.")
    
    metrics = evaluate_metrics(y_true_log, y_pred_log)
    assert 'rmse_log' in metrics
    assert metrics['r2_score'] > 0.0 # Debe tener algún desempeño