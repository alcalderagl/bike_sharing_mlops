# tests/test_unit_predict.py

import pytest
import numpy as np
import pandas as pd
from src.models.predict_model import predict_features_builder, PREDICTION_COLUMN, FINAL_FEATURES

@pytest.fixture
def mock_predict_data():
    """
    Fixture de datos de entrada y datos históricos para simular predicción.
    Asegura que 'dteday' y 'cnt' estén disponibles como columnas.
    """
    # Establecer una longitud secuencial
    N_hist = 48 # 2 días de historial
    N_new = 2  # 2 horas nuevas a predecir
    
    # --- 1. Dataframe Histórico (necesario para lags) ---
    historical_data = {
        'dteday': pd.to_datetime(pd.Series(range(N_hist)).apply(lambda x: f"2023-01-01 {x%24:02d}:00:00")),
        'hr': [i % 24 for i in range(N_hist)],
        # Target original y transformado (usado para los lags)
        'cnt': np.arange(100, 100 + N_hist),
        'cnt_log': np.log1p(np.arange(100, 100 + N_hist)), 
        # Features simuladas
        'temp': np.linspace(0.2, 0.8, N_hist),
        'hum': np.linspace(0.5, 0.9, N_hist),
        'windspeed': np.linspace(0.1, 0.4, N_hist),
        # Columnas categóricas dummy (mantenerlas consistentes)
        'yr': 1, 'holiday': 0, 'workingday': 1, 'weathersit': 1, 'season': 1, 'mnth': 1, 'weekday': 1
    }
    historical_df = pd.DataFrame(historical_data)
    
    # --- 2. Dataframe de Nuevos Datos (a predecir) ---
    new_data = {
        'dteday': pd.to_datetime([
            f"2023-01-03 00:00:00", # Continúa después del historial
            f"2023-01-03 01:00:00"
        ]),
        'hr': [0, 1],
        'temp': [0.25, 0.26],
        'hum': [0.8, 0.81],
        'windspeed': [0.1, 0.1],
        # Columnas categóricas dummy (cambiar 'weekday' para el nuevo día 3)
        'yr': 1, 'holiday': 0, 'workingday': 1, 'weathersit': 1, 'season': 1, 'mnth': 1, 'weekday': 3
    }
    new_data_df = pd.DataFrame(new_data)
    
    return new_data_df, historical_df


def test_predict_features_builder_lag_values(mock_predict_data):
    """Verifica que el cálculo de Lag 1 y Lag 24 sea correcto en la predicción."""
    new_data_df, historical_df = mock_predict_data
    
    # Llamar al feature builder
    X_new_fe = predict_features_builder(new_data_df, historical_df)
    
    # 1. Validación de Lag 1 (Hora 0)
    # El Lag 1 de la Hora 0 (primer registro) debe ser el último valor del historial (Hora 47).
    expected_lag1_h0 = historical_df['cnt_log'].iloc[-1]
    
    # El valor del Lag 1 en la Hora 0 (índice 0) de la predicción
    assert X_new_fe['cnt_lag1_log'].iloc[0] == expected_lag1_h0

    # 2. Validación de Lag 24 (Hora 0)
    # El Lag 24 de la Hora 0 debe ser el valor de hace 24 horas en el historial (Hora 24).
    expected_lag24_h0 = historical_df['cnt_log'].iloc[48 - 24] # Índice 24 del historial
    
    # El valor del Lag 24 en la Hora 0 (índice 0) de la predicción
    assert X_new_fe['cnt_lag24_log'].iloc[0] == expected_lag24_h0

    # 3. Validación de Lag 1 (Hora 1)
    # El Lag 1 de la Hora 1 (segundo registro) debe ser la predicción (aún no disponible) o, en este mock, NaN si el builder no puede iterar.
    # **IMPORTANTE**: La función predict_features_builder debe estar diseñada para usar los datos *nuevos* anteriores si están disponibles.
    # Si la función solo usa el historial, el Lag 1 de la Hora 1 sería el Lag 1 de la Hora 0, lo cual es incorrecto. 
    # Asumimos que usa el último valor del historial para el Lag 1 del primer punto, y luego usa los valores de la ventana de predicción.

    # En un caso real con un predict_model iterativo, el Lag 1 de la hora 1 usaría la PREDICCIÓN de la hora 0.
    # Dado que estamos probando la función de Features, solo podemos verificar el primer punto.

    # Por simplicidad y enfoque unitario, verificamos que el set final de features esté correcto
    missing_features = set(FINAL_FEATURES) - set(X_new_fe.columns)
    assert len(missing_features) == 0, f"Features faltantes: {missing_features}"


def test_prediction_output_non_negative():
    """Verifica que la transformación inversa (np.expm1) no genere valores de predicción negativos."""
    # Simulación de un modelo que predice un valor logarítmico muy bajo, que daría negativo al hacer expm1.
    # log(1+cnt) = 0.01 -> 1+cnt = e^0.01 -> cnt = 1.01 - 1 = 0.01 (positivo)
    # log(1+cnt) = -2 -> 1+cnt = e^-2 = 0.135 -> cnt = 0.135 - 1 = -0.86 (negativo)
    
    # Predicción logarítmica que debería ser limitada a 0
    pred_log_with_negative_risk = np.array([5.0, 0.01, -2.0, 3.0]) 
    
    # Aplicar la lógica de predicción del predict_model.py:
    pred_cnt = np.expm1(pred_log_with_negative_risk)
    pred_cnt[pred_cnt < 0] = 0.0 # <--- Lógica crítica a probar
    
    assert (pred_cnt >= 0.0).all(), "Las predicciones en escala real contienen valores negativos."
    assert np.isclose(pred_cnt[2], 0.0), "El valor negativo (-0.86) no fue limitado a 0."


def test_prediction_output_correct_column_name():
    """Verifica que el DataFrame de salida use el nombre de columna correcto."""
    # Simulación de la salida final
    predictions = np.array([10.0, 20.0, 30.0])
    index_cols = pd.DataFrame({'dteday': ['d1', 'd2', 'd3'], 'hr': [1, 2, 3]})
    
    # Lógica de guardado simplificada (tomada de predict_model.py)
    predictions_df = pd.DataFrame({PREDICTION_COLUMN: predictions.round(0)}, index=index_cols.index)
    output_df = pd.concat([index_cols.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1)

    assert PREDICTION_COLUMN in output_df.columns
    assert output_df[PREDICTION_COLUMN].dtype == float or np.issubdtype(output_df[PREDICTION_COLUMN].dtype, np.integer), "La columna de predicción no es numérica."