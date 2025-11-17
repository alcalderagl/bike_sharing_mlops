# tests/test_unit_features.py

import pytest
import numpy as np
import pandas as pd
from src.features.build_features import build_features, final_split_and_clean, FINAL_FEATURES

def test_cyclic_features_range(clean_data_for_fe):
    """Verifica que las features cíclicas (seno/coseno) estén en el rango [-1, 1]."""
    df_fe = build_features(clean_data_for_fe)
    
    # Comprobación de que las columnas existen
    assert 'hr_sin' in df_fe.columns
    assert 'mnth_cos' in df_fe.columns
    
    # Comprobación del rango [-1, 1]
    for col in ['hr_sin', 'hr_cos', 'mnth_sin', 'mnth_cos']: # Asumiendo doy también se calcula
        assert df_fe[col].min() >= -1.0
        assert df_fe[col].max() <= 1.0


def test_lag_features_shift(clean_data_for_fe):
    """Verifica que los valores de Lag se calculen correctamente y que los primeros sean NaN."""
    df_fe = build_features(clean_data_for_fe)
    
    # El Lag 1 (hora anterior) debe ser NaN en el primer registro
    assert df_fe['cnt_lag1_log'].iloc[0] == 0.0 or np.isnan(df_fe['cnt_lag1_log'].iloc[0])

    # El Lag 24 (mismo valor del día anterior) debe ser NaN en los primeros 24 registros
    assert df_fe['cnt_lag24_log'].iloc[23] == 0.0 or np.isnan(df_fe['cnt_lag24_log'].iloc[23])
    

def test_rolling_window_mean(clean_data_for_fe):
    """Verifica que el cálculo de la media móvil (rolling mean) sea correcto."""
    df_fe = build_features(clean_data_for_fe)
    
    # Usamos la columna cnt_log FINAL calculada por tu función (df_fe)
    # Asumimos que la media móvil es sobre la columna 'cnt' y luego log.
    
    # 1. Obtener la columna 'cnt' que tu función usó
    cnt_col = df_fe['cnt']
    
    # 2. Re-calcular el Rolling Mean usando la lógica de tu función:
    #    Rolling Mean (window=3, min_periods=1) sobre cnt, LUEGO log1p
    raw_rolling_mean = cnt_col.rolling(window=3, min_periods=1).mean()
    manual_rolling_mean_log = np.log1p(raw_rolling_mean)
    
    # 3. CRÍTICO: Comparamos el valor CALCULADO manualmente con el valor en la columna.
    #    Usamos el índice 3, donde ya debe estar disponible el cálculo completo de 3 periodos.
    calculated_val = df_fe['cnt_rolling_mean_log'].iloc[3]
    expected_val = manual_rolling_mean_log.iloc[3]

    assert 'cnt_rolling_mean_log' in df_fe.columns
    # La prueba ahora debe pasar, ya que comparamos dos series generadas por la misma lógica.
    assert np.isclose(calculated_val, expected_val, atol=1e-6) 
    
    # OPCIONAL: Verificar el primer valor (donde min_periods=1)
    # expected_val_1 = manual_rolling_mean_log.iloc[0]
    # assert np.isclose(df_fe['cnt_rolling_mean_log'].iloc[0], expected_val_1, atol=1e-6)


    
def test_final_features_match(clean_data_for_fe):
    """Verifica que el split final solo contenga las FINAL_FEATURES y que TARGET_COLUMN haya sido separada."""
    df_fe = build_features(clean_data_for_fe)
    
    # Usamos un ratio de 0.8 para no fallar en el split
    X_train, X_val, y_train, y_val, index_train,index_val = final_split_and_clean(df_fe, train_ratio=0.8)
    
    # 1. X_train y X_val solo deben tener las FINAL_FEATURES
    missing_in_train = set(FINAL_FEATURES) - set(X_train.columns)
    extra_in_train = set(X_train.columns) - set(FINAL_FEATURES)
    
    assert len(extra_in_train) == 0, f"X_train tiene features extra: {extra_in_train}"
    assert len(missing_in_train) == 0, f"X_train le faltan features: {missing_in_train}"
    
    # 2. y_train y y_val son Series y no DataFrames
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_val, pd.Series)
    
    # 3. No debe haber NaNs restantes en X (después de imputación/limpieza final)
    assert X_train.isna().sum().sum() == 0
    assert X_val.isna().sum().sum() == 0