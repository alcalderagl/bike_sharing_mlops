import click
import logging
from pathlib import Path
import joblib # Usado para cargar el modelo local .pkl
import pandas as pd
import numpy as np
import os
from dotenv import find_dotenv, load_dotenv
from typing import List

logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN ---
PREDICTION_COLUMN = 'cnt_prediction'
HISTORICAL_LAG_PERIOD = 24 # Mínimo 24 registros para el lag_24h

# Definición de Features Finales 
FINAL_FEATURES: List[str] = ['yr', 'holiday', 'workingday', 'temp', 'hum', 'windspeed', 'temp_x_hum', 'wind_sq', 'is_weekend', 'hr_sin', 'hr_cos', 'mnth_sin', 'mnth_cos', 'doy_sin', 'doy_cos', 'cnt_lag1_log', 'cnt_lag24_log', 'cnt_rolling_mean_log', 'season_2', 'season_3', 'season_4', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6', 'weathersit_2', 'weathersit_3', 'weathersit_4'] 

def predict_features_builder(new_data_df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combina datos nuevos con historial para calcular Lags y aplica el Feature Engineering.
    """
    
    # 1. Concatenar Historial y Nuevos Datos
    # Esto asegura que el cálculo de shift() en los nuevos datos sea correcto.
    df_combined = pd.concat([historical_df.tail(HISTORICAL_LAG_PERIOD), new_data_df], ignore_index=True)
    
    df = df_combined.copy()
    
    # 2. Replicar Feature Engineering
    df['dteday'] = pd.to_datetime(df['dteday'])
    df['dayofyear'] = df['dteday'].dt.dayofyear
    df['is_weekend'] = df['weekday'].isin([0, 6]).astype(float) 

    # Codificación Cíclica
    df['hr_sin'] = np.sin(2 * np.pi * df['hr'] / 24.0)
    df['hr_cos'] = np.cos(2 * np.pi * df['hr'] / 24.0)
    df['mnth_sin'] = np.sin(2 * np.pi * df['mnth'] / 12.0)
    df['mnth_cos'] = np.cos(2 * np.pi * df['mnth'] / 12.0)
    df['doy_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 366.0)
    df['doy_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 366.0)

    # Interacciones Climáticas
    df['temp_x_hum'] = df['temp'] * df['hum']
    df['wind_sq'] = df['windspeed'] ** 2
    
    # Creación de Lags (CRÍTICO: usa la columna 'cnt' limpia y winsorizada)
    df['cnt_lag1'] = df['cnt'].shift(1)
    df['cnt_lag24'] = df['cnt'].shift(24)
    df['cnt_rolling_mean'] = df['cnt'].rolling(window=3, min_periods=1).mean() 
    
    # Transformación Logarítmica de los Lags
    df['cnt_lag1_log'] = np.log1p(df['cnt_lag1'])
    df['cnt_lag24_log'] = np.log1p(df['cnt_lag24'])
    df['cnt_rolling_mean_log'] = np.log1p(df['cnt_rolling_mean'])

    # One-Hot Encoding (OHE)
    cols_ohe = ['season', 'weekday', 'weathersit']
    df = pd.get_dummies(df, columns=cols_ohe, drop_first=True, dtype=float)
    
    # 3. Filtrar, Alinear y Limpiar
    
    # Retener solo los nuevos datos (descartar el historial que solo se usó para lags)
    X_new_fe = df.tail(len(new_data_df)).copy()
    
    # Asegurar que todas las columnas OHE existan y alinear
    for col in FINAL_FEATURES:
        if col not in X_new_fe.columns:
            X_new_fe[col] = 0.0 # Introduce la columna faltante con valor 0
            
    # Devolver solo las features finales, en el orden correcto, y manejar nulos de lags con 0.0
    return X_new_fe[FINAL_FEATURES].fillna(0.0) 


@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('index_filepath', type=click.Path(exists=True)) # <-- ¡NUEVO!
@click.argument('output_filepath', type=click.Path())
@click.option('--inverse_transform', is_flag=True, default=True, help='Aplica la transformación inversa (np.expm1) a las predicciones.')
def main(model_filepath, input_filepath, index_filepath, output_filepath, inverse_transform):
    """
    Carga el Pipeline entrenado y genera predicciones sobre un nuevo conjunto de datos.
    """
    logger = logging.getLogger(__name__)
    logger.info("Iniciando la generación de predicciones.")
    
    # 1. Cargar el modelo desde el archivo .pkl
    try:
        loaded_model = joblib.load(model_filepath)
        logger.info(f"Modelo cargado exitosamente desde: {model_filepath}")
    except Exception as e:
        logger.error(f"Error cargando modelo desde PKL: {e}")
        return

    # 2. Cargar los Datos
    try:
        # Los datos de entrada deben ser el output de make_dataset.py (limpios, con 'cnt', etc.)
        new_data_df = pd.read_csv(input_filepath)
        historical_df = pd.read_csv(index_filepath)
    except Exception as e:
        logger.error(f"Error al cargar datos de entrada o historial: {e}")
        return

    # 3. Cargar los Datos (Features + Índice)
    try:
        # Cargamos SOLAMENTE las features (input_filepath = X_val.csv)
        X_data = pd.read_csv(input_filepath) 
        # Cargamos las columnas de índice (index_filepath = Index_val.csv)
        Index_data = pd.read_csv(index_filepath)
    except Exception as e:
        logger.error(f"Error al cargar datos de features o índice: {e}")
        return

    logger.info(f"Features (X) cargadas: {len(X_data)} registros. Índices cargados: {len(Index_data)}.")

    # 4. Asegurar que las columnas coincidan exactamente
    try:
        # Esto es crucial: SOLO le pasamos las features que el modelo necesita
        X_data = X_data[FINAL_FEATURES]
    except KeyError as e:
        logger.error(f"Falta una feature crucial en el dataset de predicción: {e}. Revise FINAL_FEATURES.")
        return
        
    # 5. Generar Predicciones (Escala Logarítmica)
    pred_log = loaded_model.predict(X_data)
    # 3. Aplicar Feature Engineering
    # X_new_fe = predict_features_builder(new_data_df, historical_df)
    # logger.info(f"Features construidas para {len(X_new_fe)} registros.")

    # 4. Asegurar que las columnas coincidan exactamente
    # try:
    #     X_new_fe = X_new_fe[FINAL_FEATURES]
    # except KeyError as e:
    #     # Esto ocurre si falta alguna columna en el feature builder
    #     logger.error(f"Falta una feature crucial en el dataset de predicción: {e}")
    #     return
    # # 5. Generar Predicciones (Escala Logarítmica)
    # pred_log = loaded_model.predict(X_new_fe)
    
    # 6. Aplicar Transformación Inversa (Escala Original)
    pred_cnt = np.expm1(pred_log)
    pred_cnt[pred_cnt < 0] = 0.0 # Asegurar no negativos
    
    # 7. Guardar Resultados
    
    # Crear un DataFrame con las predicciones
    predictions_df = pd.DataFrame({PREDICTION_COLUMN: pred_cnt.round(0)}, index=Index_data.index)
    output_df = pd.concat([Index_data, predictions_df], axis=1)
    
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_filepath, index=False)
    
    logger.info(f"Predicciones guardadas en escala original en: {output_filepath}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    main()