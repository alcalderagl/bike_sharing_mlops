# build_features.py

import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, List
from pathlib import Path
import click
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

# Definición de la lista final de features (alineada con el modelo ganador)
# Nota: 'temp_diff' fue excluida ya que 'atemp' se eliminó en make_dataset.py
FINAL_FEATURES: List[str] = [
    'yr', 'holiday', 'workingday', 'temp', 'hum', 'windspeed', 
    'temp_x_hum', 'wind_sq', 'is_weekend', 
    'hr_sin', 'hr_cos', 'mnth_sin', 'mnth_cos', 'doy_sin', 'doy_cos',
    'cnt_lag1_log', 'cnt_lag24_log', 'cnt_rolling_mean_log',
    'season_2', 'season_3', 'season_4', 
    'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6',
    'weathersit_2', 'weathersit_3', 'weathersit_4'
]
TARGET_COLUMN = 'cnt_log'


def build_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica toda la ingeniería de características (cíclica, interacciones, lags, OHE) 
    sobre el dataset limpio.
    """
    logger.info("Iniciando la construcción de features...")
    df = df_clean.copy()
    df = df.sort_values(by=['dteday', 'hr']).reset_index(drop=True)
    # 1. Asegurar que 'dteday' sea datetime y ordenar
    df['dteday'] = pd.to_datetime(df['dteday'])
    
    # 2. Ingeniería de Tiempos
    df['dayofyear'] = df['dteday'].dt.dayofyear
    # weekday=0 (domingo), weekday=6 (sábado)
    df['is_weekend'] = df['weekday'].isin([0, 6]).astype(float) 

    # 3. Codificación Cíclica
    df['hr_sin'] = np.sin(2 * np.pi * df['hr'] / 24.0)
    df['hr_cos'] = np.cos(2 * np.pi * df['hr'] / 24.0)
    df['mnth_sin'] = np.sin(2 * np.pi * df['mnth'] / 12.0)
    df['mnth_cos'] = np.cos(2 * np.pi * df['mnth'] / 12.0)
    df['doy_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 366.0)
    df['doy_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 366.0)

    # 4. Interacciones Climáticas
    df['temp_x_hum'] = df['temp'] * df['hum']
    df['wind_sq'] = df['windspeed'] ** 2
    
    # 5. Creación de Lags (Usando 'cnt', la demanda winsorizada)
    # df['cnt'] es la demanda winsorizada y limpia de make_dataset.py
    df['cnt_lag1'] = df['cnt'].shift(1)
    df['cnt_lag24'] = df['cnt'].shift(24)
    # min_periods=1 asegura que rolling mean comience inmediatamente
    df['cnt_rolling_mean'] = df['cnt'].rolling(window=3, min_periods=1).mean()
    
    # Transformación Logarítmica de los Lags
    df['cnt_log'] = np.log1p(df['cnt'])
    df['cnt_lag1_log'] = np.log1p(df['cnt_lag1'])
    df['cnt_lag24_log'] = np.log1p(df['cnt_lag24'])
    df['cnt_rolling_mean_log'] = np.log1p(df['cnt_rolling_mean'])



    # 6. One-Hot Encoding (OHE)
    cols_ohe = ['season', 'weekday', 'weathersit']
    df = pd.get_dummies(df, columns=cols_ohe, drop_first=True, dtype=float)
    
    # 7. Selección y Alineación Final
    cols_to_keep = FINAL_FEATURES + [TARGET_COLUMN, 'dteday','hr'] 
    
    # Asegurar que todas las columnas OHE existan (para un set de prueba incompleto)
    for col in FINAL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
            
    # También conservamos la columna 'cnt' limpia y winsorizada (para lags en producción)
    if 'cnt' in df.columns:
        cols_to_keep.append('cnt')

    # Usar set para eliminar duplicados y convertir a lista para indexar
    return df[list(set(cols_to_keep))] 


def final_split_and_clean(df_fe: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Aplica la limpieza final de nulos en lags (las primeras filas) y realiza la división temporal.
    """
    
    # 1. Eliminar filas con nulos en los lags (filas iniciales)
    lag_cols = [c for c in df_fe.columns if 'lag' in c and '_log' in c]
    df_clean = df_fe.dropna(subset=lag_cols)


    INDEX_COLS = ['dteday','hr']
    X_final = df_clean[FINAL_FEATURES].copy()
    X_index = df_clean[INDEX_COLS].copy()
    y_final = df_clean[TARGET_COLUMN]


    # 3. División temporal
    split_point = int(len(df_clean) * train_ratio)
    
    # Features para entrenamiento
    X_train = X_final.iloc[:split_point]
    X_val = X_final.iloc[split_point:]

    # Índices (se usan para el reporte, no para el modelo)
    Index_train = X_index.iloc[:split_point]
    Index_val = X_index.iloc[split_point:]

    # Target
    y_train = y_final.iloc[:split_point]
    y_val = y_final.iloc[split_point:]
    
    logger.info(f"Split Temporal realizado: Train={len(X_train)} | Validation={len(X_val)}")
    
    return X_train, X_val, y_train, y_val, Index_train, Index_val

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--train_ratio', default=0.8, type=float, help='Proporción de datos para entrenamiento (split temporal).')
def main_build_features(input_filepath, output_dir, train_ratio):
    """ Builds features from processed data and splits into train/validation sets."""
    
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    # Carga el output de make_dataset.py
    df_clean = pd.read_csv(input_filepath) 
    
    # 1. Construir features
    df_fe = build_features(df_clean)
    
    # 2. Split y limpieza final
    X_train, X_val, y_train, y_val, Index_train, Index_val = final_split_and_clean(df_fe, train_ratio)
    # 3. Guardar sets de entrenamiento y validación
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Guardamos los 4 archivos CSV separados
    X_train.to_csv(output_path / 'X_train.csv', index=False)
    X_val.to_csv(output_path / 'X_val.csv', index=False)
    Index_train.to_csv(output_path / 'Index_train.csv', index=False)
    Index_val.to_csv(output_path / 'Index_val.csv', index=False)
    y_train.to_csv(output_path / 'y_train.csv', index=False, header=[TARGET_COLUMN])
    y_val.to_csv(output_path / 'y_val.csv', index=False, header=[TARGET_COLUMN])
    historical_cols = ['dteday', 'hr', 'cnt'] # Columnas clave para el Feature Builder
    df_clean[historical_cols].to_csv(output_path / 'historical_for_prediction.csv', index=False)
    
    logger.info(f"Datasets de Train y Validation guardados en: {output_path}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())

    # Lógica de ejecución de prueba con rutas de Cookiecutter (ajustar si es necesario)
    project_dir = Path(__file__).resolve().parents[2]
    input_file = os.path.join(project_dir, 'data', 'processed', 'bike_sharing_processed.csv')
    output_directory = os.path.join(project_dir, 'data', 'interim')
    
    main_build_features.callback(input_file, output_directory, 0.8)
