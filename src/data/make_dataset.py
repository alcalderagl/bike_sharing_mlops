import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import click
import logging
from dotenv import find_dotenv, load_dotenv

TARGET_COLUMN = 'cnt'

def load_data(raw_data_filepath: str) -> pd.DataFrame:
    """Carga el dataset crudo desde la ruta especificada."""
    try:
        df = pd.read_csv(raw_data_filepath)
        return df
    except Exception as e:
        raise Exception(f"Error al cargar los datos: {e}")

def winsorize_data(df: pd.DataFrame, cols: list, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.DataFrame:
    """
    Aplica Winsorización a las columnas especificadas, limitando valores extremos
    al percentil superior e inferior para mitigar outliers (como en 01_preproccesing.ipynb).
    """
    for col in cols:
        if col in df.columns:
            # Calcular los umbrales específicos de la columna
            lower_bound = df[col].quantile(lower_quantile)
            upper_bound = df[col].quantile(upper_quantile)
            
            # Aplicar el límite a los valores
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            
    return df
# src/data/make_dataset.py (Función clean_data CORREGIDA)

def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Realiza la limpieza de datos, manejando inconsistencias, fugas del objetivo (leakage)
    y preparando los tipos de datos.

    Retorna: DataFrame limpio (X) y Serie objetivo transformada (y).
    """
    if df.empty:
        raise ValueError("El DataFrame de entrada está vacío.")

    df_cleaned = df.copy()

    # 1. Eliminación de Columnas: Fuga del objetivo (Target Leakage) e IDs
    COLUMNS_TO_DROP = ['instant', 'casual', 'registered', 'mixed_type_col']
    df_cleaned = df_cleaned.drop(columns=COLUMNS_TO_DROP, errors='ignore')

    # 2. Conversión de Tipos (Limpieza de Inconsistencias)
    
    # Columnas que deben ser numéricas (AHORA INCLUYE TARGET_COLUMN='cnt')
    numerical_cols = df_cleaned.columns.drop(['dteday'], errors='ignore').tolist()
    
    # Manejar 'dteday' (Fecha)
    df_cleaned['dteday'] = pd.to_datetime(df_cleaned['dteday'], errors='coerce')
    
    # Coerción a numérico. Esto asegura que 'cnt' también se limpie.
    for col in numerical_cols:
        # Los valores inválidos (ej. 'error' o 'string') se convierten a NaN.
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce') 
        
    # 3. Manejo de Valores Faltantes (NaN)
    # Se eliminan filas que tienen NaN en CUALQUIER columna, incluyendo 'cnt'.
    initial_rows = len(df_cleaned)
    df_cleaned = df_cleaned.dropna()
    
    if len(df_cleaned) < initial_rows:
        logging.getLogger(__name__).info(f"Se eliminaron {initial_rows - len(df_cleaned)} filas con valores inválidos o NaN.")
        
    # 4. Winsorización (Outlier Handling)
    continuous_features = ['temp', 'atemp', 'hum', 'windspeed']
    df_cleaned = winsorize_data(df_cleaned, continuous_features)

    # 5. Transformación de la Variable Objetivo (AHORA SEGURO)
    # 'cnt' es ahora tipo numérico (float) y no contiene NaNs.
    y_target = np.log1p(df_cleaned[TARGET_COLUMN])
    X = df_cleaned.drop(columns=[TARGET_COLUMN])
    
    # 6. Ingeniería de Características (Esencial para la predicción)
    X['year'] = df_cleaned['dteday'].dt.year # Usamos df_cleaned aquí porque X es la versión drop columns
    X['month'] = df_cleaned['dteday'].dt.month
    X['dayofweek'] = df_cleaned['dteday'].dt.dayofweek
    
    X = X.drop(columns=['dteday'])
    
    # Asegurar tipos Int64
    categorical_cols = X.columns.drop(continuous_features).tolist()
    for col in categorical_cols:
        X[col] = X[col].astype('Int64', errors='ignore')

    return X, y_target



# --- Estructura CLI de Cookiecutter ---

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Transforma los datos crudos (input_filepath) en datos limpios y preprocesados 
        listos para modelado (output_filepath).
    """
    logger = logging.getLogger(__name__)
    logger.info('Iniciando la limpieza, Winsorización y transformación del dataset.')
    
    try:
        df_raw = load_data(input_filepath)
        X_clean, y_target_log = clean_data(df_raw)
        
        # Juntar X y la y transformada (log(cnt)) para guardar el archivo procesado único
        df_processed = pd.concat([X_clean, y_target_log.rename(TARGET_COLUMN)], axis=1)
        
        # Guardar el archivo limpio (replicando el flujo de 01_preproccesing.ipynb)
        Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
        df_processed.to_csv(output_filepath, index=False)
        
        logger.info(f'Dataset procesado y transformado guardado en: {output_filepath}')
        logger.info(f'Forma final del dataset procesado: {df_processed.shape}')

    except Exception as e:
        logger.error(f"Fallo en make_dataset: {e}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    main()