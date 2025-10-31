import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import QuantileTransformer
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

TARGET_COLUMN = 'cnt'

# =========================================================================
# REGLA DE NEGOCIO: DÍAS FESTIVOS FIJOS (Añadido para corregir inconsistencias)
# =========================================================================
HOLIDAY_DATES = {
    pd.to_datetime('2011-01-17').date(), pd.to_datetime('2011-02-21').date(), pd.to_datetime('2011-04-15').date(), pd.to_datetime('2011-05-30').date(),
    pd.to_datetime('2011-07-04').date(), pd.to_datetime('2011-09-05').date(), pd.to_datetime('2011-10-10').date(), pd.to_datetime('2011-11-11').date(),
    pd.to_datetime('2011-11-24').date(), pd.to_datetime('2011-12-26').date(), pd.to_datetime('2012-01-02').date(), pd.to_datetime('2012-01-16').date(),
    pd.to_datetime('2012-02-20').date(), pd.to_datetime('2012-04-16').date(), pd.to_datetime('2012-05-28').date(), pd.to_datetime('2012-07-04').date(),
    pd.to_datetime('2012-09-03').date(), pd.to_datetime('2012-10-08').date(), pd.to_datetime('2012-11-12').date(), pd.to_datetime('2012-11-22').date(),
    pd.to_datetime('2012-12-25').date()
}


def load_data(filepath: str) -> pd.DataFrame:
    """Carga el dataset desde la ruta especificada."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        logging.getLogger(__name__).error(f"Archivo no encontrado en: {filepath}")
        raise
        
def winsorize_data(df: pd.DataFrame, features: List[str], lower=0.01, upper=0.99) -> pd.DataFrame:
    """Aplica winsorización a las características continuas."""
    df_winsorized = df.copy()
    for col in features:
        if col in df_winsorized.columns:
            # Calcular percentiles usando solo datos no nulos
            lower_bound = df_winsorized[col].quantile(lower)
            upper_bound = df_winsorized[col].quantile(upper)
            
            # Aplicar clipping (winsorización)
            df_winsorized[col] = np.where(df_winsorized[col] < lower_bound, lower_bound, df_winsorized[col])
            df_winsorized[col] = np.where(df_winsorized[col] > upper_bound, upper_bound, df_winsorized[col])
    return df_winsorized


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
    
    # Manejar 'dteday' (Fecha)
    df_cleaned['dteday'] = pd.to_datetime(df_cleaned['dteday'], errors='coerce')
    
    # Columnas que deben ser numéricas (incluyendo 'cnt', 'holiday', etc.)
    numerical_cols = df_cleaned.columns.drop(['dteday'], errors='ignore').tolist()
    
    # Coerción a numérico. Los valores inválidos (ej. 'error' o 'string') se convierten a NaN.
    for col in numerical_cols:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce') 

    # 3. **REGLA DE NEGOCIO (CORRECCIÓN DE HOLIDAY)**
    # Recalcula la columna 'holiday' basándose en la lista de fechas fijas.
    # Esto corrige cualquier inconsistencia o dato perdido en la columna original.
    df_cleaned['holiday'] = df_cleaned['dteday'].apply(
        lambda x: 1 if pd.notna(x) and x.date() in HOLIDAY_DATES else 0
    )
        
    # 4. Manejo de Valores Faltantes (NaN)
    # Se eliminan filas que tienen NaN en CUALQUIER columna, incluyendo el target 'cnt'
    # o cualquier columna numérica que falló la coerción.
    initial_rows = len(df_cleaned)
    df_cleaned = df_cleaned.dropna()
    
    if len(df_cleaned) < initial_rows:
        logging.getLogger(__name__).info(f"Se eliminaron {initial_rows - len(df_cleaned)} filas con valores inválidos o NaN.")
        
    # 5. Winsorización (Outlier Handling)
    continuous_features = ['temp', 'atemp', 'hum', 'windspeed']
    df_cleaned = winsorize_data(df_cleaned, continuous_features)

    # 6. Transformación de la Variable Objetivo
    y_target = np.log1p(df_cleaned[TARGET_COLUMN])
    X = df_cleaned.drop(columns=[TARGET_COLUMN])
    
    # 7. Ingeniería de Características (Esencial para la predicción)
    # Estas se convertirán a categóricas en build_features.py
    X['year'] = df_cleaned['dteday'].dt.year
    X['month'] = df_cleaned['dteday'].dt.month
    X['dayofweek'] = df_cleaned['dteday'].dt.dayofweek
    
    X = X.drop(columns=['dteday'])
    
    # Asegurar tipos Int64 para categóricas (como lo definiste en el notebook)
    categorical_cols = ['season', 'mnth', 'hr', 'weekday', 'weathersit', 
                        'yr', 'holiday', 'workingday', 'year', 'month', 'dayofweek']
    for col in [c for c in categorical_cols if c in X.columns]:
        X[col] = X[col].astype('Int64', errors='ignore')

    return X, y_target


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ 
    Carga datos crudos, realiza limpieza (incluyendo winsorización) y transforma 
    la variable objetivo (log1p).
    """
    logger = logging.getLogger(__name__)
    logger.info('Iniciando la limpieza, Winsorización y transformación del dataset.')
    
    # 1. Carga de datos
    df_raw = load_data(input_filepath)
    
    # 2. Limpieza y Transformación
    try:
        X_processed, y_processed = clean_data(df_raw)
    except ValueError as e:
        logger.error(f"Fallo en make_dataset: {e}")
        return

    # 3. Recombinar X e y para guardar el dataset completo procesado
    df_processed = pd.concat([X_processed, y_processed.rename(TARGET_COLUMN)], axis=1)

    # 4. Guardar el archivo procesado
    output_path = Path(output_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    
    logger.info(f"Dataset procesado guardado en: {output_filepath}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    # Busca .env para cargar variables de entorno (como configuración MLflow)
    load_dotenv(find_dotenv())
    
    # Ejecución de prueba usando rutas relativas de Cookiecutter
    project_dir = Path(__file__).resolve().parents[2]
    input_file = project_dir / 'data' / 'raw' / 'bike_sharing_modified.csv'
    output_file = project_dir / 'data' / 'processed' / 'bike_sharing_processed.csv'
    
    try:
        main([str(input_file), str(output_file)])
    except Exception as e:
        logging.getLogger(__name__).error(f"Fallo al ejecutar make_dataset.py: {e}")