from typing import Tuple, List,Optional
from datetime import date
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
# Scikit-learn imports
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


import numpy as np
import pandas as pd
import click
import logging


# Define la columna objetivo
TARGET_COLUMN = 'cnt'

# Definición de fechas de feriados fijos para la regla de negocio
# Se tiene un riesgo de holidays con años distintos en el dataset -- Por resolver

HOLIDAYS_SET = {
    date(2011, 1, 17), date(2011, 2, 21), date(2011, 4, 15), date(2011, 5, 30),
    date(2011, 7, 4), date(2011, 9, 5), date(2011, 10, 10), date(2011, 11, 11),
    date(2011, 11, 24), date(2011, 12, 26), date(2012, 1, 2), date(2012, 1, 16),
    date(2012, 2, 20), date(2012, 4, 16), date(2012, 5, 28), date(2012, 7, 4),
    date(2012, 9, 3), date(2012, 10, 8), date(2012, 11, 12), date(2012, 11, 22),
    date(2012, 12, 25)
}
INVALID_PATTERNS = ['nan', 'NaN', 'NAN', 'null', 'NULL', 'none', 'None', 'NONE', 
                        'unknown', 'Unknown', 'UNKNOWN', 'error', 'Error', 'ERROR',
                        'invalid', 'Invalid', 'INVALID', 'missing', 'Missing', 'MISSING',
                        'n/a', '<NA>','<N/A>', 'N/A', 'na', 'NA', '#N/A', '#NULL!', '#DIV/0!', '', ' ','bad','Bad','BAD','?']

COLUMNS_TO_DROP = ['instant', 'casual', 'registered', 'mixed_type_col'] # Después de la preparación de datos.


def load_data(filepath: str) -> pd.DataFrame:
    """Carga el dataset desde la ruta especificada."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        logging.getLogger(__name__).error(f"Archivo no encontrado en: {filepath}")
        raise

def clean_pattern_values(df:pd.DataFrame, invalid_values:List) -> pd.DataFrame:
    """Limpia valores inválidos en una columna específica, convirtiéndolos a NaN."""
    df_cleaned = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        # print(f"Limpieza de patrones inválidos en la columna: {col}")
        df_cleaned[col] = df_cleaned[col].astype(str).str.strip().replace(invalid_values, pd.NA, regex=False)
    return df_cleaned

def dtype_correction(df:pd.DataFrame) -> pd.DataFrame:
    """Corrige los tipos de datos de las columnas en el DataFrame."""
    df_corrected = df.copy()
    if 'instant' in df_corrected.columns:
        int_columns = ['instant','season','yr','mnth','hr','holiday','weekday','workingday', 'weathersit', 'casual', 'registered', 'cnt']
    else:
        int_columns = ['season','yr','mnth','hr','holiday','weekday','workingday', 'weathersit', 'casual', 'registered', 'cnt']
    float_columns = ['temp', 'atemp', 'hum', 'windspeed']
    date_columns = ['dteday']

    df_corrected['dteday'] = df_corrected['dteday'].astype(str).str.strip()

    for col in int_columns:
        df_corrected[col] = pd.to_numeric(df_corrected[col], errors='coerce').astype('Int64')
    for col in float_columns:
        df_corrected[col] = pd.to_numeric(df_corrected[col], errors='coerce').astype('Float64')
    for col in date_columns:
        df_corrected[col] = pd.to_datetime(df_corrected[col], errors='coerce').dt.date

    return df_corrected


def prepare_datefields(df: pd.DataFrame,base_year:Optional[int]=None) -> pd.DataFrame:
    # 1. Estandarización del tipo de dato
    df['dteday'] = pd.to_datetime(df['dteday'], errors='coerce')

    # Manejo de fechas inválidas
    rows_before = len(df)
    df = df.dropna(subset=['dteday']).reset_index(drop=True)
    rows_after = len(df)
    if rows_before != rows_after:
        logging.getLogger(__name__).warning(f"Se eliminaron {rows_before - rows_after} filas debido a fechas inválidas (NaT).")

    
    # 2. Filtrado por año base si se proporciona
    if base_year is None:
        # Se extrae el año de todas las fechas y se busca el mínimo
        min_dataset_year = df['dteday'].dt.year.min()
    else:
        min_dataset_year = base_year
    
    # Recalcular 'yr': Se resta el año mínimo del dataset
    # Si el año mínimo es 2011, la salida es 0 para 2011 y 1 para 2012.
    df['yr'] = df['dteday'].dt.year - min_dataset_year
    
    # Recalcular 'mnth'
    df['mnth'] = df['dteday'].dt.month
    
    # Recalcular 'weekday' (Lunes=0, Domingo=6)
    df['weekday'] = df['dteday'].dt.dayofweek

    # Recalcular 'season' basado en el mes
    df['season'] = df['dteday'].dt.month%12 // 3 + 1

    # Recalcular 'holiday' basado en la lista de fechas fijas, retorna 1 si es feriado, 0 si no lo es
    df['holiday'] = df['dteday'].dt.date.isin(HOLIDAYS_SET).astype(int)

    # Recalcular 'workingday': 1 si es día laborable, 0 si es fin de semana o feriado
    df['workingday'] = df.apply(lambda row: 1 if row['holiday'] == 0 and row['weekday'] < 5 else 0, axis=1)

    return df


def fix_hr(df: pd.DataFrame, hr: str = 'hr', idx: str = 'instant') -> pd.DataFrame:
    """
    Limpia e imputa la columna de la hora ('hr') secuencialmente dado que el dataset es un registro secuencial (mod 24) 
    utilizando operaciones vectorizadas de Pandas y Numpy.
    """
    out = df.copy()
    
    # 1. Limpieza inicial y preparación
    out = out.sort_values(idx).reset_index(drop=True)
    
    # Coercer a numérico y establecer outliers (<0 o >23) a NaN
    s = pd.to_numeric(out[hr], errors='coerce')
    s.loc[(s < 0) | (s > 23)] = np.nan
    
    # 2. Imputación de Gaps Iniciales (Equivalente al backward pass)
    # Si hay NaNs al principio, los imputamos primero: (siguiente - pasos) mod 24
    first_valid_idx = s.first_valid_index()
    if first_valid_idx is not None and first_valid_idx > 0:
        first_known_hour = s.iloc[first_valid_idx]
        
        # Array de pasos hacia atrás (ej. -1, -2, -3...)
        steps_back_arr = np.arange(first_valid_idx, 0, -1) * -1 
        
        # Imputar: (valor conocido + pasos hacia atrás) mod 24
        s.iloc[:first_valid_idx] = (first_known_hour + steps_back_arr) % 24
            
    # 3. Imputación Vectorizada de Gaps Internos (Equivalente al forward pass)
    
    # a. Obtener la última hora válida conocida para cada bloque de NaNs
    last_known_hour = s.ffill()
    
    # b. Crear un contador de grupo: se incrementa en cada valor no-NaN
    # Esto aísla los bloques de NaNs.
    group_counter = s.notna().cumsum()
    
    # c. Contar los pasos dentro de cada bloque de NaNs (cuántas horas faltan)
    steps_since_last_known = s.groupby(group_counter).cumcount()
    
    # d. Aplicar la imputación solo a los NaNs restantes
    # Imputación: (última hora conocida + pasos desde la última hora) mod 24
    imputed_values = (last_known_hour + steps_since_last_known) % 24
    s.loc[s.isna()] = imputed_values.loc[s.isna()]
    
    # 4. Formato final
    # Redondeo y conversión a Int64 para soportar NaN si aún quedan (aunque no deberían)
    out[hr] = s.round().astype('Int64')
    
    return out

def clip_features(X: pd.DataFrame, weather_features: list) -> pd.DataFrame:
    X = X.copy()
    for c in weather_features:
        X[c] = np.clip(X[c], 0, 1)
    X.loc[~X['weathersit'].isin([1, 2, 3, 4]), 'weathersit'] = np.nan
    return X


def build_imputation_pipeline(weather_features, categorical, demand_features, cols_out) -> Pipeline:
    """Construye y retorna el pipeline de imputación de sklearn."""
    
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('weather', KNNImputer(n_neighbors=5), weather_features),
            ('weathersit', SimpleImputer(strategy='most_frequent'), categorical),
            ('demand', IterativeImputer(max_iter=10, random_state=42), demand_features),
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        # Paso 1: Limpieza Pre-Imputación
        ('clip', FunctionTransformer(clip_features, kw_args={'weather_features': weather_features}, validate=False)),
        # Paso 2: Imputación (aplica la lógica de preprocessor)
        ('impute', preprocessor),
    ])
    return pipeline

def null_handling(df: pd.DataFrame, pipeline: Pipeline, cols_out:List) -> pd.DataFrame    :
    """Maneja valores nulos en el DataFrame."""
    df_out = df.copy()
    arr = pipeline.fit_transform(df_out)
    cols_passthrough = [c for c in df_out.columns if c not in cols_out]

    # === INICIO DE DIAGNÓSTICO TEMPORAL ===
    print("\n--- DIAGNÓSTICO DE COLUMNAS ---")
    print(f"Columnas de Input (df_out.columns): {list(df_out.columns)}")
    print(f"Columnas Esperadas (len): {len(cols_out) + len(cols_passthrough)}")
    print(f"Passthrough Esperadas: {cols_passthrough}")
    print(f"Columnas de Output Array (arr.shape[1]): {arr.shape[1]}")
    print("------------------------------\n")
    # === FIN DE DIAGNÓSTICO TEMPORAL ===

    df_imputed = pd.DataFrame(arr, columns=cols_out + cols_passthrough)
    df_imputed = df_imputed[df_out.columns]
    # Asegurar tipos correctos después de la imputación
    df_imputed['weathersit'] = df_imputed['weathersit'].round().clip(1, 4).astype(int)
    df_imputed[['casual','registered','cnt']] = (
    df_imputed[['casual','registered','cnt']].round().clip(lower=0).astype(int)
    )
    return df_imputed


def cap_outliers_by_group(
    df: pd.DataFrame,
    col: str,
    groupby: str,
    lower_q: float = 0.01,
    upper_q: float = 0.90,
    min_group_size: int = 50,
) -> pd.DataFrame:
    """
    Aplica winsorización (clipping) a una columna numérica con cuantiles 
    por grupo, usando cuantiles globales como fallback para grupos pequeños.
    """
    df_out = df.copy()
    s = pd.to_numeric(df_out[col], errors="coerce")

    # 1. Cuantiles Globales (Fallback)
    g_low, g_high = s.quantile([lower_q, upper_q])

    # 2. Cuantiles por Grupo (Vectorizado)
    q = (
        df_out[[groupby, col]]
        .dropna(subset=[col])
        .groupby(groupby)[col]
        .quantile([lower_q, upper_q])
        .unstack()
        .rename(columns={lower_q: "low", upper_q: "high"})
    )

    # 3. Mapeo y Fallback
    # Mapear los cuantiles por grupo a cada fila
    low = df_out[groupby].map(q["low"]).fillna(g_low)
    high = df_out[groupby].map(q["high"]).fillna(g_high)

    # Calcular tamaños de grupo
    sizes = df_out[[groupby, col]].dropna(subset=[col]).groupby(groupby)[col].size()
    
    # Identificar grupos pequeños
    small = df_out[groupby].map(sizes).fillna(0) < min_group_size
    
    # Aplicar Fallback: usar cuantiles globales donde el grupo es pequeño
    low = np.where(small, g_low, low)
    high = np.where(small, g_high, high)

    # 4. Aplicar Clip Vectorizado y Escribir de Vuelta
    df_out[col] = np.clip(s, low, high)
    
    return df_out


def winsorize_data(df: pd.DataFrame, features: List[str],lower_q: float, upper_q: float, groupby: str, min_group_size: int) -> pd.DataFrame:
    """
    Aplica winsorización por grupo a múltiples características numéricas.
    """
    df_out = df.copy()
    for feature in features:
        df_out = cap_outliers_by_group(
            df=df_out,
            col=feature,
            groupby=groupby,
            lower_q=lower_q,
            upper_q=upper_q,
            min_group_size=min_group_size
        )
    return df_out


def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()

    # 1. Limpieza de patrones inválidos
    df_out = clean_pattern_values(df_out, invalid_values=INVALID_PATTERNS)    

    # 2. Corrección de tipos de datos
    df_out = dtype_correction(df_out)

    # 3. Preparación de campos de fecha
    df_out = prepare_datefields(df_out)

    # 4. Corrección de la columna 'hr'
    df_out = fix_hr(df_out, hr='hr', idx='instant')

    # 5. Eliminación de columna instant para evitar problemas con el pipeline en el passthrough
    df_out = df_out.drop(columns=['instant'], errors='ignore')

    weather_features = ['temp', 'atemp', 'hum', 'windspeed']
    categorical = ['weathersit']
    demand_features = ['casual', 'registered', 'cnt']
    cols_out = weather_features + categorical + demand_features

    # 6. Construcción del pipeline de imputación
    pipeline = build_imputation_pipeline(weather_features=weather_features, categorical=categorical, demand_features=demand_features, cols_out=cols_out)

    # 7. Manejo de valores nulos
    df_out = null_handling(df_out, pipeline, cols_out=cols_out)

    # 8. Correccion de tipo de datos
    df_out = dtype_correction(df_out)

    # 9. Winsorización de características continuas
    df_out = winsorize_data(
        df_out,
        features=weather_features,
        lower_q=0.01,
        upper_q=0.95,
        groupby='weathersit',
        min_group_size=50
    )

    df_out = winsorize_data(
        df_out,
        features=['casual'],
        lower_q=0.01,
        upper_q=0.90,
        groupby='weathersit',
        min_group_size=50
    )

    df_out = winsorize_data(
        df_out,
        features=['registered','cnt'],
        lower_q=0.01,
        upper_q=0.985,
        groupby='weathersit',
        min_group_size=50
    )

    return df_out



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ 
    Función principal para ejecutar la limpieza, winsorización y transformación del dataset.
    Toma un archivo CSV de entrada, procesa los datos y guarda el dataset limpio en la ruta especificada.

    1. Carga de datos desde input_filepath.
    2. Limpieza y transformación del dataset.
    3. Guarda el dataset procesado en output_filepath.
    """
    logger = logging.getLogger(__name__)
    logger.info('Iniciando la fase de procesamiento del dataset de entrada')
    
    # 1. Carga de datos
    df_raw = load_data(input_filepath)
    
    # 2. Procesamiento del dataset
    logger.info('Procesando el dataset...')
    df_processed = process_dataset(df_raw)
    
    # 3. Limpieza de columnas innecesarias
    logger.info('Eliminando columnas innecesarias del dataset procesado')
    df_processed = df_processed.drop(columns=COLUMNS_TO_DROP, errors='ignore')

    # 4. Guardar el archivo procesado
    logger.info('Guardando el dataset procesado en la ruta de salida')
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