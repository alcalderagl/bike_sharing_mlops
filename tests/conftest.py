import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

import sys
sys.path.append(str(Path(__file__).parent.parent))

@pytest.fixture(scope="session")
def project_root():
    """Retorna la ruta raíz del proyecto."""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def sample_data():
    """
    Fixture para crear un DataFrame de datos crudos simulados con 
    los problemas conocidos del dataset original (strings, NaNs, etc.)
    """
    data = {
        'instant': range(1, 11), 'dteday': pd.to_datetime(['2011-01-01','2011-01-17', '2011-01-03', '2011-01-04', '2011-01-05', '2011-01-06', '2011-01-07', '2011-01-08', '2011-01-09', '2011-01-10']),
        'season': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'yr': [0, 0, 0, '2011', 0, 0, 0, 0, 0, 0], 
        'mnth': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'hr': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'holiday': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'weekday': [6, 0, 1, 2, 3, 4, 5, 6, 0, 1],
        'workingday': [0, 0, 1, 1, 1, 1, 1, 0, 0, 1], 'weathersit': ['unknown', 1, 1, 2, 1, 2, 1, 1, 2, 1],
        'temp': [0.24, 0.22, 0.2, 0.22, 0.2, 0.25, 0.24, 0.2, 0.18, 0.2],
        'atemp': [0.2879, 0.2727, 0.2576, 0.2727, 0.2576, 0.2879, 0.2727, 0.2576, 0.2273, 0.2424],
        'hum': [0.81, 0.8, 0.75, 0.8, 0.75, 0.81, 0.8, 0.75, 0.8, 0.75],
        'windspeed': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'casual': [3, 8, 5, 1, 0, 4, 1, 5, 2, 7], 
        'registered': [13, 32, 23, 10, 5, 16, 8, 20, 15, 25], 
        'cnt': [16, 40, 28, 11, 5, 20, 9, 25, 17,32], 
        'mixed_type_col': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'] 
    }
    df = pd.DataFrame(data)
    return df

# conftest.py (ADICIÓN DE NUEVA FIXTURE)

@pytest.fixture(scope="session")
def clean_data_for_fe():
    """
    Fixture: DataFrame limpio con una secuencia de 50 horas, listo para Feature Engineering.
    Añadido: 'cnt' (target original) y 'dteday' como columna regular.
    """
    N = 50
    np.random.seed(42)
    cnt_base = np.arange(1, N + 1) + np.random.randint(1, 10, N)
    data = {
        'dteday': pd.to_datetime(pd.Series(range(N)).apply(lambda x: f"2011-01-01 {x%24:02d}:00:00")),
        'hr': [i % 24 for i in range(N)],
        'cnt': cnt_base, 
        'cnt_log': np.log1p(cnt_base),
        'temp': np.linspace(0.2, 0.8, N),
        'hum': np.linspace(0.5, 0.9, N),
        'windspeed': np.linspace(0.1, 0.4, N),
        'yr': 0, 'holiday': 0, 'workingday': 1, 'weathersit': 1, 'season': 1, 'mnth': 1, 'weekday': 1
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture(scope="session")
def trained_pipeline(project_root):
    """Carga el pipeline final entrenado para pruebas de inferencia."""
    model_path = project_root / 'models' / 'final_xgb_model.pkl'
    if not model_path.exists():
        pytest.skip(f"No se encontró el pipeline entrenado en: {model_path}. Ejecuta 'dvc repro' primero.")
    return joblib.load(model_path)

@pytest.fixture(scope="session")
def X_test_data(project_root):
    """Carga los datos de X_test para la prueba de integración y predicción."""
    x_test_path = project_root / 'data' / 'interim' / 'X_test.csv'
    if not x_test_path.exists():
        pytest.skip(f"No se encontró X_test.csv. Ejecuta 'dvc repro' primero.")
    # Asegura que 'dteday' se cargue como objeto/string para simular entrada de la API
    return pd.read_csv(x_test_path)