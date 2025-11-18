import pytest
import pandas as pd
import numpy as np
# Importamos la función clean_data y la lista de fechas fijas
from src.data.make_dataset import process_dataset,clean_pattern_values,prepare_datefields,null_handling,build_imputation_pipeline,dtype_correction,HOLIDAYS_SET,INVALID_PATTERNS
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def test_removing_patterns(sample_data):
    """Verifica que los patrones conocidos (strings, NaNs) sean eliminados."""
    X = clean_pattern_values(sample_data, INVALID_PATTERNS)

    for col in sample_data.select_dtypes(include=["object"]).columns:
        # 1) No debe quedar ningún patrón inválido tal cual en la columna
        assert not X[col].isin(INVALID_PATTERNS).any(), f"Quedaron valores inválidos en {col}"

        # 2) Donde antes había un valor inválido, ahora debe haber NaN
        mask_invalid = sample_data[col].isin(INVALID_PATTERNS)
        if mask_invalid.any():
            assert X.loc[mask_invalid, col].isna().all(), f"Valores inválidos no fueron convertidos a NaN en {col}"

        # 3) Donde antes había valores válidos, no deberían haber cambiado de contenido
        mask_valid = ~mask_invalid
        # Evitamos comparar NaNs directamente; solo comparamos strings válidos
        mismatched = X.loc[mask_valid, col].astype(str) != sample_data.loc[mask_valid, col].astype(str)
        assert mismatched.sum() == 0, (
            f"Valores válidos modificados en {col}.\n"
            f"Filas con error:\n"
            f"{pd.DataFrame({'original': sample_data.loc[mask_valid, col], 'nuevo': X.loc[mask_valid, col]})[mismatched]}"
        )
        #assert (X.loc[mask_valid, col] == sample_data.loc[mask_valid, col]).all(), f"Valores válidos fueron modificados en {col}"


def test_preparation_for_date_features(sample_data):

    result = prepare_datefields(sample_data)
    
    # Como base_year es None, min_dataset_year = 2011 -> yr = año - 2011
    assert result.loc[0, "yr"] == 0
    assert result.loc[1, "yr"] == 0

    # Mes (enero = 1)
    assert result["mnth"].unique().tolist() == [1]

    # weekday: Lunes=0,... Domingo=6
    # 2011-01-01 fue sábado (5), 2011-01-03 fue lunes (0)
    assert set(result["weekday"].unique()) == {6, 0, 1, 2, 3, 4, 5}

    # season: month%12 // 3 + 1
    # Enero -> 1, así que season = 1
    assert result["season"].unique().tolist() == [1]

    # holiday: 1 si la fecha está en HOLIDAYS_SET
    assert set(result["holiday"].unique()) == {1, 0}

    # workingday: 1 si no es feriado y es weekday<5
    # 2011-01-01: sábado y feriado -> 0
    # 2011-01-03: lunes, no feriado -> 1
    assert set(result["workingday"].unique()) == {0, 1}


def test_prepare_datefields_invalid_dates_removed(sample_data):
    """
    Verifica que:
    - Las fechas inválidas se convierten en NaT y luego se eliminan.
    - Se emite un warning indicando cuántas filas se eliminaron.
    """
    result = prepare_datefields(sample_data, base_year=2011)

    assert len(result) > 1
    assert all(result["dteday"].notna())


def test_null_handling_imputes_and_clips(sample_data):
    """
    Verifica que null_handling:
    - Imputa valores nulos en las columnas numéricas definidas en cols_out.
    - Mantiene el orden original de columnas.
    - Aplica correctamente round/clip y tipos en weathersit, casual, registered, cnt.
    """
    X = clean_pattern_values(sample_data,INVALID_PATTERNS)
    X = dtype_correction(X)
    X_test = X.drop(columns=['instant'], errors='ignore')

    # passthrough_cols = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday']
    # for col in passthrough_cols:
    #     if col in X_test.columns:
    #         # Reemplaza NaNs con -1 temporalmente para forzar el astype, luego pon NaN de nuevo.
    #         # Esto es necesario si clean_pattern_values generó NaNs.
    #         X_test[col] = pd.to_numeric(X_test[col], errors='coerce').astype('Int64')


    # Pipeline realista: imputamos solo las columnas numéricas (cols_out)
    # y dejamos el resto (temp) como passthrough
    weather_features = ['temp', 'atemp', 'hum', 'windspeed']
    categorical = ['weathersit']
    demand_features = ['casual', 'registered', 'cnt']
    cols_out = weather_features + categorical + demand_features

    pipeline = build_imputation_pipeline(weather_features=weather_features, categorical=categorical, demand_features=demand_features, cols_out=cols_out)

    print("Columnas originales de X_test:")
    print(X_test.columns)
    # Ejecutar la función a probar
    result = null_handling(X_test, pipeline, cols_out)
    print("Columnas resultantes después de null_handling:")
    print(result.columns)
    

    # 1) El número de columnas debe ser el mismo, y todas las columnas originales deben estar presentes.
    assert len(result.columns) == len(X_test.columns), "El número de columnas de salida no coincide con el de entrada."
    assert set(X_test.columns) == set(result.columns), "Las columnas de salida no coinciden con las de entrada."

    # 2) Chequeamos que el conjunto de columnas original esté contenido en el resultado
    assert set(X_test.columns).issubset(set(result.columns))
    # 3) Debe mantener el mismo orden de columnas
    assert list(result.columns) == list(X_test.columns)

    # 4) No debe haber NaNs en las columnas transformadas
    assert result[cols_out].isna().sum().sum() == 0

    # 5) weathersit debe estar entre 1 y 4 e int
    assert result["weathersit"].between(1, 4).all()
    assert result["weathersit"].dtype == int or np.issubdtype(result["weathersit"].dtype, np.integer)

    # 6) casual, registered, cnt deben ser enteros y no negativos
    for col in ["casual", "registered", "cnt"]:
        assert (result[col] >= 0).all(), f"Valores negativos encontrados en {col}"
        assert np.issubdtype(result[col].dtype, np.integer), f"{col} no es entero"

    # 7) La columna 'temp' debe seguir presente y sin modificaciones en cantidad de nulos
    assert "temp" in result.columns
    assert result["temp"].isna().sum() == X["temp"].isna().sum()
