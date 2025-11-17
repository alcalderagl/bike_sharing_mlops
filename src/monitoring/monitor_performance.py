import click
import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

# Importamos utilidades de otros módulos
from src.features.build_features import FINAL_FEATURES
from src.models.train_model import evaluate_metrics # Para calcular RMSE

logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN DE UMBRALES ---
UMBRAL_DEGRADACION_PCT = 10.0 # Alerta si el RMSE aumenta más de 10%
MODEL_PATH = 'models/final_xgb_model.pkl'
DATA_PATH = 'data/interim/'


def simulate_data_drift(X_val: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un conjunto de datos de monitoreo alterando la distribución de features
    críticas (Sensor Drift, Pipeline Drift y Holiday Drift).
    """
    logger.info("Iniciando simulación de Data Drift...")
    X_drift = X_val.copy()
    drift_size = int(0.3 * len(X_drift)) 
    
    # Índices para aplicar los drifts (aseguramos que sean disjuntos)
    idx_sensor = X_drift.index[:drift_size]
    idx_pipeline = X_drift.index[drift_size: 2 * drift_size]
    idx_holiday = X_drift.index[2 * drift_size: 3 * drift_size]

    # 1. Sensor Drift: Aumentar la temperatura 10% (sensor descalibrado)
    X_drift.loc[idx_sensor, 'temp'] = X_drift.loc[idx_sensor, 'temp'] * 1.10
    logger.info(f" -> Sensor Drift (temp) aplicado a {len(idx_sensor)} registros.")

    # 2. Pipeline Drift: Simular que el lag de 24h es 0 (fallo en el pipeline de lag)
    if 'cnt_lag24_log' in X_drift.columns:
        X_drift.loc[idx_pipeline, 'cnt_lag24_log'] = 0.0
        logger.info(f" -> Pipeline Drift (cnt_lag24_log) aplicado a {len(idx_pipeline)} registros.")
        
    # 3. Holiday Drift (External Data Drift): Simular predicciones para un año fuera de HOLIDAYS_SET
    # Si el dataset cubre 2011/2012 (yr=0/1), simulamos 2013 (yr=2).
    if 'yr' in X_drift.columns and 'holiday' in X_drift.columns:
        # A. Alterar el año a uno fuera del rango (ej. 2013, codificado como 2)
        X_drift.loc[idx_holiday, 'yr'] = 2 
        
        # B. Invertir la feature 'holiday' para simular que el HolidaySet no cubre el nuevo año, 
        # resultando en una clasificación errónea de festivos.
        # (Donde era 0 lo ponemos a 1, y donde era 1 lo ponemos a 0, creando el error)
        X_drift.loc[idx_holiday, 'holiday'] = 1 - X_drift.loc[idx_holiday, 'holiday']
        logger.info(f" -> Holiday Drift (yr y holiday) aplicado a {len(idx_holiday)} registros.")
        
    return X_drift

def evaluate_performance_degradation(
    model, 
    X_val: pd.DataFrame, 
    y_val: pd.Series, 
    X_drift: pd.DataFrame, 
    y_drift: pd.Series
) -> None:
    """
    Evalúa el desempeño en el set con drift y genera una alerta si se supera el umbral.
    """
    
    # 1. Evaluación Línea Base (Validation Set)
    y_pred_baseline_log = model.predict(X_val)
    metrics_baseline = evaluate_metrics(y_val, y_pred_baseline_log)
    rmse_baseline = metrics_baseline['rmse_original']
    
    # 2. Evaluación Set con Drift
    y_pred_drift_log = model.predict(X_drift)
    metrics_drift = evaluate_metrics(y_drift, y_pred_drift_log)
    rmse_drift = metrics_drift['rmse_original']
    
    # 3. Análisis de Degradación
    degradacion_abs = rmse_drift - rmse_baseline
    degradacion_pct = (degradacion_abs / rmse_baseline) * 100
    
    logger.info("\n" + "="*50)
    logger.info("DIAGNÓSTICO DE PÉRDIDA DE PERFORMANCE")
    logger.info(f"RMSE Línea Base (Validación): {rmse_baseline:.2f} bicicletas")
    logger.info(f"RMSE Set con Drift: {rmse_drift:.2f} bicicletas")
    logger.info(f"Degradación: {degradacion_pct:.2f}%")
    logger.info("="*50)

    # 4. Generación de Alerta
    if degradacion_pct > UMBRAL_DEGRADACION_PCT:
        logger.error(f"🚨 ALERTA: La degradación del {degradacion_pct:.2f}% supera el umbral del {UMBRAL_DEGRADACION_PCT:.2f}%")
        logger.error("-> ACCIÓN PROPUESTA: Revisar el Feature Pipeline y forzar un Retrain del modelo.")
    else:
        logger.info(f"🟢 OK: El desempeño está dentro del umbral. Degradación de {degradacion_pct:.2f}%.")


@click.command()
@click.option('--model_filepath', type=click.Path(exists=True), default=MODEL_PATH, help='Ruta al modelo serializado (.pkl).')
@click.option('--data_dir', type=click.Path(exists=True), default=DATA_PATH, help='Directorio donde se encuentran X_val.csv y y_val.csv.')
def main(model_filepath: str, data_dir: str):
    """
    Carga el modelo y los datos de validación, simula Data Drift y detecta la pérdida de performance.
    """
    logger.info("Iniciando Monitoreo de Data Drift...")
    
    try:
        # 1. Cargar Modelo
        model = joblib.load(model_filepath)
        logger.info(f"Modelo cargado desde: {model_filepath}")
        
        # 2. Cargar Datos de Validación (Línea Base)
        X_val = pd.read_csv(Path(data_dir) / 'X_val.csv')
        y_val = pd.read_csv(Path(data_dir) / 'y_val.csv')['cnt_log']
        
        # 3. Simular Drift
        # Usamos X_val como base, ya que es el set más cercano a la realidad de producción
        X_drift = simulate_data_drift(X_val) 
        y_drift = y_val.copy() # Asumimos que el target real no está alterado.

        # 4. Alinear Features (CRÍTICO para XGBoost)
        # Esto asegura que X_val y X_drift tienen las columnas en el orden exacto del entrenamiento.
        X_val_aligned = X_val[FINAL_FEATURES]
        X_drift_aligned = X_drift[FINAL_FEATURES]

        # 5. Evaluar y Alertar
        evaluate_performance_degradation(model, X_val_aligned, y_val, X_drift_aligned, y_drift)
        
    except FileNotFoundError as e:
        logger.error(f"Error al cargar archivos. Asegúrate de que el training se haya ejecutado: {e}")
    except Exception as e:
        logger.error(f"Ocurrió un error inesperado durante el monitoreo: {e}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()