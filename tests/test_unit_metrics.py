# tests/test_unit_metrics.py

import pytest
import numpy as np
import pandas as pd
from src.models.train_model import evaluate_metrics # Importar la función de métricas

# ----------------------------------------------------------------------
# Pruebas de Evaluación de Métricas (evaluate_metrics)
# ----------------------------------------------------------------------

def test_metrics_calculation_perfect_score():
    """Verifica que el RMSE y R2 sean perfectos (0 y 1) cuando la predicción es exacta."""
    # Valores simulados en escala logarítmica
    y_true_log = np.array([1.5, 2.0, 3.5, 4.0])
    y_pred_log = np.array([1.5, 2.0, 3.5, 4.0])
    
    metrics = evaluate_metrics(pd.Series(y_true_log), y_pred_log)
    
    # El RMSE debe ser cero (o muy cercano)
    assert np.isclose(metrics['rmse_log'], 0.0, atol=1e-8)
    assert np.isclose(metrics['rmse_original'], 0.0, atol=1e-8)
    
    # El R2 debe ser uno
    assert np.isclose(metrics['r2_log'], 1.0, atol=1e-8)


def test_metrics_calculation_known_error():
    """Verifica que el RMSE_log calculado coincida con un valor manual simple."""
    # y_true_log: [1, 2, 3]
    # y_pred_log: [2, 3, 4] -> Error de +1 en cada punto
    y_true_log = np.array([1.0, 2.0, 3.0])
    y_pred_log = np.array([2.0, 3.0, 4.0])
    
    # El error cuadrático medio (MSE) es (1^2 + 1^2 + 1^2) / 3 = 1
    # El RMSE (Raíz del MSE) debe ser 1
    expected_rmse_log = 1.0 
    
    metrics = evaluate_metrics(pd.Series(y_true_log), y_pred_log)
    
    # Comprobar el RMSE en escala logarítmica
    assert np.isclose(metrics['rmse_log'], expected_rmse_log, atol=1e-8)


def test_metrics_output_keys():
    """Verifica que la función retorne todas las métricas esperadas."""
    y_true_log = np.array([1.0, 2.0, 3.0])
    y_pred_log = np.array([1.1, 2.1, 3.1])

    metrics = evaluate_metrics(pd.Series(y_true_log), y_pred_log)
    
    # Lista de claves esperadas (revisar la función evaluate_metrics)
    expected_keys = ['rmse_log', 'mae_log', 'r2_log', 'rmse_original'] 
    
    assert isinstance(metrics, dict)
    
    # Aserción modificada para verificar que TODAS las claves esperadas existen
    missing_keys = set(expected_keys) - set(metrics.keys())
    assert not missing_keys, f"Faltan claves de métricas: {missing_keys}"