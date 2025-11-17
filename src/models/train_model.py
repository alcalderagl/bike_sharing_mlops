import click
import logging
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import os
from dotenv import find_dotenv, load_dotenv

# Herramientas de Modelado y MLOps
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score, make_scorer,mean_absolute_error
# Importaciones de MLflow
import mlflow
import mlflow.xgboost 

# --- Configuración de MLflow ---
TARGET_COLUMN = 'cnt_log'
MODEL_NAME = "XGBoost_Bike_Sharing_Optimized"
MLFLOW_EXPERIMENT_NAME = "bike_sharing_time_series"

logger = logging.getLogger(__name__)


def evaluate_metrics(y_true: pd.Series, y_pred_log: np.ndarray) -> dict:
    """
    Calcula las métricas de evaluación clave (logarítmicas y en escala original).
    """
    
    # Métrica en escala logarítmica (usada para la optimización)
    rmse_log = np.sqrt(root_mean_squared_error(y_true, y_pred_log))
    r2_log = r2_score(y_true, y_pred_log)
    mae_log = mean_absolute_error(y_true, y_pred_log)
    
    # Deshacer la transformación Logarítmica para métricas originales
    y_true_original = np.expm1(y_true)
    y_pred_original = np.expm1(y_pred_log)
    
    # Asegurar no negativos y calcular RMSE original
    y_pred_original[y_pred_original < 0] = 0
    rmse_original = np.sqrt(root_mean_squared_error(y_true_original, y_pred_original))
    
    return {
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        "r2_log": r2_log,
        "rmse_original": rmse_original
    }

@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
def train_and_log_model(input_dir, model_filepath):
    """
    Carga los sets de Train/Validation, ejecuta el tuning con TimeSeriesSplit, 
    evalúa el mejor modelo y registra los resultados en MLflow.
    """
    
    # 1. Cargar Datos (archivos generados por build_features.py)
    try:
        input_path = Path(input_dir)
        X_train = pd.read_csv(input_path / 'X_train.csv')
        y_train = pd.read_csv(input_path / 'y_train.csv')[TARGET_COLUMN] 
        X_val = pd.read_csv(input_path / 'X_val.csv')
        y_val = pd.read_csv(input_path / 'y_val.csv')[TARGET_COLUMN] 
    except Exception as e:
        logger.error(f"Error cargando datasets desde {input_dir}: {e}")
        return

    # 2. Configuración de MLflow: Se establece el experimento.
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run() as run:
        
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        logger.info("Iniciando tuning con TimeSeriesSplit...")
        
        # Parámetros para el tuning (Basado en la optimización previa)
        param_grid = {
            'n_estimators': [200, 400, 600], 
            'max_depth': [5, 7],
            'learning_rate': [0.05, 0.1]
        }
        
        # Configurar Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=5) 
        
        # Usamos RMSE negativo para scoring (se maximiza)
        rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)
        
        # 3. Ejecutar GridSearchCV
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')
        
        grid_search = GridSearchCV(
            estimator=xgb_model, 
            param_grid=param_grid, 
            scoring=rmse_scorer, 
            cv=tscv, 
            verbose=1, 
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # 4. Evaluación y Métricas
        best_xgb = grid_search.best_estimator_
        y_pred_log = best_xgb.predict(X_val)
        
        metrics = evaluate_metrics(y_val, y_pred_log)
        
        # 5. Registro en MLflow
        
        # Registrar Hiperparámetros
        mlflow.log_params(grid_search.best_params_)
        
        # Registrar Métricas Finales
        for key, value in metrics.items():
             mlflow.log_metric(key, value)
        
        # Registrar el modelo final (artefacto)
        mlflow.xgboost.log_model(
            xgb_model=best_xgb, 
            name="model", 
            registered_model_name=MODEL_NAME
        )
        
        # Guardar el modelo localmente (Artefacto para el entregable)
        Path(model_filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_xgb, model_filepath)
        
        # Salida final
        logger.info("\n" + "="*50)
        logger.info(f"ENTRENAMIENTO FINALIZADO Y REGISTRADO EN MLFLOW")
        logger.info(f"-> RMSE Original: {metrics['rmse_original']:.2f} bicicletas")
        logger.info(f"-> R2 Log: {metrics['r2_log']:.4f}")
        logger.info(f"Mejores Parámetros: {grid_search.best_params_}")
        logger.info(f"Modelo registrado en MLflow como: {MODEL_NAME}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    
    # Lógica de ejecución de prueba
    project_dir = Path(__file__).resolve().parents[2]
    input_directory = project_dir / 'data' / 'interim' 
    model_file = project_dir / 'models' / 'final_xgb_model.pkl'
    
    train_and_log_model.callback(str(input_directory), str(model_file))
