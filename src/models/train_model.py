import click
import logging
from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import find_dotenv, load_dotenv
import numpy as np

# Herramientas de MLOps (Punto 4)
import mlflow
import mlflow.sklearn

TARGET_COLUMN = 'cnt'

def evaluate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Calcula las métricas de evaluación para el problema de regresión."""
    
    # IMPORTANTE: y_true (target) está en escala logarítmica (log1p).
    # Para el cálculo de MSE, MAE, R2, necesitamos las predicciones en escala logarítmica
    # y las etiquetas reales también están en escala logarítmica.
    
    # Si quieres reportar las métricas en la escala original (renta de bicis real),
    # deberías aplicar la inversa: np.expm1(y_true) y np.expm1(y_pred).
    
    # Mantendremos la evaluación en escala logarítmica para coherencia con el entrenamiento.
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculamos también el RMSE en la escala original para tener una métrica interpretable
    # En MLflow registraremos ambas.
    rmse_original_scale = np.sqrt(mean_squared_error(np.expm1(y_true), np.expm1(y_pred)))
    
    return {
        "rmse_log": rmse,
        "mae_log": mae,
        "r2_log": r2,
        "rmse_original": rmse_original_scale # Métrica para interpretación humana
    }


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('preprocessor_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
def main(input_dir, preprocessor_filepath, model_filepath):
    """
    Entrena el modelo de Machine Learning, encapsulado en un Pipeline,
    y registra el experimento con MLflow. (Puntos 3 y 4)
    """
    logger = logging.getLogger(__name__)
    
    # --- Configuración de MLflow (Punto 4) ---
    mlflow.set_experiment("Bike Sharing MLOps Regressor")
    
    with mlflow.start_run() as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        
        # 1. Cargar datos y preprocesador
        logger.info("Cargando datos de entrenamiento/prueba y preprocesador.")
        input_path = Path(input_dir)
        
        X_train = pd.read_csv(input_path / 'X_train.csv')
        y_train = pd.read_csv(input_path / 'y_train.csv')[TARGET_COLUMN]
        X_test = pd.read_csv(input_path / 'X_test.csv')
        y_test = pd.read_csv(input_path / 'y_test.csv')[TARGET_COLUMN]
        
        preprocessor = joblib.load(preprocessor_filepath)

        # 2. Definición del Modelo y Pipeline (Punto 3)
        # Random Forest Regressor (Modelo usado en 02_feature_eng.ipynb)
        
        # Definición de hiperparámetros (para el registro en MLflow)
        rf_params = {
            "n_estimators": 200,         # Un ajuste típico
            "max_depth": 15,             # Controlar la complejidad
            "random_state": 42,
            "min_samples_split": 5       # Ajuste basado en el 02_feature_eng.ipynb
        }
        
        model = RandomForestRegressor(**rf_params)
        
        # Creación del Pipeline Final: Preprocesador + Modelo (Punto 3)
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        logger.info("Pipeline creado y listo para entrenamiento.")

        # 3. Entrenamiento del Modelo
        full_pipeline.fit(X_train, y_train)
        logger.info("Entrenamiento del modelo completado.")

        # 4. Evaluación y Registro de Experimento (Punto 4)
        
        # Predicciones
        predictions = full_pipeline.predict(X_test)
        
        # Cálculo de Métricas
        metrics = evaluate_metrics(y_test, predictions)
        
        logger.info(f"Métricas (Log Scale): RMSE={metrics['rmse_log']:.4f}, R2={metrics['r2_log']:.4f}")
        logger.info(f"Métrica en escala original: RMSE={metrics['rmse_original']:.2f}")

        # Registro en MLflow
        mlflow.log_params(rf_params) # Hiperparámetros
        mlflow.log_metrics(metrics)  # Métricas
        
        # Etiquetar el tipo de modelo
        mlflow.set_tag("model_type", "RandomForest")
        
        # Registrar el Pipeline COMPLETO (Punto 4: Gestión de Modelos)
        # Esto incluye el preprocesador ajustado y el modelo entrenado.
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path="bike_sharing_model",
            registered_model_name="BikeSharingRandomForest" # Registrar en Model Registry
        )
        
        logger.info(f"Experimento y modelo registrados en MLflow. Model Name: BikeSharingRandomForest")

        # 5. Guardar el modelo localmente (Artefacto para el entregable)
        Path(model_filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(full_pipeline, model_filepath)
        logger.info(f"Pipeline completo guardado localmente en: {model_filepath}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    
    # Rutas de ejemplo para la ejecución
    project_dir = Path(__file__).resolve().parents[2]
    input_dir = project_dir / 'data' / 'interim' # X_train, y_train, etc.
    preprocessor_file = project_dir / 'models' / 'preprocessor.pkl'
    model_file = project_dir / 'models' / 'final_pipeline.pkl'
    
    # Argumentos que pasarías al script en la terminal:
    # python src/models/train_model.py data/interim models/preprocessor.pkl models/final_pipeline.pkl
    
    try:
        main([str(input_dir), str(preprocessor_file), str(model_file)])
    except Exception as e:
        logging.getLogger(__name__).error(f"Fallo al ejecutar train_model.py: {e}")