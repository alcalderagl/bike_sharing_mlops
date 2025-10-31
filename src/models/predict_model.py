import click
import logging
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from dotenv import find_dotenv, load_dotenv

# Definición de la variable objetivo (usada para nombrar la columna de predicción)
PREDICTION_COLUMN = 'cnt_prediction'
TARGET_COLUMN = 'cnt'

@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--inverse_transform', is_flag=True, default=True, help='Aplica la transformación inversa (np.expm1) a las predicciones.')
def main(model_filepath, input_filepath, output_filepath, inverse_transform):
    """
    Carga el Pipeline entrenado y genera predicciones sobre un nuevo conjunto de datos.
    
    Args:
        model_filepath (str): Ruta al archivo .pkl del Pipeline completo.
        input_filepath (str): Ruta al archivo CSV con los datos de entrada (X_test.csv).
        output_filepath (str): Ruta donde se guardará el CSV con las predicciones.
        inverse_transform (bool): Si es True, aplica la transformación inversa (expm1) 
                                  a la predicción final.
    """
    logger = logging.getLogger(__name__)
    logger.info("Iniciando la generación de predicciones.")
    
    # 1. Cargar el Pipeline y los datos
    try:
        full_pipeline = joblib.load(model_filepath)
        logger.info(f"Pipeline cargado exitosamente desde: {model_filepath}")
        
        X_data = pd.read_csv(input_filepath)
        
        # Eliminar la variable objetivo si existe en los datos de entrada (para evitar errores)
        if TARGET_COLUMN in X_data.columns:
            X_data = X_data.drop(columns=[TARGET_COLUMN], errors='ignore')
            
    except Exception as e:
        logger.error(f"Error al cargar recursos: {e}")
        return

    # 2. Generar Predicciones (El Pipeline se encarga de Preprocesar + Predecir)
    logger.info(f"Generando predicciones en {len(X_data)} filas.")
    predictions_log_scale = full_pipeline.predict(X_data)

    # 3. Aplicar Transformación Inversa (si es necesario)
    if inverse_transform:
        # np.expm1() es el inverso de np.log1p()
        predictions_original_scale = np.expm1(predictions_log_scale)
        predictions = predictions_original_scale
        logger.info("Predicciones transformadas a la escala original (renta de bicis).")
    else:
        predictions = predictions_log_scale
        logger.info("Predicciones mantenidas en la escala logarítmica.")

    # 4. Guardar las Predicciones
    
    # Crear un DataFrame con las predicciones y el índice original
    predictions_df = pd.DataFrame(predictions, columns=[PREDICTION_COLUMN], index=X_data.index)
    
    # Opcional: Juntar las predicciones con los datos originales para análisis
    output_df = pd.concat([X_data, predictions_df], axis=1)
    
    # Asegurarse de que el directorio de salida exista
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    
    output_df.to_csv(output_filepath, index=False)
    logger.info(f"Predicciones guardadas en: {output_filepath}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    
    # Rutas de ejemplo para la ejecución
    project_dir = Path(__file__).resolve().parents[2]
    model_file = project_dir / 'models' / 'final_pipeline.pkl'
    input_file = project_dir / 'data' / 'interim' / 'X_test.csv' # Usamos el X_test como ejemplo
    output_file = project_dir / 'data' / 'results' / 'predictions_final.csv'    
    # Argumentos que pasarías al script en la terminal:
    # python src/models/predict_model.py models/final_pipeline.pkl data/interim/X_test.csv data/interim/predictions_final.csv
    
    try:
        main([str(model_file), str(input_file), str(output_file)])
    except Exception as e:
        logging.getLogger(__name__).error(f"Fallo al ejecutar predict_model.py: {e}")