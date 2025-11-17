import click
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import find_dotenv, load_dotenv
import os

TARGET_COLUMN = 'cnt_log' # Nombre de la columna de entrada (logarítmica)
PREDICTION_COLUMN = 'cnt_prediction' # Nombre de la columna de predicción (original)

@click.command()
@click.argument('predictions_filepath', type=click.Path(exists=True))
@click.argument('test_data_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(predictions_filepath, test_data_filepath, output_filepath):
    """
    Genera y guarda un gráfico de comparación de Predicciones vs. Valores Reales 
    sobre el conjunto de validación (validation set).
    """
    logger = logging.getLogger(__name__)
    logger.info("Iniciando la visualización de resultados.")

    # 1. Cargar datos
    try:
        # Carga el archivo de predicciones (generado por predict_model.py, con dteday, hr y cnt_prediction)
        df_predictions = pd.read_csv(predictions_filepath)
        
        # Carga los valores reales de Y (y_val.csv)
        df_y_val = pd.read_csv(test_data_filepath)
        
    except Exception as e:
        logger.error(f"Error al cargar datos para visualización: {e}")
        return

    # 2. Preparar los datos y aplicar transformaciones inversas
    
    # Unir la columna del target real transformado (cnt_log) al DataFrame de predicciones
    # Aseguramos que los índices se alineen, asumiendo que ambos archivos se guardaron en orden cronológico.
    # Usamos TARGET_COLUMN ('cnt_log') ya que df_y_val solo tiene esa columna.
    df_predictions[TARGET_COLUMN] = df_y_val[TARGET_COLUMN].values
    
    # Aplicar la transformación inversa (expm1) para graficar en la escala original
    # La predicción ya está en escala original por predict_model.py, pero la columna real no.
    df_predictions[f'{TARGET_COLUMN}_real'] = np.expm1(df_predictions[TARGET_COLUMN])
    
    # Creamos un índice de tiempo para el eje X
    df_predictions['time_index'] = range(len(df_predictions))

    # 3. Generar el gráfico (Gráfico de línea de Predicción vs. Real)
    plt.figure(figsize=(14, 6))
    
    # Graficar la serie de tiempo de la demanda real
    sns.lineplot(x='time_index', y=f'{TARGET_COLUMN}_real', 
                 data=df_predictions, label='Demanda Real', alpha=0.7, color='blue')
    
    # Graficar la predicción del modelo (ya en escala original)
    sns.lineplot(x='time_index', y=PREDICTION_COLUMN, # Usamos la columna final de predict_model.py
                 data=df_predictions, label='Predicción del Modelo', alpha=0.7, color='orange')
    
    plt.title('Comparación de Predicciones vs. Valores Reales (Conjunto de Validación)')
    plt.xlabel('Índice de Tiempo (Orden Cronológico)')
    plt.ylabel('Demanda Total de Bicicletas (cnt)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 4. Guardar el resultado
    output_path = Path(output_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Gráfico de resultados guardado en: {output_filepath}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    
    # Rutas de ejemplo
    project_dir = Path(__file__).resolve().parents[2]
    # El output de predict_model.py (predictions_final.csv)
    predictions_file = project_dir / 'data' / 'results' / 'predictions_final.csv' 
    # El target del validation set (output de build_features.py)
    validation_target_file = project_dir / 'data' / 'interim' / 'y_val.csv' 
    output_file = project_dir / 'reports' / 'figures' / 'predictions_vs_real.png'
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    main.callback(str(predictions_file), str(validation_target_file), str(output_file))
