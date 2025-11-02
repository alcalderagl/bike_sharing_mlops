import click
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import find_dotenv, load_dotenv

TARGET_COLUMN = 'cnt'
PREDICTION_COLUMN = 'cnt_prediction'

@click.command()
@click.argument('predictions_filepath', type=click.Path(exists=True))
@click.argument('test_data_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(predictions_filepath, test_data_filepath, output_filepath):
    """
    Genera y guarda un gráfico de comparación de Predicciones vs. Valores Reales 
    (replicando la visualización clave de 02_feature_eng.ipynb).
    """
    logger = logging.getLogger(__name__)
    logger.info("Iniciando la visualización de resultados.")

    # 1. Cargar datos
    try:
        # Carga el archivo de predicciones (generado por predict_model.py)
        df_predictions = pd.read_csv(predictions_filepath)
        
        # Carga los valores reales de Y para la comparación
        df_y_test = pd.read_csv(test_data_filepath)
        
    except Exception as e:
        logger.error(f"Error al cargar datos para visualización: {e}")
        return

    # 2. Preparar los datos y aplicar transformaciones inversas
    
    # El archivo de predicciones ya tiene las features de X_test.csv y la columna 'cnt_prediction'.
    # Necesitamos añadir la columna del valor real (y_test) para comparar.
    # El archivo 'y_test.csv' contiene el target transformado (log1p), así que lo unimos.
    # NOTA: Los índices deben coincidir después de la carga si se guardaron y_test con index=False.
    
    # Aseguramos que la columna del target real exista.
    df_predictions[TARGET_COLUMN] = df_y_test[TARGET_COLUMN].values
    
    # La variable objetivo (TARGET_COLUMN) y la predicción (PREDICTION_COLUMN) están en escala logarítmica.
    # Para el gráfico final, aplicaremos la transformación inversa (expm1) para mostrar valores reales de bicicletas.
    
    # Aplicar inversa (expm1) a la predicción y al valor real para la visualización.
    df_predictions[f'{PREDICTION_COLUMN}_real'] = np.expm1(df_predictions[PREDICTION_COLUMN])
    df_predictions[f'{TARGET_COLUMN}_real'] = np.expm1(df_predictions[TARGET_COLUMN])
    
    # Si la columna 'dteday' no fue eliminada al guardar X_test.csv, podrías usarla aquí.
    # Dado que la eliminamos, usaremos un índice de tiempo simple para el gráfico.
    df_predictions['time_index'] = range(len(df_predictions))

    # 3. Generar el gráfico (Gráfico de línea de Predicción vs. Real)
    plt.figure(figsize=(14, 6))
    
    # Graficar la serie de tiempo de la demanda real
    sns.lineplot(x='time_index', y=f'{TARGET_COLUMN}_real', 
                 data=df_predictions, label='Demanda Real', alpha=0.7, color='blue')
    
    # Graficar la predicción del modelo
    sns.lineplot(x='time_index', y=f'{PREDICTION_COLUMN}_real', 
                 data=df_predictions, label='Predicción del Modelo', alpha=0.7, color='orange')
    
    plt.title('Comparación de Predicciones vs. Valores Reales (Escala Original)')
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
    predictions_file = project_dir / 'data' / 'results' / 'predictions_final.csv'
    test_target_file = project_dir / 'data' / 'interim' / 'y_test.csv'
    output_file = project_dir / 'reports' / 'figures' / 'predictions_vs_real.png'
    
    # Asegúrate de que el directorio 'reports/figures' exista
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        main([str(predictions_file), str(test_target_file), str(output_file)])
    except Exception as e:
        logging.getLogger(__name__).error(f"Fallo al ejecutar visualize.py: {e}")