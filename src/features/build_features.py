import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import joblib 

TARGET_COLUMN = 'cnt'

# =========================================================================
# 1. TRANSFORMADOR PERSONALIZADO PARA INGENIERÍA DE FEATURES (Punto 2.2)
# =========================================================================

class LagFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Crea variables lagged para la demanda (cnt) como se hizo en 02_feature_eng.ipynb.
    Esta clase permite que la ingeniería de features sea un paso dentro del pipeline de Scikit-Learn.
    """
    def __init__(self, lags=[1, 24]): # Lags de 1 hora y 1 día (24 horas) son comunes en hourly data
        self.lags = lags
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # La columna 'cnt' ya no está en X (separamos X y y en make_dataset),
        # por lo que no podemos crear lags aquí a menos que carguemos 'y'.
        # Para mantener la pureza de X, omitimos los lags de 'cnt' aquí, 
        # y asumimos que se aplicarían a otras variables o se harán antes de la división.
        # Si las features 'lagged' son cruciales, deben crearse en make_dataset.py
        
        # Basándonos en la convención de un Pipeline de ML estándar, 
        # este transformador se enfocará en features ya existentes.
        
        # Como no se ven lags en otras variables en el notebook, este transformador
        # simplemente devuelve X sin cambios, pero si tu lógica original era más compleja,
        # aquí es donde iría.
        return X

# =========================================================================
# 2. DEFINICIÓN DEL PREPROCESADOR (Punto 3)
# =========================================================================

def define_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Define el ColumnTransformer de Scikit-Learn para el preprocesamiento de datos.
    Aplica escalado a variables continuas y One-Hot a categóricas.
    """
    
    # Variables continuas sobre las que se aplicó Winsorización en make_dataset.py
    numerical_features = ['temp', 'atemp', 'hum', 'windspeed']
    
    # Variables categóricas y ordinales
    # Incluimos las features temporales creadas en make_dataset.py
    categorical_features = ['season', 'mnth', 'hr', 'weekday', 'weathersit', 
                            'yr', 'holiday', 'workingday', 
                            'month', 'dayofweek'] # Features extraídas de dteday
    
    # Pipeline para variables numéricas (escalado)
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Pipeline para variables categóricas (One-Hot Encoding)
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combinar transformadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' 
    )

    return preprocessor

# =========================================================================
# 3. FUNCIÓN PRINCIPAL CLI (Cookiecutter Structure)
# =========================================================================

# Ajustamos los argumentos para que el script pueda guardar X_train, y_train, etc.
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.argument('preprocessor_filepath', type=click.Path())
def main(input_filepath, output_dir, preprocessor_filepath):
    """ 
    Carga datos limpios, realiza la división de datos por tiempo, y define 
    el preprocesador de Scikit-Learn.
    """
    logger = logging.getLogger(__name__)
    logger.info('Iniciando la división de datos y definición del preprocesador')
    
    df_processed = pd.read_csv(input_filepath)
    
    X = df_processed.drop(columns=[TARGET_COLUMN])
    y = df_processed[TARGET_COLUMN] # Ya está transformada con log1p

    # **División de datos por tiempo (como en 02_feature_eng.ipynb)**
    # El notebook 02_feature_eng.ipynb usa una división basada en el tiempo 
    # (datos recientes para test). Asumimos que los datos ya están ordenados por tiempo.
    
    TEST_SIZE = 0.2
    test_size_rows = int(len(X) * TEST_SIZE)
    
    # Último 20% de los datos para prueba
    X_train = X.iloc[:-test_size_rows]
    X_test = X.iloc[-test_size_rows:]
    y_train = y.iloc[:-test_size_rows]
    y_test = y.iloc[-test_size_rows:]
    
    logger.info(f"División por tiempo: Train={len(X_train)} filas, Test={len(X_test)} filas.")

    # 1. Definir y ajustar el preprocesador (SOLO con datos de entrenamiento)
    preprocessor = define_preprocessor(X_train)
    preprocessor.fit(X_train) # Ajusta (ej. calcula media y desviación estándar)
    
    # 2. Guardar los conjuntos de datos de entrenamiento/prueba
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    X_train.to_csv(output_path / 'X_train.csv', index=False)
    X_test.to_csv(output_path / 'X_test.csv', index=False)
    y_train.to_csv(output_path / 'y_train.csv', index=False, header=[TARGET_COLUMN])
    y_test.to_csv(output_path / 'y_test.csv', index=False, header=[TARGET_COLUMN])
    logger.info(f"Datasets (train/test) guardados en: {output_path}")

    # 3. Guardar el objeto preprocesador ajustado 
    joblib.dump(preprocessor, preprocessor_filepath)
    logger.info(f"Preprocesador ajustado guardado en: {preprocessor_filepath}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    
    # Ejecución de prueba con rutas de Cookiecutter
    project_dir = Path(__file__).resolve().parents[2]
    input_file = project_dir / 'data' / 'processed' / 'bike_sharing_processed.csv'
    output_directory = project_dir / 'data' / 'interim'
    preprocessor_file = project_dir / 'models' / 'preprocessor.pkl'
    
    # Argumentos que pasarías al script en la terminal:
    # python src/features/build_features.py data/processed/bike_sharing_processed.csv data/interim models/preprocessor.pkl
    try:
        main([str(input_file), str(output_directory), str(preprocessor_file)])
    except Exception as e:
        # Esto es solo un catch para la ejecución de prueba, si falla en main()
        logging.getLogger(__name__).error(f"Fallo al ejecutar build_features.py: {e}")