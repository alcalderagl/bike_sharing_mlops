# 🚴 MLOps Pipeline: Predicción de Demanda de Bicicletas

Este proyecto implementa un *pipeline* de Machine Learning (ML) modularizado y reproducible para predecir la demanda horaria de bicicletas (columna `cnt`) utilizando el dataset **Bike Sharing Dataset**.

# Estructura del dataset
| Variable Name | Role     | Type         | Description                                                                                                           | Units | Missing Values |
|----------------|----------|--------------|-----------------------------------------------------------------------------------------------------------------------|--------|----------------|
| instant        | ID       | Integer      | record index                                                                                                          |        | no             |
| dteday         | Feature  | Date         | date                                                                                                                  |        | no             |
| season         | Feature  | Categorical  | 1:winter, 2:spring, 3:summer, 4:fall                                                                                  |        | no             |
| yr             | Feature  | Categorical  | year (0: 2011, 1: 2012)                                                                                               |        | no             |
| mnth           | Feature  | Categorical  | month (1 to 12)                                                                                                       |        | no             |
| hr             | Feature  | Categorical  | hour (0 to 23)                                                                                                        |        | no             |
| holiday        | Feature  | Binary       | weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)                               |        | no             |
| weekday        | Feature  | Categorical  | day of the week                                                                                                       |        | no             |
| workingday     | Feature  | Binary       | if day is neither weekend nor holiday is 1, otherwise is 0                                                            |        | no             |
| weathersit     | Feature  | Categorical  | 1: Clear, 2: Few clouds, 3: Partly cloudy, 4: Partly cloudy                                                                    |        | no             |
| temp           | Feature  | Continuous   | Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (hourly)   | C      | no             |
| atemp          | Feature  | Continuous   | Normalized feeling temperature in Celsius. Derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (hourly)         | C      | no             |
| hum            | Feature  | Continuous   | Normalized humidity. The values are divided by 100 (max)                                                              |        | no             |
| windspeed      | Feature  | Continuous   | Normalized wind speed. The values are divided by 67 (max)                                                             |        | no             |
| casual         | Other    | Integer      | count of casual users                                                                                                 |        | no             |
| registered     | Other    | Integer      | count of registered users                                                                                             |        | no             |
| cnt            | Target   | Integer      | count of total rental bikes including both casual and registered                                                      |        | no             |

--- 

El flujo de trabajo sigue las mejores prácticas de MLOps, utilizando **DVC (Data Version Control)** para la reproducibilidad de datos y **MLflow** para el seguimiento de experimentos.

---

## ⚙️ Tecnologías y Requisitos

| Herramienta | Propósito |
| :--- | :--- |
| **Python** | Lenguaje principal de desarrollo. |
| **XGBoost** | Algoritmo de machine learning principal. |
| **Scikit-learn** | Modelado, Pipelines y ColumnTransformer. |
| **DVC (Data Version Control)** | Versionado del pipeline completo y de los artefactos. |
| **MLflow** | Registro de experimentos, parámetros, métricas y modelos. |
| **FastAPI** | Framework para API REST de serving del modelo. |
| **Docker** | Containerización para despliegue en producción. |
| **pytest** | Framework de testing unitario e integración. |
| **click** | Modulación de los scripts de Python. |

---

## 🏗️ Estructura del Proyecto


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## 🧠 Lógica de Preprocesamiento

El *pipeline* implementa las siguientes transformaciones críticas derivadas del EDA y Preprocesamiento:

1.  **Limpieza:** Eliminación de *Target Leakage* (`casual`, `registered`) e inconsistencias de datos (ej. cadenas de texto) mediante coerción a tipo numérico.
2.  **Manejo de Outliers:** Aplicación de **Winsorización** a las *features* continuas (`temp`, `atemp`, `hum`, `windspeed`) a percentiles variables (`0.01` y `0.99`).
3.  **Transformación del Objetivo:** Aplicación de la transformación **Logarítmica Inversa (`np.log1p`)** a la variable `cnt` para reducir la asimetría.
4.  **Pipeline Scikit-learn:** Uso de `ColumnTransformer` para estandarizar *features* continuas y codificar categóricas, asegurando que las transformaciones se apliquen solo con los datos de entrenamiento (Punto 3).
5.  **División Temporal:** El conjunto de datos se divide por tiempo, utilizando el **20% de los datos más recientes** para el conjunto de prueba.

---

## 🏃 Reproducción del Proyecto (Punto 4: Reproducibilidad)

El proyecto está diseñado para ser completamente reproducible utilizando DVC y la línea de comandos.

### 1. Configuración del Entorno

Para comenzar, asegúrate de activar tu entorno virtual e instalar las dependencias necesarias:

# 1. Activación del entorno virtual
```
.venv\Scripts\activate      # Windows
```
# 2. Instalar dependencias (basado en requirements.txt)
```
pip install -r requirements.txt
```
### 2. Ejecución del Pipeline Completo
Asegúrese de que el archivo bike_sharing_modified.csv se encuentre en data/raw/. El pipeline completo de 5 etapas se ejecuta con el siguiente comando, que garantiza que todas las transformaciones se realicen en el orden correcto y que los artefactos sean versionados:
```
dvc repro
```

### 3. Secuencia de Ejecución de Etapas

El comando *dvc repro* ejecuta internamente la siguiente secuencia de scripts modulares:
<table style="width:100%; border-collapse: collapse; border: 1px solid #4B5263; background-color: #282C34; color: #DCDFE4;">
  <thead>
    <tr style="background-color: #3B4048;">
      <th style="border: 1px solid #4B5263; padding: 10px; text-align: left; color: #61AFEF;">Paso</th>
      <th style="border: 1px solid #4B5263; padding: 10px; text-align: left; color: #61AFEF;">Archivo</th>
      <th style="border: 1px solid #4B5263; padding: 10px; text-align: left; color: #61AFEF;">Salida Principal</th>
      <th style="border: 1px solid #4B5263; padding: 10px; text-align: left; color: #61AFEF;">Propósito</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid #4B5263; padding: 8px;">1. Limpieza</td>
      <td style="border: 1px solid #4B5263; padding: 8px;"><code style="background-color: #3B4048; color: #98C379; padding: 2px 4px; border-radius: 3px;">src/data/make_dataset.py</code></td>
      <td style="border: 1px solid #4B5263; padding: 8px;"><code style="background-color: #3B4048; color: #E5C07B; padding: 2px 4px; border-radius: 3px;">data/processed/</code></td>
      <td style="border: 1px solid #4B5263; padding: 8px;">Prepara los datos, aplica Winsorización y Log Transform.</td>
    </tr>
    <tr>
      <td style="border: 1px solid #4B5263; padding: 8px;">2. Features/Split</td>
      <td style="border: 1px solid #4B5263; padding: 8px;"><code style="background-color: #3B4048; color: #98C379; padding: 2px 4px; border-radius: 3px;">src/features/build_features.py</code></td>
      <td style="border: 1px solid #4B5263; padding: 8px;"><code style="background-color: #3B4048; color: #E5C07B; padding: 2px 4px; border-radius: 3px;">models/preprocessor.pkl</code></td>
      <td style="border: 1px solid #4B5263; padding: 8px;">Divide datos (por tiempo) y define el ColumnTransformer.</td>
    </tr>
    <tr>
      <td style="border: 1px solid #4B5263; padding: 8px;">3. Entrenamiento</td>
      <td style="border: 1px solid #4B5263; padding: 8px;"><code style="background-color: #3B4048; color: #98C379; padding: 2px 4px; border-radius: 3px;">src/models/train_model.py</code></td>
      <td style="border: 1px solid #4B5263; padding: 8px;"><code style="background-color: #3B4048; color: #E5C07B; padding: 2px 4px; border-radius: 3px;">models/final_pipeline.pkl</code></td>
      <td style="border: 1px solid #4B5263; padding: 8px;">Entrena el Pipeline completo y registra en MLflow.</td>
    </tr>
    <tr>
      <td style="border: 1px solid #4B5263; padding: 8px;">4. Predicción</td>
      <td style="border: 1px solid #4B5263; padding: 8px;"><code style="background-color: #3B4048; color: #98C379; padding: 2px 4px; border-radius: 3px;">src/models/predict_model.py</code></td>
      <td style="border: 1px solid #4B5263; padding: 8px;"><code style="background-color: #3B4048; color: #E5C07B; padding: 2px 4px; border-radius: 3px;">data/results/predictions_final.csv</code></td>
      <td style="border: 1px solid #4B5263; padding: 8px;">Genera predicciones en escala real (aplicando inversa logarítmica).</td>
    </tr>
    <tr>
      <td style="border: 1px solid #4B5263; padding: 8px;">5. Visualización</td>
      <td style="border: 1px solid #4B5263; padding: 8px;"><code style="background-color: #3B4048; color: #98C379; padding: 2px 4px; border-radius: 3px;">src/visualization/visualize.py</code></td>
      <td style="border: 1px solid #4B5263; padding: 8px;"><code style="background-color: #3B4048; color: #E5C07B; padding: 2px 4px; border-radius: 3px;">reports/figures/predictions_vs_real.png</code></td>
      <td style="border: 1px solid #4B5263; padding: 8px;">Genera el gráfico de evaluación final.</td>
    </tr>
  </tbody>
</table>


### 4. Seguimiento de Experimentos con MLflow

Para validar el seguimiento de experimentos y gestionar el modelo registrado (Punto 4: Gestión de Modelos):

```
mlflow ui
```
Abre el navegador en la dirección indicada para ver las métricas (ej. RMSE) y los parámetros del experimento.

### 5. Validación de Reproducibilidad entre Entornos

El proyecto incluye un script de validación de reproducibilidad (`tools/run_repro_two_envs.py`) que permite verificar que el pipeline produce resultados idénticos en diferentes entornos virtuales. Este script es especialmente útil para garantizar la reproducibilidad del proyecto en diferentes máquinas o configuraciones.

#### ¿Qué hace el script?

El script realiza las siguientes acciones:

1. **Crea dos entornos virtuales independientes** (`env_a` y `env_b`) en la carpeta `repro/`
2. **Instala las dependencias** desde `requirements-repro.txt` en cada entorno
3. **Ejecuta el pipeline completo** (`dvc repro`) en cada entorno de forma aislada
4. **Compara los artefactos generados** entre ambos entornos:
   - Modelos entrenados (`final_xgb_model.pkl`)
   - Predicciones (`predictions_final.csv`)
   - Datos procesados (`bike_sharing_processed.csv`)
5. **Calcula métricas de comparación**:
   - RMSE, MAE y R² para cada entorno
   - Prueba estadística de Kolmogorov-Smirnov (KS test) entre las predicciones
6. **Genera visualizaciones comparativas**:
   - Gráfico de diferencia absoluta entre predicciones
   - Scatter plot comparando predicciones de ambos entornos

#### Cómo ejecutarlo

Desde el directorio raíz del proyecto:

```
python tools/run_repro_two_envs.py
```

#### Salidas del script

El script genera los siguientes resultados:

- **Comparación binaria**: Verifica si los archivos son idénticos byte a byte
- **Hashes MD5**: Muestra los hashes MD5 de los modelos y predicciones para verificación rápida
- **Métricas de rendimiento**: Compara RMSE, MAE y R² entre ambos entornos
- **Prueba estadística**: Realiza un test KS para verificar si las distribuciones de predicciones son estadísticamente equivalentes
- **Visualizaciones**: Guarda gráficos en `repro/`:
  - `diff_absolute.png`: Muestra la diferencia absoluta entre predicciones
  - `scatter_a_vs_b.png`: Comparación visual de predicciones de ambos entornos

#### Interpretación de resultados

- **Archivos idénticos**: Si los modelos y predicciones son idénticos entre entornos, el pipeline es completamente reproducible
- **Métricas similares**: Si las métricas (RMSE, MAE, R²) son muy similares pero los archivos no son idénticos, puede indicar diferencias menores en la precisión numérica
- **Test KS**: Un p-value alto (> 0.05) sugiere que las distribuciones de predicciones son estadísticamente equivalentes

Este script es una herramienta valiosa para validar que el pipeline mantiene la reproducibilidad incluso cuando se ejecuta en entornos completamente aislados, lo cual es fundamental para proyectos de MLOps en producción.

---

## 🧪 Testing Framework

El proyecto incluye dos suites completas de pruebas para garantizar la calidad y confiabilidad tanto del pipeline MLOps como del servicio API.

### 1. Tests MLOps Pipeline

```
tests/
├── conftest.py              <- Configuración y fixtures de pytest
├── test_integration.py      <- Pruebas de integración end-to-end
├── test_unit_data.py        <- Pruebas unitarias de procesamiento de datos
├── test_unit_features.py    <- Pruebas unitarias de feature engineering
├── test_unit_metrics.py     <- Pruebas unitarias de métricas del modelo
└── test_unit_predict.py     <- Pruebas unitarias del pipeline de predicción
```

### 2. Tests API Service

```
src/tests/api/
├── test_health.py           <- Pruebas de endpoints de salud
└── test_predict.py          <- Pruebas del endpoint de predicción
```

### Cobertura de Pruebas

#### MLOps Pipeline Tests
| Módulo | Pruebas | Descripción |
|--------|---------|-------------|
| **Data Processing** | `test_unit_data.py` | Limpieza de datos, manejo de patrones inválidos, preparación de fechas, imputación |
| **Feature Engineering** | `test_unit_features.py` | Transformaciones cíclicas, cálculo de lags, ventanas móviles, validación de features finales |
| **Model Metrics** | `test_unit_metrics.py` | Cálculo de RMSE, R², validación de métricas perfectas, formato de salida |
| **Prediction Pipeline** | `test_unit_predict.py` | Constructor de features para predicción, validación de lags, salida no negativa |
| **Integration** | `test_integration.py` | Pipeline completo end-to-end, carga de modelo, formato de predicciones |

#### API Service Tests
| Módulo | Pruebas | Descripción |
|--------|---------|-------------|
| **Health Endpoints** | `test_health.py` | Verificación de `/health/live` y `/health/ready` |
| **Prediction Endpoint** | `test_predict.py` | Validación del endpoint `/v1/predict` con mocking |

### Ejecución de Tests

```bash
# Ejecutar todas las pruebas MLOps
pytest tests/

# Ejecutar todas las pruebas API
pytest src/tests/api/

# Ejecutar todas las pruebas del proyecto
pytest tests/ src/tests/

# Ejecutar pruebas con cobertura
pytest tests/ --cov=src

# Ejecutar pruebas específicas
pytest tests/test_unit_data.py -v
pytest src/tests/api/test_predict.py -v
```

---

## 🚀 API Service (FastAPI)

El modelo está expuesto a través de un servicio RESTful construido con FastAPI para serving en producción.

### Artefactos y Trazabilidad del Modelo

| Artefacto | Versión | Ruta Local |
| :--- | :--- | :--- |
| **Modelo XGBoost** | **1.0.0** | `models/final_xgb_model.pkl` |
| **Features Finales** | N/A | Definidas en `src/models/predict_model.py:FINAL_FEATURES` |
| **Rendimiento Base** | **RMSE Original:** 45.0 - 55.0 bicicletas | Registrado en MLflow |

### Ejecución Local del API

```bash
# Desde el directorio raíz del proyecto
uvicorn src.services.api.app.main:app --reload
```

### Endpoints Principales

<table style="width:100%; border-collapse: collapse; text-align: left;">
    <thead>
        <tr style="background-color: #f2f2f2;">
            <th style="border: 1px solid #ddd; padding: 8px;">Endpoint</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Método</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Descripción</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Tests</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;"><code>/v1/predict</code></td>
            <td style="border: 1px solid #ddd; padding: 8px;">POST</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Genera predicción de demanda de bicicletas</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><code>test_predict.py</code></td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;"><code>/health/live</code></td>
            <td style="border: 1px solid #ddd; padding: 8px;">GET</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Verifica si la API está funcionando</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><code>test_health.py</code></td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;"><code>/health/ready</code></td>
            <td style="border: 1px solid #ddd; padding: 8px;">GET</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Verifica si la API está lista (modelo cargado)</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><code>test_health.py</code></td>
        </tr>
    </tbody>
</table>

### Documentación Interactiva

- **Swagger UI**: http://127.0.0.1:8000/docs
- **Esquema OpenAPI**: http://127.0.0.1:8000/openapi.json

### Ejemplo de Request

```json
{
  "instances": [
    {
      "dteday": "2012-11-20", 
      "hr": 8, 
      "temp": 0.45, 
      "hum": 0.6, 
      "windspeed": 0.15, 
      "weathersit": 2, 
      "cnt_lag_1": 100.0, 
      "cnt_lag_24": 250.0, 
      "season": 3, 
      "yr": 1, 
      "mnth": 11, 
      "weekday": 2
    }
  ],
  "inverse_transform": true
}
```

### Testing del API

Los tests del API utilizan `TestClient` de FastAPI y mocking para validar:

- **Health endpoints**: Verifican respuestas correctas (status 200, formato JSON)
- **Prediction endpoint**: Usa mocking para simular predicciones y validar el formato de respuesta
- **Payload validation**: Asegura que el esquema de entrada sea correcto

---

## 🐳 Docker Deployment

El proyecto incluye containerización completa para despliegue en producción.

### Imagen Docker Publicada

La imagen del servicio está disponible en Docker Hub:

```bash
# Ejecutar el contenedor localmente
docker run -p 8000:8000 abyesses2023/ml-service:latest
```

### Construcción Local

Para construir la imagen localmente:

```bash
# Construir la imagen
docker build -t bike-sharing-api .

# Ejecutar el contenedor
docker run -p 8000:8000 bike-sharing-api
```

### Configuración del Dockerfile

El Dockerfile incluye:
- **Base Image**: Python 3.12 slim
- **Dependencias**: Instalación desde `requirements_api.txt`
- **Artefactos**: Copia del modelo entrenado y datos históricos
- **Exposición**: Puerto 8000 para FastAPI
- **Comando**: Uvicorn server con configuración de producción

### Variables de Entorno

| Variable | Valor por Defecto | Descripción |
|----------|-------------------|-------------|
| `MODEL_FILE` | `models/final_xgb_model.pkl` | Ruta del modelo entrenado |
| `HISTORICAL_FILE` | `data/interim/historical_for_prediction.csv` | Datos históricos para lags |
| `PORT` | `8000` | Puerto del servidor |

### Verificación del Servicio

Una vez ejecutado el contenedor:

```bash
# Verificar que el servicio está funcionando
curl http://localhost:8000/health/live

# Verificar que el modelo está cargado
curl http://localhost:8000/health/ready

# Acceder a la documentación
open http://localhost:8000/docs

# Ejecutar tests del API
pytest src/tests/api/ -v
```nternamente la siguiente secuencia de scripts modulares:
<table style="width:100%; border-collapse: collapse; border: 1px solid #4B5263; background-color: #282C34; color: #DCDFE4;">
  <thead>
    <tr style="background-color: #3B4048;">
      <th style="border: 1px solid #4B5263; padding: 10px; text-align: left; color: #61AFEF;">Paso</th>
      <th style="border: 1px solid #4B5263; padding: 10px; text-align: left; color: #61AFEF;">Archivo</th>
      <th style="border: 1px solid #4B5263; padding: 10px; text-align: left; color: #61AFEF;">Salida Principal</th>
      <th style="border: 1px solid #4B5263; padding: 10px; text-align: left; color: #61AFEF;">Propósito</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid #4B5263; padding: 8px;">1. Limpieza</td>
      <td style="border: 1px solid #4B5263; padding: 8px;"><code style="background-color: #3B4048; color: #98C379; padding: 2px 4px; border-radius: 3px;">src/data/make_dataset.py</code></td>
      <td style="border: 1px solid #4B5263; padding: 8px;"><code style="background-color: #3B4048; color: #E5C07B; padding: 2px 4px; border-radius: 3px;">data/processed/</code></td>
      <td style="border: 1px solid #4B5263; padding: 8px;">Prepara los datos, aplica Winsorización y Log Transform.</td>
    </tr>
    <tr>
      <td style="border: 1px solid #4B5263; padding: 8px;">2. Features/Split</td>
      <td style="border: 1px solid #4B5263; padding: 8px;"><code style="background-color: #3B4048; color: #98C379; padding: 2px 4px; border-radius: 3px;">src/features/build_features.py</code></td>
      <td style="border: 1px solid #4B5263; padding: 8px;"><code style="background-color: #3B4048; color: #E5C07B; padding: 2px 4px; border-radius: 3px;">models/preprocessor.pkl</code></td>
      <td style="border: 1px solid #4B5263; padding: 8px;">Divide datos (por tiempo) y define el ColumnTransformer.</td>
    </tr>
    <tr>
      <td style="border: 1px solid #4B5263; padding: 8px;">3. Entrenamiento</td>
      <td style="border: 1px solid #4B5263; padding: 8px;"><code style="background-color: #3B4048; color: #98C379; padding: 2px 4px; border-radius: 3px;">src/models/train_model.py</code></td>
      <td style="border: 1px solid #4B5263; padding: 8px;"><code style="background-color: #3B4048; color: #E5C07B; padding: 2px 4px; border-radius: 3px;">models/final_pipeline.pkl</code></td>
      <td style="border: 1px solid #4B5263; padding: 8px;">Entrena el Pipeline completo y registra en MLflow.</td>
    </tr>
    <tr>
      <td style="border: 1px solid #4B5263; padding: 8px;">4. Predicción</td>
      <td style="border: 1px solid #4B5263; padding: 8px;"><code style="background-color: #3B4048; color: #98C379; padding: 2px 4px; border-radius: 3px;">src/models/predict_model.py</code></td>
      <td style="border: 1px solid #4B5263; padding: 8px;"><code style="background-color: #3B4048; color: #E5C07B; padding: 2px 4px; border-radius: 3px;">data/results/predictions_final.csv</code></td>
      <td style="border: 1px solid #4B5263; padding: 8px;">Genera predicciones en escala real (aplicando inversa logarítmica).</td>
    </tr>
    <tr>
      <td style="border: 1px solid #4B5263; padding: 8px;">5. Visualización</td>
      <td style="border: 1px solid #4B5263; padding: 8px;"><code style="background-color: #3B4048; color: #98C379; padding: 2px 4px; border-radius: 3px;">src/visualization/visualize.py</code></td>
      <td style="border: 1px solid #4B5263; padding: 8px;"><code style="background-color: #3B4048; color: #E5C07B; padding: 2px 4px; border-radius: 3px;">reports/figures/predictions_vs_real.png</code></td>
      <td style="border: 1px solid #4B5263; padding: 8px;">Genera el gráfico de evaluación final.</td>
    </tr>
  </tbody>
</table>


### 4. Seguimiento de Experimentos con MLflow

Para validar el seguimiento de experimentos y gestionar el modelo registrado (Punto 4: Gestión de Modelos):

```
mlflow ui
```
Abre el navegador en la dirección indicada para ver las métricas (ej. RMSE) y los parámetros del experimento.

### 5. Validación de Reproducibilidad entre Entornos

El proyecto incluye un script de validación de reproducibilidad (`tools/run_repro_two_envs.py`) que permite verificar que el pipeline produce resultados idénticos en diferentes entornos virtuales. Este script es especialmente útil para garantizar la reproducibilidad del proyecto en diferentes máquinas o configuraciones.

#### ¿Qué hace el script?

El script realiza las siguientes acciones:

1. **Crea dos entornos virtuales independientes** (`env_a` y `env_b`) en la carpeta `repro/`
2. **Instala las dependencias** desde `requirements-repro.txt` en cada entorno
3. **Ejecuta el pipeline completo** (`dvc repro`) en cada entorno de forma aislada
4. **Compara los artefactos generados** entre ambos entornos:
   - Modelos entrenados (`final_xgb_model.pkl`)
   - Predicciones (`predictions_final.csv`)
   - Datos procesados (`bike_sharing_processed.csv`)
5. **Calcula métricas de comparación**:
   - RMSE, MAE y R² para cada entorno
   - Prueba estadística de Kolmogorov-Smirnov (KS test) entre las predicciones
6. **Genera visualizaciones comparativas**:
   - Gráfico de diferencia absoluta entre predicciones
   - Scatter plot comparando predicciones de ambos entornos

#### Cómo ejecutarlo

Desde el directorio raíz del proyecto:

```
python tools/run_repro_two_envs.py
```

#### Salidas del script

El script genera los siguientes resultados:

- **Comparación binaria**: Verifica si los archivos son idénticos byte a byte
- **Hashes MD5**: Muestra los hashes MD5 de los modelos y predicciones para verificación rápida
- **Métricas de rendimiento**: Compara RMSE, MAE y R² entre ambos entornos
- **Prueba estadística**: Realiza un test KS para verificar si las distribuciones de predicciones son estadísticamente equivalentes
- **Visualizaciones**: Guarda gráficos en `repro/`:
  - `diff_absolute.png`: Muestra la diferencia absoluta entre predicciones
  - `scatter_a_vs_b.png`: Comparación visual de predicciones de ambos entornos

#### Interpretación de resultados

- **Archivos idénticos**: Si los modelos y predicciones son idénticos entre entornos, el pipeline es completamente reproducible
- **Métricas similares**: Si las métricas (RMSE, MAE, R²) son muy similares pero los archivos no son idénticos, puede indicar diferencias menores en la precisión numérica
- **Test KS**: Un p-value alto (> 0.05) sugiere que las distribuciones de predicciones son estadísticamente equivalentes

Este script es una herramienta valiosa para validar que el pipeline mantiene la reproducibilidad incluso cuando se ejecuta en entornos completamente aislados, lo cual es fundamental para proyectos de MLOps en producción.