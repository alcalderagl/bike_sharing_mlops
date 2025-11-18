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
| **Scikit-learn** | Modelado, Pipelines y ColumnTransformer (Punto 3). |
| **DVC (Data Version Control)** | Versionado del pipeline completo y de los artefactos (Punto 4). |
| **MLflow** | Registro de experimentos, parámetros, métricas y modelos (Punto 4). |
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
Asegúrese de que el archivo bike_sharing_modified.csv se encuentre en data/raw/. El pipeline completo de 5 etapas se ejecuta con el siguiente comando, que garantiza que todas las transformaciones se realicen en el orden correcto y q