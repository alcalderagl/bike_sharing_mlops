# Dockerfile

# --- STAGE 1: Build Stage (Usamos una imagen base de Python) ---
FROM python:3.12-slim-bookworm AS base

# Definir variables de entorno para FastAPI y Uvicorn
ENV PYTHONUNBUFFERED=1 \
    APP_HOME=/app \
    # Usamos la variable de entorno MODEL_PATH definida en config.py
    MODEL_FILE=models/final_xgb_model.pkl \
    HISTORICAL_FILE=data/interim/historical_for_prediction.csv \
    # Puerto donde correrá Uvicorn dentro del contenedor
    PORT=8000 

# Crear y establecer el directorio de trabajo
WORKDIR ${APP_HOME}

# Copiar solo el archivo de dependencias y la configuración de MLflow
COPY requirements_api.txt .

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements_api.txt

# --- STAGE 2: Final Image (Copia los archivos y configura el inicio) ---

# Copiar todo el código de la aplicación (src, services, etc.)
# Nota: Si usas una estructura diferente, ajusta este COPY
COPY src/ ${APP_HOME}/src/
# COPY services/ ${APP_HOME}/services/

# Copiar el modelo entrenado y los datos históricos (CRÍTICO)
# Asume que estos archivos están en la raíz del proyecto.
COPY models/final_xgb_model.pkl ${APP_HOME}/${MODEL_FILE}
#COPY data/interim/historical_for_prediction.csv ${APP_HOME}/${HISTORICAL_FILE}

# Exponer el puerto
EXPOSE ${PORT}

# Comando de inicio

CMD ["uvicorn", "src.services.api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# NOTA: Asegúrate de que config.py tiene MODEL_PATH="models/model.pkl" 
# (o el nombre que uses) y PORT=8000 (o el que uses aquí).
