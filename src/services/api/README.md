# 🚲 MLOps: Predicción de Demanda de Bicicletas

Este proyecto implementa un pipeline completo de Machine Learning (MLOps) para predecir la demanda horaria de bicicletas en un sistema de Bike Sharing, utilizando **XGBoost** como modelo principal y **FastAPI** para el *serving* en producción.

---

## 1. ⚙️ Pipeline del Proyecto

El flujo de trabajo MLOps consta de las siguientes etapas:

| Etapa | Descripción | Módulos Clave |
| :--- | :--- | :--- |
| **Ingesta/Limpieza** | Carga y limpieza inicial de datos brutos. | `src/data/make_dataset.py` |
| **Feature Engineering** | Creación de *features* cíclicas, interacciones, *lags* y *rolling means*. | `src/features/build_features.py` |
| **Entrenamiento** | Optimización de hiperparámetros (Grid Search) y entrenamiento del modelo **XGBoost**. | `src/models/train_model.py` |
| **Registro (MLflow)** | Registro de métricas, parámetros y el modelo final en MLflow. | `src/models/train_model.py` |
| **Serving (API)** | Exposición del modelo mediante un servicio **FastAPI** listo para producción. | `src/services/api/app/` |
| **Monitoreo** | Detección de degradación de rendimiento y *drift* de datos. | `src/monitoring/` |

---

## 2. 📦 Artefactos y Trazabilidad del Modelo

Esta sección registra la ruta y versión del modelo final utilizado por el servicio API, asegurando la trazabilidad.

| Artefacto | Versión | Ruta Local |
| :--- | :--- | :--- |
| **Modelo XGBoost** | **1.0.0** (Definido en `src/services/api/app/config.py`) | `models/final_xgb_model.pkl` |
| **Features Finales** | N/A | Definidas en `src/models/predict_model.py:FINAL_FEATURES` |
| **Rendimiento Base** | **RMSE Original:** 45.0 - 55.0 bicicletas (referencia de validación) | Registrado en MLflow |

---

## 3. 🚀 Servicio de Predicción (FastAPI)

El modelo está expuesto a través de un servicio RESTful.

### A. Ejecución Local

Para iniciar el servicio (asumiendo que estás en el directorio raíz `bike_sharing_mlops`):

```bash
# Se requiere tener los archivos __init__.py en services/api/app/
uvicorn src.services.api.app.main:app --reload
```
### B. Endpoint Principal

<table style="width:100%; border-collapse: collapse; text-align: left;">
    <thead>
        <tr style="background-color: #f2f2f2;">
            <th style="border: 1px solid #ddd; padding: 8px;">Endpoint</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Método</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Descripción</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Esquema</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;"><code>/v1/predict</code></td>
            <td style="border: 1px solid #ddd; padding: 8px;">POST</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Genera una predicción de demanda de bicicletas (conteo en escala original) para una lista de horas.</td>
            <td style="border: 1px solid #ddd; padding: 8px;"><code>PredictionRequest</code></td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;"><code>/health/live</code></td>
            <td style="border: 1px solid #ddd; padding: 8px;">GET</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Verifica si la API está funcionando.</td>
            <td style="border: 1px solid #ddd; padding: 8px;">N/A</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;"><code>/health/ready</code></td>
            <td style="border: 1px solid #ddd; padding: 8px;">GET</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Verifica si la API está lista (modelo cargado).</td>
            <td style="border: 1px solid #ddd; padding: 8px;">N/A</td>
        </tr>
    </tbody>
</table>

### C. Documentación y Esquema (OpenAPI)
La validación de entrada/salida está definida con Pydantic (schemas.py), lo que genera automáticamente la documentación:

Swagger UI (Interactiva): http://127.0.0.1:8000/docs

Esquema OpenAPI (JSON): http://127.0.0.1:8000/openapi.json

### D. Ejemplo de Request
El payload de la petición POST /v1/predict debe seguir el esquema PredictionRequest definido en schemas.py, que requiere los valores raw y los valores de lag precalculados por el cliente (o sistema upstream).
```
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
      // Otras features RAW requeridas
    }
  ],
  "inverse_transform": true
}
```

---

## 🐳 Docker Image & Versioning

Repositorio en Docker Hub:  
**`isaiambrocio/bike-ml-service`**

### 🔖 Tags disponibles

| Tag | Estado | Descripción |
|------|----------|-------------|
| **`0.1.0`** | ✅ Estable | Primera versión productiva del servicio FastAPI usando `final_xgb_model.pkl`. |
| **`latest`** | 🟡 Alias | Actualmente apunta a `0.1.0`. Puede cambiar en nuevas versiones. |

> El versionado sigue el estándar **SemVer**: `MAJOR.MINOR.PATCH`

---

### 🚀 Cómo ejecutar el contenedor

```bash
docker pull isaiambrocio/bike-ml-service:0.1.0

docker run -d \
  --name bike-ml-api \
  -p 8000:8000 \
  isaiambrocio/bike-ml-service:0.1.0
| URL                                  | Función         |
| ------------------------------------ | --------------- |
| `http://localhost:8000/docs`         | Swagger UI      |
| `http://localhost:8000/health/live`  | Liveness probe  |
| `http://localhost:8000/health/ready` | Readiness probe |

# Construir imagen con nuevo tag
docker build -t isaiambrocio/bike-ml-service:<NEW_VERSION> .

# (Opcional) actualizar el tag latest
docker tag isaiambrocio/bike-ml-service:<NEW_VERSION> isaiambrocio/bike-ml-service:latest

# Subir al registry
docker push isaiambrocio/bike-ml-service:<NEW_VERSION>
docker push isaiambrocio/bike-ml-service:latest

docker build -t isaiambrocio/bike-ml-service:0.2.0 .
docker tag isaiambrocio/bike-ml-service:0.2.0 isaiambrocio/bike-ml-service:latest
docker push isaiambrocio/bike-ml-service:0.2.0
docker push isaiambrocio/bike-ml-service:latest
