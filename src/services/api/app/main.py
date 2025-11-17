from fastapi import FastAPI
from src.services.api.app.routes import health, predict
from src.services.api.app.version import get_version # Importación de la versión

tags_metadata = [
    {"name": "health", "description": "liveness & readiness"},
    {"name": "predict", "description": "bike sharing model prediction"}
]

app = FastAPI(
    title="Bike sharing API",
    description="On demand bike sharing inference service.",
    version=get_version(), # Usar la función para obtener la versión
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(predict.router, prefix="/v1", tags=["predict"])