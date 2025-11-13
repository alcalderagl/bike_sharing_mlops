from fastapi import FastAPI
# from .version import get_version
# import services.api.app.routes.health #import health
from services.api.app.routes import health, predict

tags_metadata = [
    {"name": "health", "description": "liveness & readiness"},
    {"name": "predict", "description": "bike sharing model prediction"}
]

app = FastAPI(
    title="Bike sharing API",
    description="On demand bike sharing inference service.",
    version='get_version()',
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(predict.router, prefix="/v1", tags=["predict"])