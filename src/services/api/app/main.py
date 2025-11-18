from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_500_INTERNAL_SERVER_ERROR,
)
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

# Errores de validación de Pydantic
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Error de validación en la petición.",
            "details": exc.errors(),
        },
    )


# Errores genéricos (bug, excepción sin capturar, etc.)
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Ocurrió un error interno en el servidor.",
            "message": str(exc),
        },
    )