from src.services.api.app.config import settings

def get_version() -> str:
    return settings.MODEL_VERSION