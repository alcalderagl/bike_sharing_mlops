from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MODEL_PATH: str = "models/final_pipeline.pkl"
    MODEL_VERSION: str = "1.0.0" # Versión final del modelo
    PORT: int = 8000 

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
