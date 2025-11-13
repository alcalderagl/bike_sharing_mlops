from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MODEL_PATH: str = "models/final_pipeline.pkl"
    MODEL_VERSION: str = "0.1.0"
    # ej. variables para Cloud Run
    PORT: int = 8080

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()