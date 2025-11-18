from fastapi import Depends
from src.services.api.app.inference import load_model

def get_model():
    return load_model()