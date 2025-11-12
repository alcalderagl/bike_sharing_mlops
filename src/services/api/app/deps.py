from fastapi import Depends
from .inference import load_model

def get_model():
    return load_model()