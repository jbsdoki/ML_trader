# model_factory.py
from models.xgboost_model import XGBoostModel

MODEL_REGISTRY = {
    "xgboost": XGBoostModel,
}

def get_model(name: str, params: dict = None):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name](params or {})