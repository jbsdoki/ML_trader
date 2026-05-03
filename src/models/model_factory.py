# model_factory.py
from models.xgboost import XGBoostModel, XGBoostRegressorModel

MODEL_REGISTRY = {
    "xgboost": XGBoostModel,
    "xgboost_regressor": XGBoostRegressorModel,
}


def get_model(name: str, params: dict | None = None):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name](params or {})