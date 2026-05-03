import xgboost as xgb

from models.base_model import BaseModel


class XGBoostModel(BaseModel):
    """Classification wrapper (legacy default)."""

    def __init__(self, params: dict | None = None):
        self.params = params or {"n_estimators": 100, "max_depth": 4}
        self.model = xgb.XGBClassifier(**self.params)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model.load_model(path)


class XGBoostRegressorModel(BaseModel):
    """Regression for continuous targets (e.g. open-to-close return)."""

    def __init__(self, params: dict | None = None):
        self.params = params or {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }
        self.model = xgb.XGBRegressor(**self.params)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: str) -> None:
        self.model.save_model(path)

    def load(self, path: str) -> None:
        self.model.load_model(path)