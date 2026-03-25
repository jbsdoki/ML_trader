import xgboost as xgb
from models.base_model import BaseModel

class XGBoostModel(BaseModel):

    def __init__(self, params: dict = None):
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