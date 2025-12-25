import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from typing import Any

MODEL_NAME_DEFAULT = "CaliforniaHousingRegressor"
MODEL_VERSION_DEFAULT = "1"

class ModelService:
    def __init__(self):
        self.model: Any = None

    def load(self):
        # Option A: modèle embarqué (pour Docker)
        model_path = os.getenv("MODEL_PATH", "model.joblib")
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            return

        # Option B: fallback MLflow registry
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        model_name = os.getenv("MODEL_NAME", MODEL_NAME_DEFAULT)
        model_version = os.getenv("MODEL_VERSION", MODEL_VERSION_DEFAULT)
        self.model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")

    def predict_one(self, row: dict) -> float:
        df = pd.DataFrame([row])
        return float(self.model.predict(df)[0])

    def predict_batch(self, rows: list[dict]) -> list[float]:
        df = pd.DataFrame(rows)
        preds = self.model.predict(df)
        return [float(x) for x in preds]
