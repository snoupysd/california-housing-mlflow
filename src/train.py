from __future__ import annotations

import numpy as np
import mlflow
import mlflow.sklearn

from dataclasses import dataclass
from typing import Dict, Tuple

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


@dataclass(frozen=True)
class TrainResult:
    best_model_name: str
    best_rmse: float
    best_run_id: str


def eval_regression(y_true, y_pred):
    rmse=float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae=float(mean_absolute_error(y_true, y_pred))
    r2=float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def get_data(test_size: float=0.2, random_state:int=42):
    X, y=fetch_california_housing(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def get_models(random_state: int=42):
    lr=Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    rf=RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1
    )

    gbr=GradientBoostingRegressor(
        random_state=random_state
    )

    return {
        "LinearRegression": lr,
        "RandomForest": rf,
        "GradientBoosting": gbr
    }


def train_and_track(
    experiment_name: str="california_housing_regression",
    tracking_uri: str | None=None,
    random_state: int=42
) -> TrainResult:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    X_train, X_test, y_train, y_test=get_data(random_state=random_state)
    models=get_models(random_state=random_state)

    best_name=""
    best_rmse=float("inf")
    best_run_id=""

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            # Log params (minimal + utiles)
            mlflow.log_param("model_name", name)
            mlflow.log_param("random_state", random_state)

            if name == "RandomForest":
                mlflow.log_params({
                    "n_estimators": model.n_estimators,
                    "max_depth": model.max_depth,
                    "min_samples_split": model.min_samples_split,
                    "min_samples_leaf": model.min_samples_leaf,
                    "bootstrap": model.bootstrap
                })
            elif name == "GradientBoosting":
                mlflow.log_params({
                    "n_estimators": model.n_estimators,
                    "learning_rate": model.learning_rate,
                    "max_depth": model.max_depth,
                    "subsample": model.subsample
                })

            # Train + eval
            model.fit(X_train, y_train)
            preds=model.predict(X_test)
            metrics=eval_regression(y_test, preds)

            mlflow.log_metrics(metrics)

            # Log model artifact
            mlflow.sklearn.log_model(model, artifact_path="model")

            # Keep best (RMSE)
            if metrics["rmse"] < best_rmse:
                best_rmse=metrics["rmse"]
                best_name=name
                best_run_id=run.info.run_id

    return TrainResult(best_model_name=best_name, best_rmse=best_rmse, best_run_id=best_run_id)


def main():
    result=train_and_track()
    print(f"Best model: {result.best_model_name} | RMSE={result.best_rmse:.4f} | run_id={result.best_run_id}")


if __name__ == "__main__":
    main()