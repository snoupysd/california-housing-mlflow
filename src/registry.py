import mlflow


def register_best_model(run_id: str, model_name: str = "CaliforniaHousingRegressor"):
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"Registered model: {result.name} v{result.version}")


if __name__ == "__main__":
    RUN_ID = "b9b4ee4aea144154a9b96b26b25f5029"
    register_best_model(RUN_ID)