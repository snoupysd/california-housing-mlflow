from src.train import train_and_track


def test_train_and_track_runs():
    result = train_and_track(experiment_name="test_experiment")

    assert result.best_model_name in {"LinearRegression", "RandomForest", "GradientBoosting"}
    assert result.best_rmse > 0
    assert isinstance(result.best_run_id, str)
    assert len(result.best_run_id) > 0