import mlflow
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing


def main():
    MODEL_NAME = "CaliforniaHousingRegressor"
    MODEL_VERSION = 1

    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")

    X, y = fetch_california_housing(return_X_y=True, as_frame=True)

    # Échantillon pour aller vite
    X_small = X.sample(1000, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_small)

    # Global
    shap.summary_plot(shap_values, X_small, show=False)
    plt.tight_layout()
    plt.show()

    # Local
    i = 0
    exp = shap.Explanation(
        values=shap_values[i],
        base_values=explainer.expected_value,
        data=X_small.iloc[i].values,
        feature_names=X_small.columns
    )
    shap.plots.waterfall(exp, show=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
