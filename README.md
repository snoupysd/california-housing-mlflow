Parfait. Voici un **README.md complet, propre, et “niveau GitHub”**, prêt à être copié-collé tel quel à la racine de ton repo.
Il couvre **tous les livrables 1 → 6** et explique **comment tout relancer**.

---

```markdown
# California Housing – End-to-End MLOps Project

This project implements a complete **MLOps pipeline** for a regression task based on the **California Housing** dataset.  
It covers the full lifecycle of a machine learning model: data exploration, training, experiment tracking, model registry, API serving, containerization, CI/CD, monitoring, and drift detection.

---

## Project Overview

**Objective**  
Predict the median house value in California districts using classical regression models, while applying industry-standard MLOps best practices.

**Key components**
- Exploratory Data Analysis (EDA)
- Model training and evaluation
- Experiment tracking and model registry with MLflow
- REST API with FastAPI
- Dockerized deployment
- CI/CD with GitHub Actions
- Monitoring and data drift detection with Evidently
- Streamlit client for API testing

---

## Project Structure

```

california-housing-mlflow/
├── src/
│   ├── train.py                 # Training + MLflow tracking
│   ├── registry.py              # Model registration in MLflow
│   ├── monitoring_drift.py      # Drift detection with Evidently
│   ├── api/
│   │   └── app.py               # FastAPI application
│   └── streamlit_app/
│       └── app.py               # Streamlit client
├── tests/                        # Unit tests (pytest)
├── reports/
│   ├── eda/                      # Exploratory Data Analysis
│   └── data_drift_report_*.html  # Evidently drift reports
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── poetry.lock
├── .gitignore
└── README.md

````

---

## Installation

### Prerequisites
- Python **3.11**
- Poetry
- Docker (optional, for containerized API)

### Install dependencies
```bash
poetry install
````

---

## Exploratory Data Analysis (EDA)

The EDA analyzes:

* Feature distributions
* Correlations
* Potential outliers
* Target variable behavior

The report is available in:

```
reports/eda/
```

---

## Model Training & Experiment Tracking (MLflow)

### Train models

This script trains and compares:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor

Each experiment logs:

* Parameters
* Metrics (RMSE, MAE, R²)
* Model artifacts

```bash
poetry run python src/train.py
```

### Launch MLflow UI

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then open:

```
http://127.0.0.1:5000
```

---

## Model Registry

The best-performing model is registered in the MLflow Model Registry.

```bash
poetry run python src/registry.py
```

This creates a versioned model named:

```
CaliforniaHousingRegressor
```

---

## API Serving (FastAPI)

### Run API locally

```bash
poetry run uvicorn api.app:app --app-dir src --host 127.0.0.1 --port 8000
```

Swagger documentation:

```
http://127.0.0.1:8000/docs
```

---

## Dockerized API

### Build image

```bash
docker build -t housing-api .
```

### Run container

```bash
docker run -p 8000:8000 housing-api
```

The API will be available at:

```
http://127.0.0.1:8000/docs
```

---

## Streamlit Client

A Streamlit interface allows testing the API interactively.

```bash
poetry run streamlit run src/streamlit_app/app.py
```

The UI sends requests to the FastAPI endpoint and displays predictions.

---

## CI/CD with GitHub Actions

The project includes a CI pipeline using **GitHub Actions**:

* Triggered on each push to `main`
* Runs unit tests using pytest
* Ensures code quality before deployment

Workflow configuration:

```
.github/workflows/ci.yml
```

---

## Model Serving with MLflow

The registered model can be served directly using MLflow:

```bash
poetry run mlflow models serve \
  -m "models:/CaliforniaHousingRegressor/1" \
  -p 5001 \
  --no-conda
```

Endpoint:

```
POST http://127.0.0.1:5001/invocations
```

---

## Monitoring & Data Drift Detection

Production data is simulated and compared with training data using **Evidently**.

### Generate drift report

```bash
poetry run python src/monitoring_drift.py
```

### Output

* HTML report generated in:

```
reports/data_drift_report_*.html
```

The report highlights:

* Feature-wise drift
* Global data drift detection
* Statistical tests

---

## Retraining Strategy (Drift Handling)

If significant drift is detected:

* Trigger conditional retraining
* Compare new model metrics in MLflow
* Promote the new model version in the registry if performance improves

Possible strategies:

* Time-based retraining
* Drift-based retraining
* Sliding window training
* Weighted recent data

---




## Technologies Used

* Python 3.11
* Scikit-learn
* MLflow
* FastAPI
* Docker
* GitHub Actions
* Streamlit
* Evidently

---

## Author

Project developed as part of an MLOps coursework / applied machine learning project.


