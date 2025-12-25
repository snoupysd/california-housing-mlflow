FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY model.joblib ./model.joblib

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
