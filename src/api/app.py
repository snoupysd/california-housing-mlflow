from fastapi import FastAPI
from api.schemas import PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse
from api.model import ModelService

app = FastAPI(title="California Housing Predictor", version="1.0")

model_service = ModelService()

@app.on_event("startup")
def startup_event():
    model_service.load()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    pred = model_service.predict_one(payload.model_dump())
    return PredictResponse(prediction=pred)

@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest):
    rows = [r.model_dump() for r in payload.rows]
    preds = model_service.predict_batch(rows)
    return BatchPredictResponse(predictions=preds)
