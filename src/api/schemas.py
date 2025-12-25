from pydantic import BaseModel, Field
from typing import List

FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]

class PredictRequest(BaseModel):
    # Une seule observation
    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

class PredictResponse(BaseModel):
    prediction: float

class BatchPredictRequest(BaseModel):
    rows: List[PredictRequest]

class BatchPredictResponse(BaseModel):
    predictions: List[float]