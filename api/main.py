from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import pandas as pd  # <-- add this

# Feature order must match training:
FEATURES = [
    "MedInc","HouseAge","AveRooms","AveBedrms",
    "Population","AveOccup","Latitude","Longitude"
]

# Pydantic v2 model
class HousingFeatures(BaseModel):
    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

class PredictRequest(BaseModel):
    records: List[HousingFeatures]

class PredictResponse(BaseModel):
    predictions: List[float]

app = FastAPI(title="Housing Predictor", version="0.1.0")

# Load the best model by alias
MODEL_URI = "models:/housing_best_model@best"
_model = None

def get_model():
    global _model
    if _model is None:
        _model = mlflow.pyfunc.load_model(MODEL_URI)
    return _model

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.records:
        raise HTTPException(status_code=400, detail="Empty records")
    model = get_model()

    # Build a DataFrame with the exact schema columns
    rows = [{f: getattr(r, f) for f in FEATURES} for r in req.records]
    df = pd.DataFrame(rows, columns=FEATURES)

    preds = model.predict(df)
    return PredictResponse(predictions=[float(p) for p in preds])
