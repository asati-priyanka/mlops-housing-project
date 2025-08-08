from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import pandas as pd

# NOTE: When you open a new terminal, remember:
# .\.venv\Scripts\Activate.ps1
# $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

# Feature order must match training
FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]

# Pydantic v2 models
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
_model: mlflow.pyfunc.PyFuncModel | None = None

def get_model() -> mlflow.pyfunc.PyFuncModel:
    global _model
    if _model is None:
        try:
            _model = mlflow.pyfunc.load_model(MODEL_URI)
        except Exception as e:
            # Usually happens if MLflow server/registry isn't reachable
            raise HTTPException(status_code=503, detail=f"Model load failed: {e}")
    return _model

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.records:
        raise HTTPException(status_code=400, detail="Empty records")

    model = get_model()

    # Build a DataFrame with exact schema columns (required by MLflow signature)
    rows = [{f: getattr(r, f) for f in FEATURES} for r in req.records]
    df = pd.DataFrame(rows, columns=FEATURES)

    try:
        preds = model.predict(df)
    except Exception as e:
        # Surface input/schema issues nicely
        raise HTTPException(status_code=422, detail=f"Prediction failed: {e}")

    return PredictResponse(predictions=[float(p) for p in preds])
