from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import pandas as pd
import time
import hashlib
import json
import os
import sqlite3

# NOTE: When you open a new terminal, remember:
# .\.venv\Scripts\Activate.ps1
# $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

LOG_DIR = os.getenv("LOG_DIR", "logs")
DB_PATH = os.path.join(LOG_DIR, "api.db")
os.makedirs(LOG_DIR, exist_ok=True)

# Feature order must match training
FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
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


# --- sqlite helpers ---
def _db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
    CREATE TABLE IF NOT EXISTS requests(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts TEXT NOT NULL,
      features_json TEXT NOT NULL,
      features_hash TEXT NOT NULL,
      prediction REAL,
      latency_ms REAL,
      error TEXT
    )
    """
    )
    return conn


DB = _db()


def _hash_features(dct: dict) -> str:
    payload = json.dumps(dct, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.records:
        raise HTTPException(status_code=400, detail="Empty records")

    start = time.time()
    error_msg = None
    prediction_value = None

    model = get_model()

    # Build a DataFrame with exact schema columns (required by MLflow signature)
    rows = [{f: getattr(r, f) for f in FEATURES} for r in req.records]
    df = pd.DataFrame(rows, columns=FEATURES)

    try:
        preds = model.predict(df)
        prediction_value = float(preds[0])
    except Exception as e:
        error_msg = str(e)
        # Surface input/schema issues nicely
        raise HTTPException(status_code=422, detail=f"Prediction failed: {e}")
    finally:
        latency_ms = (time.time() - start) * 1000.0
        feat = rows[0]
        try:
            DB.execute(
                "INSERT INTO requests(ts,features_json,features_hash,prediction,latency_ms,error) VALUES(?,?,?,?,?,?)",
                (
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    json.dumps(feat, separators=(",", ":")),
                    _hash_features(feat),
                    prediction_value,
                    latency_ms,
                    error_msg,
                ),
            )
            DB.commit()
        except Exception as db_e:
            raise HTTPException(status_code=500, detail=f"sqlite-insert-failed: {db_e}")

    return PredictResponse(predictions=[prediction_value])


@app.get("/metrics")
def metrics():
    cur = DB.execute(
        "SELECT COUNT(*), SUM(latency_ms), SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) FROM requests"
    )
    total_requests, total_latency, total_errors = cur.fetchone()

    if total_requests == 0:
        avg_latency = 0.0
    else:
        avg_latency = total_latency / total_requests

    return {
        "total_requests": total_requests,
        "total_errors": total_errors,
        "avg_latency_ms": round(avg_latency, 3),
        "sqlite_db": DB_PATH,
    }


@app.on_event("shutdown")
def shutdown_event():
    DB.close()
