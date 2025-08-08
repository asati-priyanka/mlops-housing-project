import os
import math
import httpx
import pytest

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

def _api_available() -> bool:
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

@pytest.mark.skipif(not _api_available(), reason="API not running on API_BASE_URL")
def test_health_ok():
    r = httpx.get(f"{API_BASE}/health", timeout=5)
    assert r.status_code == 200
    assert r.json().get("status") == "ok"

@pytest.mark.skipif(not _api_available(), reason="API not running on API_BASE_URL")
def test_predict_returns_number():
    payload = {
        "records": [{
            "MedInc": 8.3252,
            "HouseAge": 41.0,
            "AveRooms": 6.9841,
            "AveBedrms": 1.0238,
            "Population": 322.0,
            "AveOccup": 2.5556,
            "Latitude": 37.88,
            "Longitude": -122.23
        }]
    }
    r = httpx.post(f"{API_BASE}/predict", json=payload, timeout=10)
    assert r.status_code == 200, r.text
    data = r.json()
    preds = data.get("predictions")
    assert isinstance(preds, list) and len(preds) == 1
    assert isinstance(preds[0], (int, float))
    assert math.isfinite(preds[0])
