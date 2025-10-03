import os
import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
INITIAL_MODEL_URI = os.getenv("INITIAL_MODEL_URI", "models:/iris-model/1")

MODEL: Optional[mlflow.pyfunc.PyFuncModel] = None
CURRENT_MODEL_URI: Optional[str] = None

def _load_model(uri: str) -> mlflow.pyfunc.PyFuncModel:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.pyfunc.load_model(uri)

app = FastAPI(title="MLflow FastAPI Service")

try:
    MODEL = _load_model(INITIAL_MODEL_URI)
    CURRENT_MODEL_URI = INITIAL_MODEL_URI
    print(f"[startup] Loaded model from {CURRENT_MODEL_URI}")
except Exception as e:
    MODEL = None
    CURRENT_MODEL_URI = None
    print(f"[startup] WARNING: failed to load initial model '{INITIAL_MODEL_URI}': {e}")

@app.post("/predict")
async def predict(request: Request):
    global MODEL
    if MODEL is None:
        return JSONResponse(status_code=503, content={"error": "model_not_loaded"})

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid_json"})

    if not isinstance(payload, dict) or "data" not in payload:
        return JSONResponse(status_code=400, content={"error": "missing_field:data"})

    try:
        x = pd.DataFrame(payload["data"])
        preds = MODEL.predict(x)
        return {"model_uri": CURRENT_MODEL_URI, "predictions": preds.tolist() if not isinstance(preds, list) else preds}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "prediction_failed", "details": str(e)})

@app.post("/update-model")
async def update_model(request: Request):
    global MODEL, CURRENT_MODEL_URI

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid_json"})

    if not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={"error": "invalid_body"})

    model_uri: Optional[str] = payload.get("model_uri")
    if not model_uri:
        return JSONResponse(status_code=400, content={"error": "missing_model_uri"})

    try:
        new_model = _load_model(model_uri)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "load_failed", "details": str(e)})

    MODEL = new_model
    CURRENT_MODEL_URI = model_uri
    return {"model_uri": CURRENT_MODEL_URI}

@app.get("/health")
async def health():
    return "ok"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

