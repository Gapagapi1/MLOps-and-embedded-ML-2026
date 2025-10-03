import os
import json
import random
from typing import Optional

import numpy as np
import pandas as pd

import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
INITIAL_MODEL_URI = os.getenv("INITIAL_MODEL_URI", "models:/iris-model/1")
CANARY_P = float(os.getenv("CANARY_P", "1.0"))


CURRENT_MODEL: Optional[mlflow.pyfunc.PyFuncModel] = None
NEXT_MODEL: Optional[mlflow.pyfunc.PyFuncModel] = None
CURRENT_MODEL_URI: Optional[str] = None
NEXT_MODEL_URI: Optional[str] = None


def _load_model(uri: str) -> mlflow.pyfunc.PyFuncModel:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.pyfunc.load_model(uri)


app = FastAPI(title="MLflow FastAPI Service")


try:
    CURRENT_MODEL = _load_model(INITIAL_MODEL_URI)
    NEXT_MODEL = CURRENT_MODEL
    CURRENT_MODEL_URI = INITIAL_MODEL_URI
    NEXT_MODEL_URI = INITIAL_MODEL_URI
    print(f"[startup] Loaded current and next from {INITIAL_MODEL_URI}; p={CANARY_P}")
except Exception as e:
    CURRENT_MODEL = None
    NEXT_MODEL = None
    CURRENT_MODEL_URI = None
    NEXT_MODEL_URI = None
    print(f"[startup] Failed to load initial model '{INITIAL_MODEL_URI}': {e}")


@app.post("/predict")
async def predict(request: Request):
    global CURRENT_MODEL, NEXT_MODEL, CURRENT_MODEL_URI, NEXT_MODEL_URI, CANARY_P
    if CURRENT_MODEL is None or NEXT_MODEL is None:
        return JSONResponse(status_code=503, content={"error": "model_not_loaded"})

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid_json"})

    if not isinstance(payload, dict) or "data" not in payload:
        return JSONResponse(status_code=400, content={"error": "missing_field:data"})

    try:
        X = pd.DataFrame(payload["data"])
        use_current = random.random() < max(0.0, min(1.0, CANARY_P))
        model = CURRENT_MODEL if use_current else NEXT_MODEL
        model_uri = CURRENT_MODEL_URI if use_current else NEXT_MODEL_URI
        which = "current" if use_current else "next"

        preds = model.predict(X)
        return {"used": which, "model_uri": model_uri, "prob_current": CANARY_P, "predictions": preds.tolist() if not isinstance(preds, list) else preds}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "prediction_failed", "details": str(e)})


@app.post("/update-model")
async def update_model(request: Request):
    global NEXT_MODEL, NEXT_MODEL_URI

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid_json"})

    if not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={"error": "invalid_body"})

    model_uri = payload.get("model_uri")
    if not model_uri:
        return JSONResponse(status_code=400, content={"error": "missing_model_uri"})

    try:
        new_model = _load_model(model_uri)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "load_failed", "details": str(e)})

    NEXT_MODEL = new_model
    NEXT_MODEL_URI = model_uri
    return {"current_model_uri": CURRENT_MODEL_URI, "next_model_uri": NEXT_MODEL_URI}


@app.post("/accept-next-model")
async def accept_next_model():
    global CURRENT_MODEL, NEXT_MODEL, CURRENT_MODEL_URI, NEXT_MODEL_URI

    if NEXT_MODEL is None or NEXT_MODEL_URI is None:
        return JSONResponse(status_code=409, content={"error": "no_next_model"})

    CURRENT_MODEL = NEXT_MODEL
    CURRENT_MODEL_URI = NEXT_MODEL_URI
    return {"current_model_uri": CURRENT_MODEL_URI, "next_model_uri": NEXT_MODEL_URI}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "current_model_loaded": CURRENT_MODEL is not None,
        "next_model_loaded": NEXT_MODEL is not None,
        "current_model_uri": CURRENT_MODEL_URI,
        "next_model_uri": NEXT_MODEL_URI,
        "p": CANARY_P,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
