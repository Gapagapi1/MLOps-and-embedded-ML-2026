import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/old/predict")
async def predict():
    return {"y_pred": 2}


import joblib

model = joblib.load("regression.joblib")

@app.post("/predict")
async def predict(size: int, bedrooms: int, garden: int):
    return {"y_pred": model.predict([[size, bedrooms, garden]])[0]}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5212, reload=True)

