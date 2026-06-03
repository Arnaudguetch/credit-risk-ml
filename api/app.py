import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager


model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load("models/xgboost_pipeline.pkl")
    yield

app = FastAPI(
    title="Credit Risk Scoring API",
    lifespan=lifespan,
    root_path=""
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"message": "Credit Risk API is running"}


@app.post("/predict")
def predict(data: dict):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    
    df = pd.DataFrame([data])
    probability = model.predict_proba(df)[0][1]
    prediction = int(probability >= 0.5)
    
    return {
        "default_probability": float(probability),
        "prediction": prediction,
        "risk_label": "bad" if prediction == 1 else "good",
    }