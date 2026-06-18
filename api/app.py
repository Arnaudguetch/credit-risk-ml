import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

class CreditRequest(BaseModel):
    Age: int
    Sex: str
    Job: int
    Housing: str

    Saving_accounts: str = Field(alias="Saving accounts")
    Checking_account: str = Field(alias="Checking account")
    Credit_amount: float = Field(alias="Credit amount")

    Duration: int
    Purpose: str

    model_config = {
        "populate_by_name": True
    }
    
MODEL_PATH = os.getenv("MODEL_PATH", "models/xgboost_pipeline.pkl")

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model loading failed: {e}")
        model = None

    yield


app = FastAPI(
    title="Credit Risk Scoring API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready"}


@app.post("/predict")
def predict(payload: CreditRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame([payload.model_dump(by_alias=True)]) 
        proba = model.predict_proba(df)[0][1]

        return {
            "default_probability": float(proba),
            "prediction": int(proba >= 0.5),
            "risk_label": "bad" if proba >= 0.5 else "good"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))