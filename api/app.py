import joblib
import pandas as pd
from fastapi import FastAPI
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load("models/xgboost_pipeline.pkl")
    yield

app = FastAPI(
    title="Credit Risk Scoring API",
    lifespan=lifespan
)


@app.get("/")
def root():
    return {"message": "Credit Risk API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    model = app.state.model 
    
    df = pd.DataFrame([data])
    probability = model.predict_proba(df)[0][1]
    prediction = int(probability >= 0.5)
    
    return {
        "default_probability": float(probability),
        "prediction": prediction,
        "risk_label": "bad" if prediction == 1 else "good",
    }