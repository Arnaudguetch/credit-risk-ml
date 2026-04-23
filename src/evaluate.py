from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split

from data_preprocessing import load_data, split_features_target

def evaluate_saved_model(
    model_path: str | Path,
    data_path: str | Path,
    target_col: str = "Risk",
    random_state: int = 42,
) -> None:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    pipeline = joblib.load(model_path)
    
    df = load_data(data_path)
    X, y = split_features_target(df, target_col=target_col)
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state
    )
    
    y_pred = pipeline.predict(X_test)
    
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure()
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
    plt.title("ROC Curve")
    plt.savefig(reports_dir / "roc_curve.png", bbox_inches="tight")
    plt.close()
    
    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Maxtrix")
    plt.savefig(reports_dir / "confusion_matrix.png", bbox_inches="tight") 
    plt.close()
    
    print("Saved evaluation plots to reports/")

if __name__ == "__main__":
    evaluate_saved_model(
        model_path="models/xgboost_pipeline.pkl",
        data_path="data/raw/german_credit_data.csv",
        target_col="Risk",
    )