from pathlib import Path
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from data_preprocessing import load_data, split_features_target

MODEL_PATH = "models/xgboost_pipeline.pkl"
DATA_PATH = "data/raw/german_credit_data.csv"

def explain_model():
    pipeline = joblib.load(MODEL_PATH)
    
    df = load_data(DATA_PATH)
    X, y = split_features_target(df, target_col="Risk")
    
    _, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, stratify=y,random_state=42
    )
    
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    
    X_test_processed = preprocessor.transform(X_test)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_processed)
    
    Path("reports").mkdir(exist_ok=True)
    
    shap.summary_plot(shap_values, X_test_processed, show=False)
    plt.savefig("reports/shap_summary.png", bbox_inches="tight")
    plt.close()
    
    print("SHAP report saved to reports/shap_summary.png")
    
if __name__ == "__main__":
    explain_model() 
    
    
    