from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, 
    RocCurveDisplay,
    classification_report, 
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from data_preprocessing import load_data, split_features_target

def get_feature_names(pipeline):
    preprocessor = pipeline.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out()


def plot_feature_importance(pipeline, output_path: str):
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(pipeline)
    
    if not hasattr(model, "feature_importances_"):
        print("This model does not expose feature_importances_. Skipping.")
        return 
        
    importances = model.feature_importances_
    
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)
    
    importance_df.to_csv("reports/feature_importance.csv", index=False)
    
    top_features = importance_df.head(20)
    
    plt.figure(figsize=(10,6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 20 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()         
    


def evaluate_saved_model(
    model_path: str | Path,
    data_path: str | Path,
    target_col: str = "Risk",
    random_state: int = 42,
) -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
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
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred)
    
    print("ROC-AUC:", round(roc_auc, 4))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    
    plt.figure()
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve - AUC={roc_auc:.4f}")
    plt.savefig(reports_dir / "roc_curve.png", bbox_inches="tight")
    plt.close()
    
    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Maxtrix")
    plt.savefig(reports_dir / "confusion_matrix.png", bbox_inches="tight") 
    plt.close()
    
    plot_feature_importance(
        pipeline,
        output_path=reports_dir / "feature_importance.png",   
    )
    
    print("Saved evaluation plots to reports/")

if __name__ == "__main__":
    evaluate_saved_model(
        model_path="models/xgboost_pipeline.pkl",
        data_path="data/raw/german_credit_data.csv",
        target_col="Risk",
    )