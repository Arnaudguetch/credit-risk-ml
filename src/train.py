from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import data_preprocessing
from data_preprocessing import build_preprocessor, load_data, split_features_target


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def train_model(
    data_path: str | Path,
    target_col: str = "Risk",
    model_name: str = "xgboost",
    random_state: int = 42,
) -> None:
    print("RUNNING THIS FILE:", __file__)
    print("DATA_PREPROCESSING FILE:", data_preprocessing.__file__)

    df = load_data(data_path)
    print("Columns:", df.columns.tolist())

    X, y = split_features_target(df, target_col=target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    preprocessor: ColumnTransformer = build_preprocessor(X_train)

    if model_name == "logreg":
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
    elif model_name == "xgboost":
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
        )
    else:
        raise ValueError("model_name must be either 'logreg' or 'xgboost'")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    mlflow.set_experiment("credit-risk-scoring")

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", random_state)

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_train shape:", y_train.shape)
        print("y_test shape:", y_test.shape)
        print("y_pred shape:", y_pred.shape)
        print("y_prob shape:", y_prob.shape)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        Path("models").mkdir(parents=True, exist_ok=True)
        model_path = Path("models") / f"{model_name}_pipeline.pkl"
        joblib.dump(pipeline, model_path)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
        )

        print(f"Model saved to: {model_path}")
        print("Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    train_model(
        data_path="data/raw/german_credit_data.csv",
        target_col="Risk",
        model_name="xgboost",
    )