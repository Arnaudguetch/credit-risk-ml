from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder 


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load dataset from csv file"""
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    return pd.read_csv(csv_path)

def split_features_target(
    df: pd.DataFrame,
    target_col= "target",
    ) -> Tuple[pd.DataFrame, pd.Series]:   
    """Split dataframe into features X and target y"""
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    
    y = y.map({
        "good": 0,
        "bad": 1
    })
    return X,y

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """"
    Build preprocessing pipeline:
    - numeric : median imputation
    - categorical : most frequent imputation + one-hot encoding
    """
    
    numerica_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()
    
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False),),
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerica_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor

def save_object(obj, output_path: str | Path) -> None:
    """"Save python object with joblib."""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, output_path)
    
    
if __name__ == "__main__":
    raw_path = Path("data/raw/german_credit_data.csv")
    df = load_data(raw_path)
    X,y = split_features_target(df, target_col="Risk")
    preprocessor = build_preprocessor(X)
    save_object(preprocessor, "models/preprocessor.pkl")
    print("Preprocessor saved to models/preprocessor.pkl")
    
    
    
    