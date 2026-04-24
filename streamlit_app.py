import joblib
import pandas as pd
import streamlit as st 

MODEL_PATH = "models/xgboost_pipeline.pkl"
DATA_PATH = "data/raw/german_credit_data.csv"
TARGET_COL = "Risk"

st.set_page_config(page_title="Credit Risk Scoring", layout="centered")

st.title("Credit Risk Scoring")
st.write("Demo of client default risk prediction.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_reference_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    return X

model = load_model()
X_ref = load_reference_data()

st.subheader("Client Information")

user_input = {}

for col in X_ref.columns:
    if pd.api.types.is_numeric_dtype(X_ref[col]):
        min_val = float(X_ref[col].min())
        max_val = float(X_ref[col].max())
        mean_val = float(X_ref[col].mean())
        
        user_input[col] = st.number_input(
            label=col,
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
        )
    else:
        values = sorted(X_ref[col].dropna().unique().tolist())
        
        user_input[col] = st.selectbox(
            label=col,
            options=values,
        )
if st.button("Prédire le risque"):
    input_df = pd.DataFrame([user_input])
    
    probability = model.predict_proba(input_df)[0][1]
    prediction = int(probability >= 0.5)
    
    st.subheader("Results")
    
    st.metric(
        label="Probability of default",
        value=f"{probability:.2%}",
    )
    
    if prediction == 1:
        st.error("Status of prediction : BAD RISK")
    else:
        st.success("Status of prediction : GOOD RISK")

