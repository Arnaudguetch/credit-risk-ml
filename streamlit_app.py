import os
import requests
import pandas as pd
import streamlit as st

DATA_PATH = "data/raw/german_credit_data.csv"
TARGET_COL = "Risk"  
API_URL = os.getenv("API_URL", "http://localhost:8000")

MAINTENANCE_MODE = os.getenv("MAINTENANCE_MODE", "false").lower() == "true"

st.set_page_config(page_title="Credit Risk Scoring", layout="centered")


def maintenance_page():
    st.title("Application en maintenance")
    st.warning("Nous effectuns actuellement une mise à jour.")
    st.write("Merci de réessayer plus tard !!")
    
    if st.button("Réessayer"):
        st.rerun()
        
if MAINTENANCE_MODE:
    maintenance_page()
    st.stop()
    
    
def app():
    st.title("Credit Risk Scoring")
    st.write("Demo of client default risk prediction using FastAPI.")

    @st.cache_data
    def load_reference_data():
        df = pd.read_csv(DATA_PATH)

        if TARGET_COL not in df.columns:
            raise ValueError(
                f"Target column '{TARGET_COL}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        return df.drop(columns=[TARGET_COL])

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
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=user_input,
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json()

                st.subheader("Results")

                st.metric(
                    label="Probability of default",
                    value=f"{result['default_probability']:.2%}",
                )

                if result["prediction"] == 1:
                    st.error("Status of prediction: BAD RISK")
                else:
                    st.success("Status of prediction: GOOD RISK")

            else:
                st.error("Service temporairement indisponible.")
                st.write(f"API error: {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Impossible de contacter l'API FastAPI. "
                "Vérifie que l'API tourne bien sur http://localhost:8000"
            )
        except requests.exceptions.Timeout:
            st.warning("La requête a expiré. Veuillez réessayer.")

        except Exception as e:
            st.error(f"Unexpected error: {e}")
    
try:
    app()
    
except Exception:
    maintenance_page()
    
    
