import os
import requests
import pandas as pd
import streamlit as st

DATA_PATH = "data/raw/german_credit_data.csv"
TARGET_COL = "Risk"
API_URL = os.getenv("API_URL", "http://credit-risk-service")

MAINTENANCE_MODE = os.getenv("MAINTENANCE_MODE", "false").lower() == "true"

st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="💳",
    layout="centered"
)

st.markdown("""
<style>
    .main {
        background: linear-gradient(
            135deg,
            #eef2ff 0%,
            #f8fafc 40%,
            #e0f2fe 100%
        );
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(
            135deg,
            #eef2ff 0%,
            #f8fafc 40%,
            #e0f2fe 100%
        );
    }
    
    [data-testid="stHeader"] {
        background: rgba(255,255,255,0);
    }

    .title {
        text-align: center;
        font-size: 46px;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 8px;
        letter-spacing: -1px;
    }

    .subtitle {
        text-align: center;
        color: #5f6f89;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.3);
        margin-bottom: 25px;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 35px rgba(0,0,0,0.12);
    }

    .success-card {
        background: #e9f8ef;
        padding: 25px;
        border-radius: 18px;
        border-left: 7px solid #22c55e;
        color: #166534;
    }

    .danger-card {
        background: #fdecec;
        padding: 25px;
        border-radius: 18px;
        border-left: 7px solid #ef4444;
        color: #991b1b;
    }

    .maintenance {
        text-align: center;
        background: white;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }
    
    [data-testid="metric-container"] {
        background: white;
        border-radius: 16px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        transition: 0.2s ease;
    }
    
    [data-testid="stHeader"] {
        background: rgba(255,255,255,0.6);
        backdrop-filter: blur(10px);
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        color: white;
        border-radius: 14px;
        padding: 14px;
        font-weight: 700;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 6px 18px rgba(37,99,235,0.25);
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #1d4ed8, #3b82f6);
        box-shadow: 0 10px 24px rgba(37,99,235,0.35);
    }
    
    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        border: 1px solid #dbe4ff !important;
    }

    input {
        border-radius: 12px !important;
    }

    .stSlider > div[data-baseweb="slider"] {
        padding-top: 10px;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(8px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .card, .success-card, .danger-card {
        animation: fadeIn 0.4s ease-in-out;
    }
    
    div[data-baseweb="base-input"] {
        border-radius: 12px !important;
    }
    
</style>
""", unsafe_allow_html=True)


def maintenance_page():
    st.markdown("""
    <div class="maintenance">
        <h1>🚧 Application en maintenance</h1>
        <p>Nous effectuons actuellement une mise à jour.</p>
        <p>Merci de réessayer plus tard.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Réessayer"):
        st.rerun()


if MAINTENANCE_MODE:
    maintenance_page()
    st.stop()


@st.cache_data
def load_reference_data():
    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    return df.drop(columns=[TARGET_COL])


def app():
    st.markdown('<div class="title">💳 Credit Risk Scoring</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Machine Learning application for customer default risk prediction</div>',
        unsafe_allow_html=True
    )

    X_ref = load_reference_data()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👤 Client Information")

    user_input = {}

    for col in X_ref.columns:
        if pd.api.types.is_numeric_dtype(X_ref[col]):
            user_input[col] = st.number_input(
                label=col,
                min_value=float(X_ref[col].min()),
                max_value=float(X_ref[col].max()),
                value=float(X_ref[col].mean()),
            )
        else:
            values = sorted(X_ref[col].dropna().unique().tolist())
            user_input[col] = st.selectbox(
                label=col,
                options=values,
            )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🏦 Business Risk Strategy")

    threshold = st.slider(
        "Risk approval threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
        help="If the default probability is above this threshold, the loan is rejected."
    )

    loan_amount = st.number_input(
        "Loan amount (€)",
        min_value=1000,
        max_value=1000000,
        value=5000,
        step=1000,
    )

    interest_rate = st.slider(
        "Interest rate (%)",
        min_value=1.0,
        max_value=25.0,
        value=8.0,
        step=0.5,
    )

    lgd = st.slider(
        "Loss Given Default - LGD",
        min_value=0.10,
        max_value=1.00,
        value=0.45,
        step=0.05,
        help="Estimated percentage loss if the client defaults."
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Prédire le risque"):
        try:
            with st.spinner("Analyse du profil client en cours..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json=user_input,
                    timeout=10,
                )

            if response.status_code == 200:
                result = response.json()
                probability = result["default_probability"]

                decision = "APPROVE" if probability < threshold else "REJECT"

                expected_loss = probability * loan_amount * lgd
                expected_return = loan_amount * (interest_rate / 100)
                net_expected_value = expected_return - expected_loss

                st.subheader("📊 Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label="Probability of default",
                        value=f"{probability:.2%}",
                    )

                with col2:
                    st.metric(
                        label="Decision threshold",
                        value=f"{threshold:.0%}",
                    )

                if decision == "REJECT":
                    st.markdown(f"""
                    <div class="danger-card">
                        <h3>🔴 LOAN REJECTED</h3>
                        <p>This client has a high probability of default compared to the selected risk threshold.</p>
                        <p><strong>Default probability:</strong> {probability:.2%}</p>
                        <p><strong>Risk threshold:</strong> {threshold:.0%}</p>
                        <p><strong>Loan amount:</strong> €{loan_amount:,.2f}</p>
                        <p><strong>Expected loss:</strong> €{expected_loss:,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="success-card">
                        <h3>🟢 LOAN APPROVED</h3>
                        <p>This client has an acceptable probability of default according to the selected risk threshold.</p>
                        <p><strong>Default probability:</strong> {probability:.2%}</p>
                        <p><strong>Risk threshold:</strong> {threshold:.0%}</p>
                        <p><strong>Loan amount:</strong> €{loan_amount:,.2f}</p>
                        <p><strong>Expected return:</strong> €{expected_return:,.2f}</p>
                        <p><strong>Expected loss:</strong> €{expected_loss:,.2f}</p>
                        <p><strong>Net expected value:</strong> €{net_expected_value:,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("📈 Business Impact")

                metric1, metric2, metric3 = st.columns(3)

                with metric1:
                    st.metric(
                        label="Expected return",
                        value=f"€{expected_return:,.0f}",
                    )

                with metric2:
                    st.metric(
                        label="Expected loss",
                        value=f"€{expected_loss:,.0f}",
                    )

                with metric3:
                    st.metric(
                        label="Net expected value",
                        value=f"€{net_expected_value:,.0f}",
                    )

                st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.error("Service temporairement indisponible.")
                st.write(f"API error: {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Impossible de contacter l'API FastAPI. "
                "Vérifie que le service API est bien actif."
            )

        except requests.exceptions.Timeout:
            st.warning("La requête a expiré. Veuillez réessayer.")

        except Exception as e:
            st.error(f"Unexpected error: {e}")


try:
    app()

except Exception:
    maintenance_page()