from fastapi.testclient import TestClient
from api.app import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    sample_input = {
        "unnamed:_0": 0,
        "age": 35,
        "sex": "male",
        "job": 2,
        "housing": "own",
        "saving_accounts": "little",
        "checking_account": "little",
        "credit_amount": 5000,
        "duration": 24,
        "purpose": "car"
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=sample_input)
        
        print(response.status_code)
        print(response.json()) 

    assert response.status_code == 200

    data = response.json()
    assert "default_probability" in data
    assert "prediction" in data
    assert "risk_label" in data