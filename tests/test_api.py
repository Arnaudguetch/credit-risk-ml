def test_predict():
    sample_input = {
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

    assert response.status_code == 200

    data = response.json()

    assert "default_probability" in data
    assert "prediction" in data
    assert "risk_label" in data