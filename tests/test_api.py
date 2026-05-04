from fastapi.testclient import TestClient
from api.app import app



def test_health():
    with TestClient(app) as client:
        
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    
    
def test_predict():
    sample_input = {
        "Age": 35,
        "Sex": "male",
        "Job": 2,
        "Housing": "own",
        "Saving accounts": "little",
        "Checking account": "little",
        "Credit amount": 5000,
        "Duration": 24,
        "Purpose": "car"
    }
    
    with TestClient(app) as client:
        
        response = client.post("/predict", json=sample_input)
    
    assert response.status_code == 200
    
    data = response.json()
    
    assert "default_probability" in data
    assert "prediction" in data 
    assert "risk_label" in data