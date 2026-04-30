from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    
    
def test_predit():
    sample_input = {
        "age": 35,
        "credit_amount": 5000,
        "duration": 24
    }
    
    response = client.post("/predict", json=sample_input)
    
    assert response.status_code == 200
    
    data = response.json()
    
    assert "default_probability" in data
    assert "prediction" in data 