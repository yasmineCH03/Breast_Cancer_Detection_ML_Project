from fastapi.testclient import TestClient
import sys
import os

# Add the api folder to system path so we can import 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "online", "system": "OncoGuard AI"}

def test_prediction_flow():
    # 1. Create a dummy patient (using mean values)
    # This ensures the API structure handles the JSON correctly
    fake_patient = {
        "features": {
            "radius_mean": 14.0, "texture_mean": 20.0, "perimeter_mean": 90.0, "area_mean": 600.0,
            "smoothness_mean": 0.1, "compactness_mean": 0.1, "concavity_mean": 0.1, "concave points_mean": 0.05,
            "symmetry_mean": 0.2, "fractal_dimension_mean": 0.06,
            "radius_se": 0.4, "texture_se": 1.0, "perimeter_se": 3.0, "area_se": 40.0,
            "smoothness_se": 0.006, "compactness_se": 0.02, "concavity_se": 0.03, "concave points_se": 0.01,
            "symmetry_se": 0.02, "fractal_dimension_se": 0.004,
            "radius_worst": 15.0, "texture_worst": 25.0, "perimeter_worst": 100.0, "area_worst": 700.0,
            "smoothness_worst": 0.14, "compactness_worst": 0.3, "concavity_worst": 0.4, "concave points_worst": 0.2,
            "symmetry_worst": 0.3, "fractal_dimension_worst": 0.09
        }
    }

    # 2. Send POST request
    response = client.post("/predict", json=fake_patient)
    
    # 3. Assertions (Checks)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "status" in data
    assert "action_plan" in data
    # Check that risk score is a valid probability (0 to 1)
    assert 0 <= data["risk_score"] <= 1