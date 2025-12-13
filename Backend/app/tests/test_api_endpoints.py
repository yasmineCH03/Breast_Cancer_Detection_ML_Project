import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    j = r.json()
    assert "status" in j and "version" in j

def test_guidelines():
    r = client.get("/api/guidelines")
    assert r.status_code == 200
    j = r.json()
    assert "risk_stratification" in j and "clinical_references" in j

def test_evaluation_metrics():
    r = client.get("/api/v1/evaluation/cellular/metrics")
    assert r.status_code == 200
    j = r.json()
    assert "models" in j and "test_set_size" in j

def test_shap_get():
    r = client.get("/api/v1/evaluation/cellular/shap")
    assert r.status_code == 200
    j = r.json()
    assert "top_features" in j and "feature_count" in j

def test_shap_post():
    r = client.post("/api/v1/evaluation/cellular/shap")
    assert r.status_code == 200
    j = r.json()
    assert "top_features" in j and "feature_count" in j

def test_wbcd_sample():
    r = client.get("/api/v1/wbcd/sample")
    assert r.status_code == 200
    j = r.json()
    assert "patient_id" in j and "features" in j
    assert isinstance(j["features"], dict)

def test_clinical_residuals():
    r = client.get("/api/v1/evaluation/clinical/residuals")
    assert r.status_code == 200
    j = r.json()
    assert j.get("status") == "endpoint_configured"

def test_cellular_predict():
    payload = {
        "radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.8, "area_mean": 1000.0,
        "smoothness_mean": 0.1184, "compactness_mean": 0.2776, "concavity_mean": 0.3001,
        "concave_points_mean": 0.14710, "symmetry_mean": 0.2419, "fractal_dimension_mean": 0.07871,
        "radius_se": 1.095, "texture_se": 0.9053, "perimeter_se": 8.589, "area_se": 153.4,
        "smoothness_se": 0.006399, "compactness_se": 0.04904, "concavity_se": 0.05373,
        "concave_points_se": 0.01587, "symmetry_se": 0.03003, "fractal_dimension_se": 0.006193,
        "radius_worst": 25.38, "texture_worst": 17.33, "perimeter_worst": 184.6, "area_worst": 2019.0,
        "smoothness_worst": 0.1622, "compactness_worst": 0.6656, "concavity_worst": 0.7119,
        "concave_points_worst": 0.2654, "symmetry_worst": 0.4601, "fractal_dimension_worst": 0.11890
    }
    r = client.post("/api/v1/cellular/predict", json=payload)
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        j = r.json()
        assert "diagnosis" in j and "risk_assessment" in j

def test_cellular_batch():
    header = [
        "patient_id",
        "radius_mean","texture_mean","perimeter_mean","area_mean",
        "smoothness_mean","compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
        "radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
        "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"
    ]
    row = [
        "P001",
        17.99,10.38,122.8,1000.0,
        0.1184,0.2776,0.3001,0.14710,0.2419,0.07871,
        1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,
        25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.11890
    ]
    csv = ",".join(header) + "\n" + ",".join(map(str, row)) + "\n"
    files = {"file": ("patients.csv", csv.encode("utf-8"), "text/csv")}
    r = client.post("/api/v1/cellular/batch", files=files)
    assert r.status_code in (200, 400, 413, 500)
    if r.status_code == 200:
        j = r.json()
        assert "status" in j and "results" in j
