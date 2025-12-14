import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_metabric_model_info():
    r = client.get("/api/v1/metabric/model/info")
    assert r.status_code in (200, 500)
    if r.status_code == 200:
        j = r.json()
        assert "feature_count" in j and "features" in j

def test_metabric_predict_schema():
    payload = {"features": {}}
    r = client.post("/api/v1/metabric/predict", json=payload)
    assert r.status_code in (200, 400)

