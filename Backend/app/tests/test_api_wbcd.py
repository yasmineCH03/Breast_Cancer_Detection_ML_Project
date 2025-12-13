import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_wbcd_eda_summary():
    r = client.get("/api/v1/wbcd/eda/summary")
    assert r.status_code == 200
    j = r.json()
    assert "shape" in j and "rows" in j["shape"]

def test_wbcd_correlations():
    r = client.get("/api/v1/wbcd/eda/correlations?top_n=10")
    assert r.status_code == 200
    j = r.json()
    assert "correlations" in j
    assert len(j["correlations"]) <= 10

def test_wbcd_features():
    r = client.get("/api/v1/wbcd/features")
    assert r.status_code == 200
    j = r.json()
    assert "features" in j
    assert len(j["features"]) == 30
