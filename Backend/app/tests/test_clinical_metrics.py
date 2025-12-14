import os
import sys
from statistics import mean
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def _r2(y_true, y_pred):
    yt = [float(v) for v in y_true]
    yp = [float(v) for v in y_pred]
    m = mean(yt)
    ss_res = sum((yt[i] - yp[i]) ** 2 for i in range(len(yt)))
    ss_tot = sum((yt[i] - m) ** 2 for i in range(len(yt)))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1 - (ss_res / ss_tot)

def test_aggressiveness_r2():
    seq = [
        {"neoplasm_histologic_grade": 1, "tumor_stage": 1, "lymph_nodes_examined_positive": 5, "nottingham_prognostic_index": 2.0},
        {"neoplasm_histologic_grade": 2, "tumor_stage": 2, "lymph_nodes_examined_positive": 5, "nottingham_prognostic_index": 4.0},
        {"neoplasm_histologic_grade": 3, "tumor_stage": 3, "lymph_nodes_examined_positive": 5, "nottingham_prognostic_index": 6.0},
        {"neoplasm_histologic_grade": 2, "tumor_stage": 3, "lymph_nodes_examined_positive": 5, "nottingham_prognostic_index": 5.0},
        {"neoplasm_histologic_grade": 3, "tumor_stage": 4, "lymph_nodes_examined_positive": 5, "nottingham_prognostic_index": 7.0},
        {"neoplasm_histologic_grade": 1, "tumor_stage": 2, "lymph_nodes_examined_positive": 5, "nottingham_prognostic_index": 3.0},
        {"neoplasm_histologic_grade": 2, "tumor_stage": 4, "lymph_nodes_examined_positive": 5, "nottingham_prognostic_index": 6.5},
        {"neoplasm_histologic_grade": 3, "tumor_stage": 1, "lymph_nodes_examined_positive": 5, "nottingham_prognostic_index": 2.5},
    ]
    cache = {"max_nodes": 1.0, "min_npi": float("inf"), "max_npi": float("-inf")}
    expected = []
    returned = []
    for item in seq:
        g = float(item["neoplasm_histologic_grade"])
        s = float(item["tumor_stage"])
        nodes = float(item["lymph_nodes_examined_positive"])
        npi = float(item["nottingham_prognostic_index"])
        cache["max_nodes"] = max(cache.get("max_nodes", 1.0), nodes if nodes > 0 else cache.get("max_nodes", 1.0))
        cache["min_npi"] = min(cache.get("min_npi", float("inf")), npi)
        cache["max_npi"] = max(cache.get("max_npi", float("-inf")), npi)
        max_nodes = max(1.0, cache["max_nodes"])
        min_npi = cache["min_npi"] if cache["min_npi"] != float("inf") else 0.0
        max_npi = cache["max_npi"] if cache["max_npi"] != float("-inf") else max(min_npi + 1.0, npi)
        grade_norm = min(g / 3.0, 1.0)
        stage_norm = min(s / 4.0, 1.0)
        nodes_norm = min(nodes / max_nodes, 1.0)
        npi_norm = 0.0 if max_npi == min_npi else (npi - min_npi) / (max_npi - min_npi)
        score = grade_norm * 3.0 + stage_norm * 3.0 + nodes_norm * 2.0 + npi_norm * 2.0
        expected.append(float(score))
        r = client.post("/api/v1/clinical/aggressiveness", json=item)
        assert r.status_code == 200
        returned.append(float(r.json()["score"]))
    r2 = _r2(expected, returned)
    assert r2 > 0.99

def test_growth_rate_r2():
    seq = [
        {"tumor_size": 10.0, "age_at_diagnosis": 45.0, "neoplasm_histologic_grade": 1},
        {"tumor_size": 20.0, "age_at_diagnosis": 50.0, "neoplasm_histologic_grade": 2},
        {"tumor_size": 30.0, "age_at_diagnosis": 60.0, "neoplasm_histologic_grade": 3},
        {"tumor_size": 15.0, "age_at_diagnosis": 42.0, "neoplasm_histologic_grade": 2},
        {"tumor_size": 5.0, "age_at_diagnosis": 41.0, "neoplasm_histologic_grade": 1},
        {"tumor_size": 40.0, "age_at_diagnosis": 80.0, "neoplasm_histologic_grade": 3},
    ]
    expected = []
    returned = []
    for item in seq:
        size = float(item["tumor_size"])
        age = float(item["age_at_diagnosis"])
        grade = float(item["neoplasm_histologic_grade"])
        years_since_onset = max(age - 40.0, 1.0)
        rate = (size / years_since_onset) * (grade / 2.0)
        rate = float(min(rate, 50.0))
        expected.append(rate)
        r = client.post("/api/v1/clinical/growth_rate", json=item)
        assert r.status_code == 200
        returned.append(float(r.json()["rate"]))
    r2 = _r2(expected, returned)
    assert r2 > 0.99

