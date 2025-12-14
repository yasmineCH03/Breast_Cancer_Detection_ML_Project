import asyncio
import time
import os
import csv
import random
import httpx
try:
    import pytest
    pytestmark = pytest.mark.skip(reason="requires running server")
except Exception:
    pass

BASE = os.environ.get("API_BASE", "http://localhost:8000")

def assert_keys(d, keys):
    missing = [k for k in keys if k not in d]
    assert not missing, f"Missing keys: {missing}"

async def call(method, path, **kwargs):
    url = BASE + path
    async with httpx.AsyncClient(timeout=30) as client:
        t0 = time.perf_counter()
        r = await client.request(method, url, **kwargs)
        dt = (time.perf_counter() - t0) * 1000
        return r.status_code, r.headers, (await r.aread()), dt

async def test_health():
    code, headers, body, dt = await call("GET", "/api/health")
    assert code == 200, f"health status {code}"
    print("health ok", f"{dt:.1f} ms")

async def test_metabric_info():
    code, headers, body, dt = await call("GET", "/api/v1/metabric/model/info")
    assert code == 200, f"info status {code} body={body.decode()}"
    import json
    j = json.loads(body.decode())
    assert_keys(j, ["model_path","scaler_path","feature_count","features","metadata"])
    print("info ok", f"{dt:.1f} ms", f"features={len(j['features'])}")
    return j["features"]

async def test_metabric_predict(features):
    payload = {"features": {k: 0 for k in features}}
    code, headers, body, dt = await call("POST", "/api/v1/metabric/predict", json=payload)
    assert code == 200, f"predict status {code} body={body.decode()}"
    import json
    j = json.loads(body.decode())
    assert_keys(j, ["aggressiveness_score","growth_rate","evolution_6m_raw","evolution_6m_class"])
    print("predict ok", f"{dt:.1f} ms", f"class={j['evolution_6m_class']}")

async def test_metabric_evaluate():
    code, headers, body, dt = await call("GET", "/api/v1/metabric/evaluate")
    assert code == 200, f"evaluate status {code} body={body.decode()}"
    import json
    j = json.loads(body.decode())
    assert_keys(j, ["r2_aggressiveness","mae_aggressiveness","mse_aggressiveness",
                    "r2_growth_rate","mae_growth_rate","mse_growth_rate","accuracy_evolution_6m"])
    print("evaluate ok", f"{dt:.1f} ms", f"acc={j['accuracy_evolution_6m']:.3f}")

async def test_metabric_batch(features):
    path = "tmp_batch.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(features)
        for _ in range(5):
            w.writerow([random.random() for _ in features])
    files = {"file": ("tmp_batch.csv", open(path, "rb"), "text/csv")}
    async with httpx.AsyncClient(timeout=30) as client:
        t0 = time.perf_counter()
        r = await client.post(BASE + "/api/v1/metabric/batch", files=files)
        dt = (time.perf_counter() - t0) * 1000
        assert r.status_code == 200, f"batch status {r.status_code} body={r.text}"
        import json
        j = r.json()
        assert_keys(j, ["status","rows","count"])
        assert j["status"] == "success"
        print("batch ok", f"{dt:.1f} ms", f"rows={len(j['rows'])}")
    try:
        os.remove(path)
    except Exception:
        pass

async def main():
    await test_health()
    feats = await test_metabric_info()
    # run predict, evaluate, batch concurrently
    await asyncio.gather(
        test_metabric_predict(feats),
        test_metabric_evaluate(),
        test_metabric_batch(feats),
    )

if __name__ == "__main__":
    asyncio.run(main())
