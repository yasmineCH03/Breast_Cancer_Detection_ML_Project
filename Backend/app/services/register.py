import os
import json
from datetime import datetime

def update_manifest(model_path, scaler_path, metrics, status="PRODUCTION"):
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    manifest_path = os.path.join(data_dir, 'deployment_manifest.json')
    timestamp = datetime.now().isoformat()
    entry = {
        "model_name": "SGD-SVM (v2.4)",
        "version": "v2.4",
        "model_path": model_path,
        "scaler_path": scaler_path,
        "metrics": metrics,
        "status": status,
        "updated_at": timestamp
    }
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    else:
        manifest = {"models": [], "last_updated": timestamp}
    found = False
    for i, m in enumerate(manifest.get("models", [])):
        if m.get("model_name") == "SGD-SVM (v2.4)":
            manifest["models"][i] = entry
            found = True
            break
    if not found:
        manifest.setdefault("models", []).append(entry)
    manifest["last_updated"] = timestamp
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return entry
