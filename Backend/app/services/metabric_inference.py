import os
import joblib
import numpy as np
from typing import Dict, Any, List, Tuple
from .metabric_preprocessing import (
    validate_columns,
    coerce_types,
    scale_features,
    sanitize_record,
)

_CACHE: Dict[str, Any] = {}

def _fix_loss_import():
    try:
        import sys
        import sklearn._loss as skl_loss
        if "_loss" not in sys.modules:
            sys.modules["_loss"] = skl_loss
        if "loss" not in sys.modules:
            sys.modules["loss"] = skl_loss
    except Exception:
        pass

def _default_paths() -> Dict[str, str]:
    base = os.path.join(os.path.dirname(__file__), "..", "models")
    return {
        "model": os.environ.get("METABRIC_MODEL_PATH") or os.path.join(base, "metabric_model.pkl"),
        "scaler": os.environ.get("METABRIC_SCALER_PATH") or os.path.join(base, "metabric_scaler.pkl"),
        "features": os.environ.get("METABRIC_FEATURES_PATH") or os.path.join(base, "metabric_features.pkl"),
        "metadata": os.environ.get("METABRIC_METADATA_PATH") or os.path.join(base, "metadata.json"),
    }

def load_artifacts() -> Tuple[Any, Any, List[str], Dict[str, Any]]:
    if "artifacts" in _CACHE:
        return _CACHE["artifacts"]
    paths = _default_paths()
    _fix_loss_import()
    metadata = {}
    try:
        import json
        with open(paths["metadata"], "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        metadata = {}
    test_mode = os.environ.get("METABRIC_TEST_MODE") == "1"
    if test_mode:
        from sklearn.preprocessing import StandardScaler
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.linear_model import LinearRegression
        import pandas as pd
        candidates = [
            os.environ.get("METABRIC_EVAL_DATA_PATH"),
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "row", "metabric_cleaned_final.csv"),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "row", "metabric_cleaned_final.csv"),
        ]
        eval_path = next((p for p in candidates if p and os.path.exists(p)), None)
        if not eval_path:
            raise RuntimeError("Test mode fallback requires METABRIC_EVAL_DATA_PATH or metabric_cleaned_final.csv")
        df = pd.read_csv(eval_path)
        targets = ['aggressiveness_score','growth_rate','evolution_6m']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_names = [c for c in numeric_cols if c not in targets and not c.lower().startswith("unnamed")]
        X = df[feature_names].copy()
        y = df[targets].copy()
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X.values)
        model = MultiOutputRegressor(LinearRegression())
        model.fit(Xs, y.values)
        metadata["test_mode"] = True
    else:
        model = joblib.load(paths["model"])
        scaler = joblib.load(paths["scaler"])
        feature_names = joblib.load(paths["features"])
    _CACHE["artifacts"] = (model, scaler, feature_names, metadata)
    return _CACHE["artifacts"]

def predict_single(features: Dict[str, Any]) -> Dict[str, Any]:
    model, scaler, feature_names, _ = load_artifacts()
    row = [float(features.get(c, 0) or 0) for c in feature_names]
    X = np.array([row], dtype=float)
    Xs = scaler.transform(X)
    y_pred = model.predict(Xs)
    out = {
        "aggressiveness_score": float(y_pred[0][0]),
        "growth_rate": float(y_pred[0][1]),
        "evolution_6m_raw": float(y_pred[0][2]),
        "evolution_6m_class": int(np.round(y_pred[0][2]).clip(0, 2)),
    }
    return sanitize_record(out)

def predict_batch(csv_path: str) -> List[Dict[str, Any]]:
    model, scaler, feature_names, _ = load_artifacts()
    import csv
    rows_out: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data_rows = [r for r in reader]
    if not data_rows:
        return []
    for i, r in enumerate(data_rows):
        vals = []
        for c in feature_names:
            v = r.get(c, 0)
            try:
                vals.append(float(v))
            except Exception:
                vals.append(0.0)
        X = np.array([vals], dtype=float)
        Xs = scaler.transform(X)
        pred = model.predict(Xs)[0]
        rec = {
            "row_index": int(i),
            "aggressiveness_score": float(pred[0]),
            "growth_rate": float(pred[1]),
            "evolution_6m_raw": float(pred[2]),
            "evolution_6m_class": int(np.round(pred[2]).clip(0, 2)),
            "processing_status": "SUCCESS",
        }
        rows_out.append(sanitize_record(rec))
    return rows_out
