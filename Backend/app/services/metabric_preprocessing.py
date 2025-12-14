import os
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any

def validate_columns(df: pd.DataFrame, feature_names: List[str]) -> List[str]:
    present = set(df.columns.tolist())
    return [c for c in feature_names if c not in present]

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].fillna(0)
    return out

def scale_features(df: pd.DataFrame, scaler: Any, features: List[str]) -> pd.DataFrame:
    X = df[features].copy()
    X_scaled = scaler.transform(X.values)
    return pd.DataFrame(X_scaled, columns=features, index=df.index)

def sanitize_value(v: Any) -> Any:
    try:
        if v is None:
            return None
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None
            return float(v)
        if isinstance(v, (np.floating,)):
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                return None
            return fv
        if isinstance(v, (np.integer,)):
            return int(v)
        if v is np.nan:
            return None
        return v
    except Exception:
        return None

def sanitize_record(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: sanitize_value(v) for k, v in d.items()}
