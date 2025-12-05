import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier

# ---------------------------------------------------------
# SHARED CLASS DEFINITIONS
# ---------------------------------------------------------

class DataHandler:
    def __init__(self, filepath): self.filepath = filepath
    def load_and_clean(self): 
        df = pd.read_csv(self.filepath)
        return df # Simplified for inference

class FeatureEngineer:
    def __init__(self): self.scaler = StandardScaler()
    def fit_transform(self, X): return self.scaler.fit_transform(X)
    def transform(self, X): return self.scaler.transform(X)

class CancerModel:
    def __init__(self): self.model = None
    def train(self, X_train, y_train): pass
    def predict_proba(self, X_data):
        if self.model is None: raise Exception("Model not loaded")
        return self.model.predict_proba(X_data)

class RiskStratifier:
    def __init__(self):
        self.feature_map = {
            'radius_mean': 'Mean Radius', 'texture_mean': 'Mean Texture', 
            'perimeter_mean': 'Mean Perimeter', 'area_mean': 'Mean Area', 
            'radius_worst': 'Worst Radius', 'texture_worst': 'Worst Texture', 
            'perimeter_worst': 'Worst Perimeter', 'area_worst': 'Worst Area',
            'concave points_worst': 'Worst Concave Pts', 'concavity_worst': 'Worst Concavity'
        }

    def get_triage_data(self, prob, features_series_scaled, features_series_raw, feature_names):
        # Thresholds
        if prob >= 0.85:
            status, action, css, icon = "CRITICAL", "IMMEDIATE BIOPSY", "status-critical", "🔴"
        elif prob >= 0.15:
            status, action, css, icon = "URGENT", "Radiology Review (24h)", "status-urgent", "🟠"
        else:
            status, action, css, icon = "BENIGN", "Routine Follow-up", "status-safe", "🟢"
            
        # Explainability (Top 3 Drivers)
        if isinstance(features_series_scaled, (pd.Series, pd.DataFrame)):
             vals = features_series_scaled.values.flatten()
        else:
             vals = features_series_scaled.flatten()

        top_indices = np.argsort(np.abs(vals))[::-1][:3]
        factors = []
        for idx in top_indices:
            raw_name = feature_names[idx]
            # Handle raw value extraction safely
            if isinstance(features_series_raw, pd.Series): val = features_series_raw.iloc[idx]
            elif isinstance(features_series_raw, (np.ndarray, list)): val = features_series_raw[idx]
            else: val = 0.0

            readable = self.feature_map.get(raw_name, raw_name)
            factors.append(f"<b>{readable}</b>: {val:.2f}")
            
        return status, action, icon, css, factors