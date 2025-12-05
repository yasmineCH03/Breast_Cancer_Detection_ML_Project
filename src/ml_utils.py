# FILE: src/ml_utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier

# --- PASTE YOUR CLASSES HERE ---

class DataHandler:
    def __init__(self, filepath):
        self.filepath = filepath
    def load_and_clean(self):
        df = pd.read_csv(self.filepath)
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
        df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')
        return df.drop('diagnosis', axis=1), df['diagnosis']

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
    def fit_transform(self, X):
        return self.scaler.fit_transform(X)
    def transform(self, X):
        return self.scaler.transform(X)

class CancerModel:
    def __init__(self):
        self.model = None
    def train(self, X_train, y_train):
        base_model = SGDClassifier(loss='hinge', penalty='l2', alpha=0.01, 
                                   random_state=42, class_weight='balanced')
        self.model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
        self.model.fit(X_train, y_train)
    def predict_proba(self, X_data):
        if self.model is None:
             raise Exception("Model not trained")
        return self.model.predict_proba(X_data)

class RiskStratifier:
    def __init__(self):
        # Mapping for display logic
        self.feature_map = {
            'radius_mean': 'Radius Mean', 'texture_mean': 'Texture Mean', 
            'perimeter_mean': 'Perimeter Mean', 'area_mean': 'Area Mean', 
            'radius_worst': 'Worst Radius', 'texture_worst': 'Worst Texture', 
            'perimeter_worst': 'Worst Perimeter', 'area_worst': 'Worst Area',
            'concave points_worst': 'Worst Concave Points', 'concavity_worst': 'Worst Concavity'
            # ... add others if needed
        }

    def get_triage_data(self, prob, features_series_scaled, features_series_raw, feature_names):
        if prob >= 0.80:
            status = "🔴 Critical"
            action = "Book Biopsy Now"
        elif prob >= 0.40:
            status = "🟠 Urgent"
            action = "Review Today"
        else:
            status = "🟢 Low"
            action = "Auto-scheduled 6mo"
            
        # Top 3 Factors Logic
        top_indices = np.argsort(np.abs(features_series_scaled))[::-1][:3]
        factor_strings = []
        for idx in top_indices:
            raw_name = feature_names[idx]
            raw_val = features_series_raw[idx] # Access by integer index if numpy array
            readable_name = self.feature_map.get(raw_name, raw_name)
            factor_strings.append(f"{readable_name}: {raw_val:.2f}")
            
        return status, ", ".join(factor_strings), action