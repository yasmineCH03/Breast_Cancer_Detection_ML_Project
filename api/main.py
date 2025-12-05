from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import sys

# 1. SETUP & IMPORTS
# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml_utils import CancerModel, FeatureEngineer, RiskStratifier, DataHandler

# =========================================================
# 🔴 CRITICAL FIX FOR NOTEBOOK PICKLES
# =========================================================
# We map the imported classes to the '__main__' namespace
# so the pickle loader finds them where it expects them.
import __main__
__main__.CancerModel = CancerModel
__main__.FeatureEngineer = FeatureEngineer
__main__.RiskStratifier = RiskStratifier
__main__.DataHandler = DataHandler
# =========================================================

# 2. INITIALIZE APP
app = FastAPI(
    title="OncoGuard Pro API", 
    description="Industrial Grade Breast Cancer Inference Engine",
    version="2.0"
)

# 3. LOAD MODELS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to project root, then into Models
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'Models'))

print(f"🔎 Looking for models in: {MODEL_DIR}")

try:
    # Now this load should work because we tricked __main__
    cm = joblib.load(os.path.join(MODEL_DIR, 'cancer_model_v1.pkl'))
    fe = joblib.load(os.path.join(MODEL_DIR, 'feature_engineer.pkl'))
    rs = joblib.load(os.path.join(MODEL_DIR, 'risk_stratifier.pkl'))
    print("✅ All Models Loaded Successfully.")
except Exception as e:
    print(f"❌ CRITICAL ERROR: {e}")
    cm, fe, rs = None, None, None

# 4. DEFINE INPUT SCHEMA
class PatientData(BaseModel):
    features: dict

# 5. HEALTH CHECK
@app.get("/")
def health_check():
    return {"status": "online", "system": "OncoGuard AI"}

# 6. PREDICTION ENDPOINT
@app.post("/predict")
def predict(payload: PatientData):
    if cm is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    try:
        # A. Convert to DataFrame
        input_df = pd.DataFrame([payload.features])
        
        # B. Scale
        scaled_data = fe.transform(input_df)

        # C. Predict
        prob = float(cm.predict_proba(scaled_data)[0][1])
        
        # D. Stratify
        status, action, icon, css, factors = rs.get_triage_data(
            prob, 
            scaled_data[0], 
            input_df.iloc[0], 
            input_df.columns
        )
        
        return {
            "risk_score": prob,
            "status": status,
            "action_plan": action,
            "visual_icon": icon,
            "key_drivers": factors
        }

    except Exception as e:
        
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")