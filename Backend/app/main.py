"""
FASTAPI BACKEND - EVIDENCE-BASED MEDICAL AI API
Implements all endpoints from your master plan with NCCN/ACR compliance
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
from typing import List, Dict
import os
from pydantic import BaseModel

# Import your evidence-based utilities
from app.utils.evidence_based_prediction_utils import (
    predict_diagnosis_with_evidence,
    batch_predict_with_evidence,
    load_medical_guidelines
)

# ============================================================================
# 1. INITIALIZE FASTAPI APP
# ============================================================================
app = FastAPI(
    title="OncoAI Medical API",
    description="Evidence-based breast cancer detection with NCCN/ACR/USPSTF/ACS compliance",
    version="2.4.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# 2. LOAD PRODUCTION MODELS ON STARTUP
# ============================================================================
MODEL_PATH = "app/models/cellular_sgd_svm_v2.4.joblib"
SCALER_PATH = "app/models/cellular_scaler.joblib"

try:
    production_model = joblib.load(MODEL_PATH)
    production_scaler = joblib.load(SCALER_PATH)
    print("✅ Production models loaded successfully")
except Exception as e:
    print(f"❌ Failed to load models: {e}")
    production_model = None
    production_scaler = None

# ============================================================================
# 3. REQUEST/RESPONSE SCHEMAS (Pydantic)
# ============================================================================
class CellularFeatures(BaseModel):
    """30 WBCD features for single prediction"""
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float

class DiagnosisResponse(BaseModel):
    """Evidence-based diagnosis response"""
    diagnosis: str
    probability: float
    risk_tier: str
    recommended_action: str
    confidence: str
    model_used: str
    timestamp: str
    evidence_based: bool = True

# ============================================================================
# 4. GROUP A: MEDICAL INTERFACE ENDPOINTS (DOCTOR'S VIEW)
# ============================================================================
@app.post("/api/v1/cellular/predict", response_model=DiagnosisResponse)
async def predict_cellular(features: CellularFeatures):
    """
    Single prediction endpoint for doctors (WBCD)
    Returns evidence-based diagnosis with NCCN/ACR compliant actions
    """
    if not production_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to dict and use evidence-based prediction
        features_dict = features.dict()
        result = predict_diagnosis_with_evidence(features_dict)
        
        return DiagnosisResponse(
            diagnosis=result["diagnosis"]["predicted_class"],
            probability=result["diagnosis"]["malignancy_probability"],
            risk_tier=result["risk_assessment"]["risk_tier"],
            recommended_action=result["clinical_actions_summary"]["urgent_actions"][0]["action"] if result["clinical_actions_summary"]["urgent_actions"] else result["clinical_actions_summary"]["follow_up_actions"][0]["action"],
            confidence=result["diagnosis"]["confidence"],
            model_used=result["model_information"]["model_name"],
            timestamp=result["prediction_timestamp"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/v1/cellular/batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Batch CSV processing for multiple patients
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")
    
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process with evidence-based batch prediction
        results = batch_predict_with_evidence(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        return {
            "status": "success",
            "patients_processed": len(results),
            "results": results,
            "clinical_disclaimer": "AI-assisted evidence-based predictions. Physician review required."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")

# ============================================================================
# 5. GROUP B: TECHNICAL INTERFACE ENDPOINTS (ML ENGINEER'S VIEW)
# ============================================================================
@app.get("/api/v1/evaluation/cellular/metrics")
async def get_model_metrics():
    """
    Returns model comparison metrics for evaluation dashboard
    """
    try:
        with open("app/data/model_comparison.json", "r") as f:
            metrics = json.load(f)
        
        return {
            "test_set_size": metrics["test_set_size"],
            "models": metrics["models"],
            "last_updated": metrics["last_updated"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics loading error: {str(e)}")

@app.get("/api/v1/evaluation/cellular/shap")
async def get_shap_data():
    """
    Returns SHAP feature importance for explainability dashboard
    """
    try:
        with open("app/data/shap_explainability.json", "r") as f:
            shap_data = json.load(f)
        
        return {
            "top_features": shap_data["top_features"],
            "feature_count": len(shap_data.get("all_features", [])),
            "last_calculated": shap_data.get("last_calculated", "unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP data loading error: {str(e)}")

@app.get("/api/v1/evaluation/clinical/residuals")
async def get_residuals_data():
    """
    Returns regression error data for METABRIC model (placeholder for clinical workflow)
    """
    # Placeholder - you'll implement this when adding clinical model
    return {
        "status": "endpoint_configured",
        "message": "Clinical prognosis endpoints will be implemented in Phase 2",
        "planned_features": [
            "Actual vs Predicted survival months",
            "Residual plots for regression model",
            "Clinical feature weights"
        ]
    }

# ============================================================================
# 6. HEALTH & SYSTEM ENDPOINTS
# ============================================================================
@app.get("/api/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy" if production_model else "degraded",
        "service": "OncoAI Medical API",
        "version": "2.4.0",
        "model_loaded": production_model is not None,
        "evidence_based": True,
        "clinical_guidelines": ["NCCN v3.2023", "ACR BI-RADS", "USPSTF", "ACS 2023"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/guidelines")
async def get_clinical_guidelines():
    """Returns evidence-based medical guidelines"""
    try:
        guidelines = load_medical_guidelines()
        return {
            "risk_stratification": guidelines["risk_stratification"],
            "clinical_references": guidelines["clinical_references"],
            "implementation_date": guidelines["implementation_date"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Guidelines loading error: {str(e)}")

# ============================================================================
# 7. MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Evidence-Based Medical API...")
    print("📚 Clinical Guidelines: NCCN, ACR, USPSTF, ACS")
    print("🔗 API Documentation: http://localhost:8000/api/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)