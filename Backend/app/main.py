"""
FASTAPI BACKEND - EVIDENCE-BASED MEDICAL AI API
Implements all endpoints from your master plan with NCCN/ACR compliance
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import json
from datetime import datetime
import os
from pydantic import BaseModel
import logging

# Import your evidence-based utilities
from app.utils.evidence_based_prediction_utils import (
    predict_diagnosis_with_evidence,
    batch_predict_with_evidence,
    load_medical_guidelines
)
from app.services.wbcd_preprocessing import find_wbcd_csv, load_wbcd, clean_wbcd, get_feature_order


# ============================================================================
# 1. INITIALIZE FASTAPI APP
# ============================================================================
app = FastAPI(
    title="OncoAI Medical API",
    description="Evidence-based breast cancer detection with NCCN/ACR/USPSTF/ACS compliance",
    version="2.4.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("oncoai.api")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ============================================================================
# 2. LOAD PRODUCTION MODELS ON STARTUP
# ============================================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "cellular_sgd_svm_v2.4.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "models", "cellular_scaler.joblib")

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
    used_model: str
    timestamp: str
    evidence_based: bool = True

# ============================================================================
# 4. GROUP A: MEDICAL INTERFACE ENDPOINTS (DOCTOR'S VIEW)
# ============================================================================
@app.post("/api/v1/cellular/predict")
async def predict_cellular(features: CellularFeatures):
    """
    Single prediction endpoint for doctors (WBCD)
    Returns evidence-based diagnosis with NCCN/ACR compliant actions
    """
    if not production_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to dict and use evidence-based prediction
        features_dict = features.model_dump()
        result = predict_diagnosis_with_evidence(features_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/v1/cellular/batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Batch CSV processing for multiple patients
    """
    if not (file.filename and file.filename.endswith('.csv')):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")
    
    try:
        content = await file.read()
        size_bytes = len(content)
        if size_bytes > 20 * 1024 * 1024:
            logger.warning(f"Batch file too large: {size_bytes} bytes from {file.filename}")
            raise HTTPException(status_code=413, detail="CSV file too large (max 20MB)")

        # Save uploaded file temporarily
        temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Validate required columns before heavy processing
        try:
            df_head = pd.read_csv(temp_path, nrows=0)
        except Exception as e:
            logger.error(f"Failed to parse CSV {file.filename}: {e}")
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")
        
        required_cols = [
            'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean',
            'radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave_points_se','symmetry_se','fractal_dimension_se',
            'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave_points_worst','symmetry_worst','fractal_dimension_worst'
        ]
        present = set(df_head.columns.tolist())
        missing = [c for c in required_cols if c not in present]
        if missing:
            logger.info(f"Batch validation failed: missing columns {missing}")
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
        
        # Ensure CSV contains at least one data row
        try:
            df_rows = pd.read_csv(temp_path)
        except Exception as e:
            logger.error(f"Failed to parse CSV rows {file.filename}: {e}")
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail=f"Invalid CSV rows: {str(e)}")
        if df_rows.shape[0] == 0:
            logger.info(f"Batch validation failed: no data rows in {file.filename}")
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail="CSV contains headers only and no data rows")
        
        # Process with evidence-based batch prediction
        logger.info(f"Starting batch processing for {file.filename}, size={size_bytes} bytes")
        try:
            results = batch_predict_with_evidence(temp_path)
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass
        
        success_count = sum(1 for r in results if r.get("processing_status") == "SUCCESS")
        error_count = sum(1 for r in results if r.get("processing_status") != "SUCCESS")
        logger.info(f"Batch processed: {success_count} success, {error_count} errors")
        
        response = {
            "status": "success",
            "patients_processed": len(results),
            "results": results,
            "clinical_disclaimer": "AI-assisted evidence-based predictions. Physician review required."
        }
        if len(results) == 0:
            logger.warning(f"Batch processed but produced zero results for {file.filename}")
            response["status"] = "error"
            response["message"] = "No predictions computed. Verify CSV row contents."
        return response
    except HTTPException as he:
        logger.error(f"Batch processing error (HTTP {he.status_code}): {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")

class SavedRow(BaseModel):
    patient_id: str
    predicted_class: str
    probability: float
    risk_tier: str
    action: str
    timestamp: str
    top1_name: str | None = None
    top1_value: float | None = None
    top2_name: str | None = None
    top2_value: float | None = None
    top3_name: str | None = None
    top3_value: float | None = None
    top4_name: str | None = None
    top4_value: float | None = None
    top5_name: str | None = None
    top5_value: float | None = None

@app.post("/api/v1/cellular/batch/save")
async def save_batch_outcomes(payload: dict):
    """
    Append predicted outcomes to a CSV 'batch_predictions_db.csv'.
    If the file does not exist, writes header first.
    """
    try:
        rows = payload.get("rows", [])
        if not isinstance(rows, list) or len(rows) == 0:
            raise HTTPException(status_code=400, detail="No rows provided")
        to_save: list[SavedRow] = []
        for r in rows:
            try:
                sr_kwargs = {
                    "patient_id": str(r.get("patient_id") or r.get("id") or ""),
                    "predicted_class": str(r.get("predicted_class") or r.get("diagnosis") or "").upper(),
                    "probability": float(r.get("probability")),
                    "risk_tier": str(r.get("risk_tier") or "").upper(),
                    "action": str(r.get("action") or ""),
                    "timestamp": str(r.get("timestamp") or datetime.now().isoformat())
                }
                for i in range(1, 6):
                    name_key = f"top{i}_name"
                    value_key = f"top{i}_value"
                    if r.get(name_key):
                        sr_kwargs[name_key] = str(r.get(name_key))
                    if r.get(value_key) is not None:
                        try:
                            sr_kwargs[value_key] = float(r.get(value_key))
                        except Exception:
                            sr_kwargs[value_key] = None
                sr = SavedRow(**sr_kwargs)
                if not sr.patient_id or not sr.predicted_class:
                    continue
                to_save.append(sr)
            except Exception:
                continue
        if len(to_save) == 0:
            raise HTTPException(status_code=400, detail="No valid rows to save")
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        db_path = os.path.join(data_dir, "batch_predictions_db.csv")
        standard_header = ["patient_id","predicted_class","probability","risk_tier","action","timestamp",
                           "top1_name","top1_value","top2_name","top2_value","top3_name","top3_value","top4_name","top4_value","top5_name","top5_value"]
        write_header = not os.path.exists(db_path) or os.path.getsize(db_path) == 0
        # If file exists with different header, rewrite with standard header preserving rows
        if not write_header:
            try:
                import csv
                # Read existing header and rows FIRST, then close file handle before replacing (Windows-safe)
                with open(db_path, "r", encoding="utf-8", newline="") as fr:
                    reader = csv.reader(fr)
                    existing_header = next(reader, None)
                    # Re-read with DictReader for row mapping
                    fr.seek(0)
                    dict_reader = csv.DictReader(fr)
                    rows_existing = list(dict_reader)
                base6 = ["patient_id","predicted_class","probability","risk_tier","action","timestamp"]
                needs_upgrade = (
                    existing_header is None
                    or existing_header != standard_header
                    or len(existing_header) != len(standard_header)
                )
                if needs_upgrade:
                    tmp_path = db_path + ".tmp"
                    with open(tmp_path, "w", encoding="utf-8", newline="") as fw:
                        w = csv.writer(fw)
                        w.writerow(standard_header)
                        # Map existing rows to new schema by column names
                        for row in rows_existing:
                            base = [
                                row.get("patient_id",""),
                                row.get("predicted_class",""),
                                row.get("probability",""),
                                row.get("risk_tier",""),
                                row.get("action",""),
                                row.get("timestamp","")
                            ]
                            w.writerow(base + ["","","","","","","","","",""])
                    os.replace(tmp_path, db_path)
                    write_header = False
            except Exception:
                pass
        import csv
        with open(db_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(standard_header)
            for sr in to_save:
                w.writerow([
                    sr.patient_id, sr.predicted_class, sr.probability, sr.risk_tier, sr.action, sr.timestamp,
                    sr.top1_name or "", sr.top1_value if sr.top1_value is not None else "",
                    sr.top2_name or "", sr.top2_value if sr.top2_value is not None else "",
                    sr.top3_name or "", sr.top3_value if sr.top3_value is not None else "",
                    sr.top4_name or "", sr.top4_value if sr.top4_value is not None else "",
                    sr.top5_name or "", sr.top5_value if sr.top5_value is not None else ""
                ])
        return {"status":"saved","rows_saved":len(to_save),"path":"data/batch_predictions_db.csv"}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save error: {str(e)}")

# ============================================================================
# 5. GROUP B: TECHNICAL INTERFACE ENDPOINTS (ML ENGINEER'S VIEW)
# ============================================================================
@app.get("/api/v1/evaluation/cellular/metrics")
async def get_model_metrics():
    """
    Returns model comparison metrics for evaluation dashboard
    """
    try:
        with open(os.path.join(os.path.dirname(__file__), "data", "model_comparison.json"), "r") as f:
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
        with open(os.path.join(os.path.dirname(__file__), "data", "shap_explainability.json"), "r") as f:
            shap_data = json.load(f)
        
        return {
            "top_features": shap_data["top_features"],
            "feature_count": len(shap_data.get("all_features", [])),
            "last_calculated": shap_data.get("last_calculated", "unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP data loading error: {str(e)}")

@app.post("/api/v1/evaluation/cellular/shap")
async def post_shap_data():
    try:
        with open(os.path.join(os.path.dirname(__file__), "data", "shap_explainability.json"), "r") as f:
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
# 5b. WBCD EDA ENDPOINTS (Data Understanding)
# ============================================================================
@app.get("/api/v1/wbcd/eda/summary")
async def get_wbcd_eda_summary():
    try:
        csv_path = find_wbcd_csv()
        df = load_wbcd(csv_path)
        numeric_cols = df.select_dtypes(include=['float64','float32','int64','int32']).columns.tolist()
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        missing_counts = df.isnull().sum()
        missing_total = int(missing_counts.sum())
        return {
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "numeric_columns": numeric_cols,
            "object_columns": object_cols,
            "missing_values_total": missing_total,
            "missing_values_by_column": {col: int(missing_counts[col]) for col in df.columns if missing_counts[col] > 0}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"WBCD EDA summary error: {str(e)}")

@app.get("/api/v1/wbcd/eda/correlations")
async def get_wbcd_correlations(top_n: int = 30):
    try:
        csv_path = find_wbcd_csv()
        df = load_wbcd(csv_path)
        if df['diagnosis'].dtype == 'object':
            df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
        corr_series = df.corr(numeric_only=True)['diagnosis'].drop(labels=['diagnosis']).sort_values(ascending=False)
        top = corr_series.head(top_n)
        return {
            "top_n": top_n,
            "correlations": [{"feature": k, "correlation": float(v)} for k, v in top.items()]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"WBCD correlation error: {str(e)}")

@app.get("/api/v1/wbcd/features")
async def get_wbcd_feature_order():
    return {
        "features": [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
            'smoothness_mean', 'compactness_mean', 'concavity_mean',
            'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
            'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
            'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
            'fractal_dimension_se', 'radius_worst', 'texture_worst',
            'perimeter_worst', 'area_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'concave_points_worst',
            'symmetry_worst', 'fractal_dimension_worst'
        ]
    }

@app.get("/api/v1/wbcd/sample")
async def get_wbcd_random_sample(diagnosis: str | None = None):
    try:
        csv_path = find_wbcd_csv()
        df = load_wbcd(csv_path)
        X, y = clean_wbcd(df)
        order = get_feature_order()
        if diagnosis:
            key = diagnosis.strip().lower()
            if key in ("malignant", "m"):
                subset = X[y == 1]
                if subset.shape[0] > 0:
                    X = subset
            elif key in ("benign", "b"):
                subset = X[y == 0]
                if subset.shape[0] > 0:
                    X = subset
        sample = X.sample(1, random_state=datetime.now().microsecond).iloc[0]
        features = {k: float(sample[k]) for k in order}
        return {"patient_id": f"WBCD_{sample.name}", "features": features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"WBCD sample error: {str(e)}")

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
    print("🔗 API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
