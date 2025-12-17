from fastapi import APIRouter, HTTPException, Body
from app.schemas_triage import TriageRequest, TriageResponse, TriageInput
from app.services.triage_service import TriageService
from datetime import datetime
import os
import pandas as pd
import json
from pydantic import BaseModel

router = APIRouter(
    prefix="/triage",
    tags=["triage"]
)

class UpdateDateRequest(BaseModel):
    patient_id: str
    new_date: str

@router.get("/plan", response_model=TriageResponse)
async def get_triage_plan():
    """
    Get the current triage plan based on saved batch predictions.
    """
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        db_path = os.path.join(data_dir, "batch_predictions_db.csv")
        
        if not os.path.exists(db_path):
            return TriageResponse(planning=[], csv_content="")
            
        df = pd.read_csv(db_path)
        
        patients = []
        for _, row in df.iterrows():
            try:
                # Handle potentially missing columns safely
                d_count = 0
                if "danger_flag_count" in df.columns and pd.notna(row["danger_flag_count"]):
                    d_count = int(row["danger_flag_count"])
                
                prediction = "Malignant" if str(row.get("predicted_class", "")).upper() == "MALIGNANT" else "Benign"
                risk_tier = str(row.get("risk_tier", "Low")).upper()
                
                # Timestamp parsing
                ts_str = str(row.get("timestamp", datetime.now().isoformat()))
                try:
                    ts = datetime.fromisoformat(ts_str)
                except:
                    ts = datetime.now()

                ci = None
                if "clinical_interpretation" in df.columns:
                    val = row.get("clinical_interpretation")
                    if pd.notna(val) and isinstance(val, str):
                        try: ci = json.loads(val)
                        except: pass
                
                of = None
                if "original_features" in df.columns:
                    val = row.get("original_features")
                    if pd.notna(val) and isinstance(val, str):
                        try: of = json.loads(val)
                        except: pass

                manual_date = None
                if "manual_date" in df.columns:
                    val = row.get("manual_date")
                    if pd.notna(val) and isinstance(val, str) and str(val).strip():
                        manual_date = str(val).strip()

                p = TriageInput(
                    patient_id=str(row.get("patient_id", "Unknown")),
                    prediction=prediction,
                    probability=float(row.get("probability", 0.0)),
                    risk_tier=risk_tier,
                    danger_flag_count=d_count,
                    prediction_timestamp=ts,
                    clinical_interpretation=ci,
                    original_features=of,
                    manual_date=manual_date
                )
                patients.append(p)
            except Exception:
                continue
        
        if not patients:
             return TriageResponse(planning=[], csv_content="")

        planning_result = TriageService.plan_patients(patients)
        csv_content = TriageService.to_csv(planning_result)
        
        return TriageResponse(
            planning=planning_result,
            csv_content=csv_content
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update_date")
async def update_appointment_date(req: UpdateDateRequest):
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        db_path = os.path.join(data_dir, "batch_predictions_db.csv")
        if not os.path.exists(db_path):
            raise HTTPException(status_code=404, detail="Database not found")
            
        df = pd.read_csv(db_path)
        # Ensure manual_date column exists
        if "manual_date" not in df.columns:
            df["manual_date"] = None
            
        # Update
        mask = df["patient_id"].astype(str) == str(req.patient_id)
        if not mask.any():
            raise HTTPException(status_code=404, detail="Patient not found")
            
        df.loc[mask, "manual_date"] = req.new_date
        df.to_csv(db_path, index=False)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete/{patient_id}")
async def delete_patient(patient_id: str):
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        db_path = os.path.join(data_dir, "batch_predictions_db.csv")
        if not os.path.exists(db_path):
            raise HTTPException(status_code=404, detail="Database not found")
            
        df = pd.read_csv(db_path)
        mask = df["patient_id"].astype(str) == str(patient_id)
        if not mask.any():
            raise HTTPException(status_code=404, detail="Patient not found")
            
        df = df[~mask]
        df.to_csv(db_path, index=False)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/plan", response_model=TriageResponse)
async def generate_planning(request: TriageRequest):
    """
    Generate a prioritized clinical planning based on patient predictions.
    """
    try:
        # Perform planning starting from today
        # Note: In a real scenario, we might want to pass the start date or fetch existing schedule
        planning_result = TriageService.plan_patients(request.patients)
        
        # Generate CSV content
        csv_content = TriageService.to_csv(planning_result)
        
        return TriageResponse(
            planning=planning_result,
            csv_content=csv_content
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
