from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class TriageInput(BaseModel):
    patient_id: str
    prediction: str  # "Malignant" or "Benign"
    probability: float  # 0.0 to 1.0
    risk_tier: Optional[str] = None  # Added field
    danger_flag_count: int  # 0 to 30
    prediction_timestamp: datetime
    clinical_interpretation: Optional[List[dict]] = None
    original_features: Optional[dict] = None
    manual_date: Optional[str] = None

class TriageOutput(BaseModel):
    patient_id: str
    prediction: str
    probability: float
    risk_tier: Optional[str] = None  # Added field
    danger_flag_count: int
    urgency_score: float
    priority_category: str
    appointment_date: str  # YYYY-MM-DD
    planning_status: str  # "CONFIRMED" or "DELAY_ALERT"
    clinical_interpretation: Optional[List[dict]] = None
    original_features: Optional[dict] = None

class TriageRequest(BaseModel):
    patients: List[TriageInput]

class TriageResponse(BaseModel):
    planning: List[TriageOutput]
    csv_content: Optional[str] = None
