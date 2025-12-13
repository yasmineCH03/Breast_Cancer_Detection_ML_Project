"""
PRODUCTION PREDICTION UTILITIES - BREAST CANCER DETECTION
Medical-grade prediction functions for deployment.
"""

import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Paths (will be configured in deployment)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'cellular_sgd_svm_v2.4.joblib')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'cellular_scaler.joblib')

def load_production_model():
    """Load the production model with error handling."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        raise RuntimeError(f"Failed to load production model: {str(e)}")

def predict_diagnosis(features_dict):
    """
    Main prediction function for API endpoints.
    
    Args:
        features_dict: Dictionary with 30 WBCD feature values
        
    Returns:
        dict: Complete diagnosis report
    """
    model, scaler = load_production_model()
    
    # Feature order must match training
    feature_order = [
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
    
    # Validate input
    missing_features = [f for f in feature_order if f not in features_dict]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    # Prepare features
    features_array = np.array([features_dict[f] for f in feature_order]).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    # Risk stratification
    if probability >= 0.9:
        risk_tier = "CRITICAL"
        action = "Immediate biopsy & specialist referral"
    elif probability >= 0.8:
        risk_tier = "HIGH"
        action = "Schedule biopsy within 7 days"
    elif probability >= 0.6:
        risk_tier = "MEDIUM"
        action = "Ultrasound follow-up in 3 months"
    elif probability >= 0.4:
        risk_tier = "LOW"
        action = "Routine screening in 6 months"
    else:
        risk_tier = "VERY LOW"
        action = "Routine annual screening"
    
    return {
        'diagnosis': "MALIGNANT" if prediction == 1 else "BENIGN",
        'probability': float(probability),
        'risk_tier': risk_tier,
        'recommended_action': action,
        'confidence': 'HIGH' if probability >= 0.85 else 'MODERATE' if probability >= 0.70 else 'LOW',
        'model_used': 'SGD-SVM v2.4 (Calibrated)',
        'timestamp': datetime.now().isoformat()
    }

def batch_predict(csv_file_path):
    """
    Process batch predictions from CSV file.
    
    Args:
        csv_file_path: Path to CSV file with patient data
        
    Returns:
        list: Predictions for all patients
    """
    df = pd.read_csv(csv_file_path)
    predictions = []
    
    for _, row in df.iterrows():
        try:
            pred = predict_diagnosis(row.to_dict())
            pred['patient_id'] = row.get('patient_id', 'unknown')
            predictions.append(pred)
        except Exception as e:
            predictions.append({
                'patient_id': row.get('patient_id', 'unknown'),
                'error': str(e),
                'status': 'FAILED'
            })
    
    return predictions
