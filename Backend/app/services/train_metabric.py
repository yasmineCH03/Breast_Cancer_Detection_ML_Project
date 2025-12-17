import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

def find_metabric_csv():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    possible_paths = [
        os.path.join(base_dir, "data", "row", "metabric_cleaned_final.csv"),
        os.path.join(base_dir, "Backend", "app", "data", "row", "metabric_cleaned_final.csv"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Could not find metabric_cleaned_final.csv")

def run_training(csv_path=None, test_size=0.2, random_state=42):
    if csv_path is None:
        csv_path = find_metabric_csv()
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Define targets and features
    targets = ['aggressiveness_score', 'growth_rate', 'evolution_6m']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_names = [c for c in numeric_cols if c not in targets and not c.lower().startswith("unnamed")]
    
    X = df[feature_names]
    y = df[targets]
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Set MLflow experiment
    mlflow.set_experiment("Prognosis Metabric Prediction")
    
    with mlflow.start_run():
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model Parameters
        n_estimators = 200
        learning_rate = 0.1
        max_depth = 5
        
        mlflow.log_params({
            "model_type": "MultiOutputRegressor(GradientBoostingRegressor)",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "test_size": test_size,
            "random_state": random_state
        })
        
        # Training
        print("Training model...")
        gb = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
        model = MultiOutputRegressor(gb)
        model.fit(X_train_scaled, y_train)
        
        # Evaluation
        print("Evaluating model...")
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        # Target 0: Aggressiveness Score
        r2_aggr = r2_score(y_test.iloc[:, 0], y_pred[:, 0])
        mae_aggr = mean_absolute_error(y_test.iloc[:, 0], y_pred[:, 0])
        
        # Target 1: Growth Rate
        r2_growth = r2_score(y_test.iloc[:, 1], y_pred[:, 1])
        mae_growth = mean_absolute_error(y_test.iloc[:, 1], y_pred[:, 1])
        
        # Target 2: Evolution 6M (Classification treated as regression, rounded)
        y_pred_class = np.round(y_pred[:, 2]).astype(int).clip(0, 2)
        acc_evol = accuracy_score(y_test.iloc[:, 2], y_pred_class)
        
        metrics = {
            "r2_aggressiveness": r2_aggr,
            "mae_aggressiveness": mae_aggr,
            "r2_growth_rate": r2_growth,
            "mae_growth_rate": mae_growth,
            "accuracy_evolution_6m": acc_evol
        }
        
        mlflow.log_metrics(metrics)
        print(f"Metrics: {metrics}")
        
        # Save Artifacts Locally
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, 'metabric_model.pkl')
        scaler_path = os.path.join(models_dir, 'metabric_scaler.pkl')
        features_path = os.path.join(models_dir, 'metabric_features.pkl')
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(feature_names, features_path)
        
        # Log Artifacts to MLflow
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(scaler_path, "scaler")
        mlflow.log_artifact(features_path, "features")
        
        return {
            "metrics": metrics,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "features_path": features_path
        }
