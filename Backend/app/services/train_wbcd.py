import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from .wbcd_preprocessing import load_wbcd, clean_wbcd, find_wbcd_csv

def run_training(csv_path=None, test_size=0.3, random_state=42):
    if csv_path is None:
        csv_path = find_wbcd_csv()
    df = load_wbcd(csv_path)
    X, y = clean_wbcd(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Set MLflow experiment
    mlflow.set_experiment("Breast Cancer Detection")

    with mlflow.start_run():
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        alpha = 1.0 / (5.0 * float(X_train.shape[0]))
        
        # Log parameters
        mlflow.log_params({
            "test_size": test_size,
            "random_state": random_state,
            "alpha": alpha,
            "model_type": "SGDClassifier",
            "loss": "hinge",
            "penalty": "l2",
            "max_iter": 3000
        })

        sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=alpha, learning_rate='constant', eta0=0.001, max_iter=3000, tol=1e-4, class_weight='balanced', random_state=random_state, n_jobs=-1)
        clf = CalibratedClassifierCV(base_estimator=sgd, cv=5, method='sigmoid')
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec_m = precision_score(y_test, y_pred, pos_label=1)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).ravel().tolist()

        # Log metrics
        mlflow.log_metrics({
            "accuracy": acc,
            "precision_malignant": prec_m,
            "recall": rec,
            "f1_score": f1,
            "auc": auc
        })

        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, 'cellular_sgd_svm_v2.4.joblib')
        scaler_path = os.path.join(models_dir, 'cellular_scaler.joblib')
        joblib.dump(clf, model_path)
        joblib.dump(scaler, scaler_path)

        # Log model to MLflow
        mlflow.sklearn.log_model(clf, "model")
        mlflow.log_artifact(scaler_path, "scaler")

        return {
            "accuracy": float(acc),
            "precision_malignant": float(prec_m),
            "recall": float(rec),
            "f1_score": float(f1),
            "auc": float(auc),
            "confusion_matrix": cm,
            "model_path": model_path,
            "scaler_path": scaler_path
        }