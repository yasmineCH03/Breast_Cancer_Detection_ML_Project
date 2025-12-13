import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from app.services.train_wbcd import run_training
from app.services.register import update_manifest

def main():
    metrics = run_training()
    update_manifest(metrics["model_path"], metrics["scaler_path"], {
        "accuracy": metrics["accuracy"],
        "precision_malignant": metrics["precision_malignant"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "auc": metrics["auc"],
        "confusion_matrix": metrics["confusion_matrix"]
    }, status="PRODUCTION")
    print(metrics)

if __name__ == "__main__":
    main()
