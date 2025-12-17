import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from app.services.train_metabric import run_training

def main():
    print("Starting Metabric Training Pipeline...")
    try:
        results = run_training()
        print("\nTraining Completed Successfully!")
        print("Metrics:", results["metrics"])
        print("Artifacts saved at:")
        print(f"- Model: {results['model_path']}")
        print(f"- Scaler: {results['scaler_path']}")
        print(f"- Features: {results['features_path']}")
    except Exception as e:
        print(f"Training Failed: {e}")
        raise e

if __name__ == "__main__":
    main()
