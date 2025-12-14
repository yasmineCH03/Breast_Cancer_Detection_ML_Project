import os
import shutil
import joblib

def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

def main():
    root = repo_root()
    src = os.path.join(root, "Notebooks", "deployment", "models")
    dest = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(dest, exist_ok=True)
    files = ["metabric_model.pkl", "metabric_scaler.pkl", "metabric_features.pkl"]
    copied = []
    for f in files:
        sp = os.path.join(src, f)
        dp = os.path.join(dest, f)
        if not os.path.exists(sp):
            print(f"❌ Missing source: {sp}")
            continue
        shutil.copy2(sp, dp)
        copied.append(dp)
        print(f"✓ Copied {sp} -> {dp}")
    if not copied:
        print("No files copied. Ensure notebooks saved artifacts to Notebooks/deployment/models")
        return
    # Validate loading
    try:
        model = joblib.load(os.path.join(dest, "metabric_model.pkl"))
        scaler = joblib.load(os.path.join(dest, "metabric_scaler.pkl"))
        features = joblib.load(os.path.join(dest, "metabric_features.pkl"))
        print(f"✅ Validation OK: model={type(model)}, scaler={type(scaler)}, features={len(features)}")
    except Exception as e:
        print(f"❌ Validation failed: {e}")

if __name__ == "__main__":
    main()
