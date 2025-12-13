#!/usr/bin/env python3
"""
Diagnostic script for OncoAI setup
"""

import sys
import os

def check_python():
    print(f"Python: {sys.version}")
    return True

def check_venv():
    venv_path = os.path.abspath("venv")
    if os.path.exists(venv_path):
        print(f"Virtual env: {venv_path}")
        return True
    else:
        print("Virtual env: NOT FOUND")
        return False

def check_imports():
    packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "run"),
        ("sklearn", "__version__"),
        ("joblib", "load"),
        ("numpy", "__version__"),
        ("pandas", "__version__"),
    ]
    
    all_ok = True
    for package, attr in packages:
        try:
            module = __import__(package)
            if attr == "__version__":
                version = getattr(module, attr, "unknown")
                print(f"{package:15} ✅ {version}")
            else:
                print(f"{package:15} ✅ OK")
        except ImportError:
            print(f"{package:15} ❌ NOT INSTALLED")
            all_ok = False
    
    return all_ok

def main():
    print("=" * 50)
    print("🔍 ONCOAI SETUP DIAGNOSTIC")
    print("=" * 50)
    
    results = [
        ("Python", check_python()),
        ("Virtual Env", check_venv()),
        ("Packages", check_imports()),
    ]
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{name:15} [{status}]")
    
    print(f"\n✅ {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 Setup looks good!")
        print("Next: Run 'make api' to start the server")
    else:
        print("\n⚠️ Some checks failed. Please review above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
