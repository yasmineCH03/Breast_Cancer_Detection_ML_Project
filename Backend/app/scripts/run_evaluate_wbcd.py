import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from app.services.evaluate_wbcd import evaluate

def main():
    result = evaluate()
    print(result["metrics"])

if __name__ == "__main__":
    main()
