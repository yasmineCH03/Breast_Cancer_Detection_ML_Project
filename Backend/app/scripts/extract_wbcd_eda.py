import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import json
from app.services.wbcd_preprocessing import load_wbcd, wbcd_summary, wbcd_correlations

def main():
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'row', 'data.csv')
    df = load_wbcd(csv_path)
    summary = wbcd_summary(df)
    corrs = wbcd_correlations(df, top_n=30)
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, 'wbcd_eda_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(data_dir, 'wbcd_correlations.json'), 'w', encoding='utf-8') as f:
        json.dump({"top_n": 30, "correlations": corrs}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
