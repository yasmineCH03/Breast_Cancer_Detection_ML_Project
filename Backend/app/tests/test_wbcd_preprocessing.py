import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from app.services.wbcd_preprocessing import load_wbcd, clean_wbcd, get_feature_order

def test_clean_wbcd_feature_order():
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'row', 'data.csv')
    df = load_wbcd(csv_path)
    X, y = clean_wbcd(df)
    order = get_feature_order()
    assert list(X.columns) == order
    assert y.dtype.kind in ('i', 'u')
    assert 'id' not in df.columns or 'id' not in X.columns
    assert 'Unnamed: 32' not in X.columns
