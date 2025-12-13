import os
import pandas as pd

def find_wbcd_csv():
    base = os.path.dirname(__file__)
    candidates = [
        os.path.abspath(os.path.join(base, '..', '..', '..', 'data', 'row', 'data.csv')),
        os.path.abspath(os.path.join(base, '..', '..', 'data', 'row', 'data.csv')),
        os.path.abspath(os.path.join(base, '..', 'data', 'row', 'data.csv')),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("WBCD CSV not found in expected locations")

def get_feature_order():
    return [
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

def load_wbcd(csv_path):
    return pd.read_csv(csv_path)

def clean_wbcd(df):
    rename_map = {
        'concave points_mean': 'concave_points_mean',
        'concave points_se': 'concave_points_se',
        'concave points_worst': 'concave_points_worst'
    }
    df = df.rename(columns=rename_map)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'Unnamed: 32' in df.columns:
        df = df.drop(columns=['Unnamed: 32'])
    if df['diagnosis'].dtype == 'object':
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0}).astype(int)
    order = get_feature_order()
    X = df[order].copy()
    y = df['diagnosis'].astype(int).copy()
    return X, y

def wbcd_summary(df):
    numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.tolist()
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    missing_counts = df.isnull().sum()
    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "numeric_columns": numeric_cols,
        "object_columns": object_cols,
        "missing_values_total": int(missing_counts.sum()),
        "missing_values_by_column": {col: int(missing_counts[col]) for col in df.columns if missing_counts[col] > 0}
    }

def wbcd_correlations(df, top_n=30):
    if df['diagnosis'].dtype == 'object':
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0}).astype(int)
    corr_series = df.corr(numeric_only=True)['diagnosis'].drop(labels=['diagnosis']).sort_values(ascending=False)
    top = corr_series.head(top_n)
    return [{"feature": k, "correlation": float(v)} for k, v in top.items()]
