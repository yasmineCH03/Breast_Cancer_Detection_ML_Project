import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================================================
# 1. CLASS DEFINITIONS (Must match Notebook exactly)
# =========================================================
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier

class DataHandler:
    def __init__(self, filepath):
        self.filepath = filepath
    def load_and_clean(self):
        df = pd.read_csv(self.filepath)
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
        df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')
        return df.drop('diagnosis', axis=1), df['diagnosis']

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
    def fit_transform(self, X):
        return self.scaler.fit_transform(X)
    def transform(self, X):
        return self.scaler.transform(X)

class CancerModel:
    def __init__(self):
        self.model = None
    def train(self, X_train, y_train):
        pass 
    def predict_proba(self, X_data):
        return self.model.predict_proba(X_data)

class RiskStratifier:
    def __init__(self):
        self.feature_map = {
            'radius_mean': 'Radius Mean', 'texture_mean': 'Texture Mean', 
            'perimeter_mean': 'Perimeter Mean', 'area_mean': 'Area Mean', 
            'radius_worst': 'Worst Radius', 'texture_worst': 'Worst Texture', 
            'perimeter_worst': 'Worst Perimeter', 'area_worst': 'Worst Area',
            'concave points_worst': 'Worst Concave Points', 'concavity_worst': 'Worst Concavity'
        }

    def get_triage_data(self, prob, features_series_scaled, features_series_raw, feature_names):
        if prob >= 0.80:
            status = "Critical"
            action = "Book Biopsy Now"
        elif prob >= 0.40:
            status = "Urgent"
            action = "Review Today"
        else:
            status = "Low Risk"
            action = "Auto-scheduled 6mo"
            
        top_indices = np.argsort(np.abs(features_series_scaled))[::-1][:3]
        factor_strings = []
        for idx in top_indices:
            raw_name = feature_names[idx]
            if isinstance(features_series_raw, pd.Series):
                raw_val = features_series_raw.iloc[idx]
            else:
                raw_val = features_series_raw[idx]
            readable_name = self.feature_map.get(raw_name, raw_name)
            factor_strings.append(f"{readable_name} ({raw_val:.2f})")
            
        return status, ", ".join(factor_strings), action

# =========================================================
# 2. APP CONFIGURATION & LOADING
# =========================================================
st.set_page_config(page_title="OncoGuard Pro", page_icon="⚕️", layout="wide")

@st.cache_resource
def load_artifacts():
    # ABSOLUTE PATHS
    model_path = r'C:\Users\LENOVO\Desktop\Machine Learning\Bo2_Triage_risque\Breast_Cancer_Detection_ML_Project\Models'
    data_path = r'C:\Users\LENOVO\Desktop\Machine Learning\Bo2_Triage_risque\Breast_Cancer_Detection_ML_Project\data\row\data.csv'
    
    try:
        cm = joblib.load(os.path.join(model_path, 'cancer_model_v1.pkl'))
        fe = joblib.load(os.path.join(model_path, 'feature_engineer.pkl'))
        rs = joblib.load(os.path.join(model_path, 'risk_stratifier.pkl'))
        
        df = pd.read_csv(data_path)
        df_clean = df.drop(['id', 'Unnamed: 32', 'diagnosis'], axis=1, errors='ignore')
        return cm, fe, rs, df_clean
    except Exception as e:
        st.error(f"Error loading system files: {e}")
        return None, None, None, None

cm, fe, rs, df_ref = load_artifacts()

# =========================================================
# 3. HEADER
# =========================================================
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #2c3e50; font-weight: 700;}
    .sub-header {font-size: 1.2rem; color: #7f8c8d;}
    .highlight {background-color: #ecf0f1; padding: 10px; border-radius: 5px;}
    </style>
    """, unsafe_allow_html=True)

col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown('<div class="main-header">⚕️ OncoGuard Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Assisted Breast Cancer Triage System</div>', unsafe_allow_html=True)
with col_head2:
    st.image("https://img.icons8.com/color/96/000000/doctor-male.png", width=60)
    st.caption("Logged in: Chief Oncologist")

st.markdown("---")

if cm is not None:
    
    # TABS FOR DIFFERENT WORKFLOWS
    tab1, tab2 = st.tabs(["👤 Individual Patient Analysis", "📋 Batch Triage (Worklist)"])

    # =========================================================
    # TAB 1: INDIVIDUAL ANALYSIS (Deep Dive)
    # =========================================================
    with tab1:
        col_side, col_main = st.columns([1, 3])
        
        with col_side:
            st.markdown("### 🧬 Patient Simulator")
            st.info("Load a profile to simulate an incoming scan.")
            
            if 'current_patient_idx' not in st.session_state:
                st.session_state['current_patient_idx'] = np.random.randint(0, len(df_ref))

            if st.button("🎲 Load Random Case", type="primary"):
                st.session_state['current_patient_idx'] = np.random.randint(0, len(df_ref))

            idx = st.session_state['current_patient_idx']
            patient_data = df_ref.iloc[idx]
            
            st.markdown(f"**Patient ID:** P-{9000+idx}")
            
            with st.expander("⚙️ Adjust Clinical Values"):
                def user_input_features(ref_data):
                    input_dict = ref_data.copy()
                    r_w = float(ref_data['radius_worst'])
                    t_w = float(ref_data['texture_worst'])
                    cp_w = float(ref_data['concave points_worst'])

                    input_dict['radius_worst'] = st.slider('Worst Radius', 5.0, 40.0, r_w)
                    input_dict['texture_worst'] = st.slider('Worst Texture', 10.0, 50.0, t_w)
                    input_dict['concave points_worst'] = st.slider('Worst Concave Pts', 0.0, 0.3, cp_w)
                    return pd.DataFrame([input_dict])

                input_df = user_input_features(patient_data)

        with col_main:
            # Prediction Logic
            scaled_data = fe.transform(input_df)
            prob = cm.predict_proba(scaled_data)[0][1]
            status, factors, action = rs.get_triage_data(prob, scaled_data[0], input_df.iloc[0], input_df.columns)

            # Dashboard Display
            st.markdown("### 📊 Diagnostic Report")
            
            # 1. Top Stats
            m1, m2, m3 = st.columns(3)
            m1.metric("Malignancy Probability", f"{prob:.1%}", delta_color="inverse")
            m2.metric("Triage Category", status)
            m3.metric("Recommended Action", action)
            
            # 2. Alert Banner
            if status == "Critical":
                st.error(f"⚠️ **CRITICAL ALERT:** Immediate Biopsy Required. Probability > 80%")
            elif status == "Urgent":
                st.warning(f"⚠️ **URGENT:** Patient requires review within 24 hours.")
            else:
                st.success(f"✅ **LOW RISK:** Routine follow-up recommended.")

            # 3. Evidence
            st.markdown("#### 🔎 Key Clinical Drivers")
            st.markdown(f"""
            The AI model identified the following features as the most significant deviations from the norm:
            *   **{factors.split(', ')[0]}**
            *   **{factors.split(', ')[1]}**
            *   **{factors.split(', ')[2]}**
            """)
            
            with st.expander("View Raw Feature Data"):
                st.dataframe(input_df)

    # =========================================================
    # TAB 2: BATCH TRIAGE (Worklist)
    # =========================================================
    with tab2:
        st.markdown("### 📋 Incoming Scan Queue (Batch Processing)")
        
        c1, c2 = st.columns([1, 4])
        with c1:
            batch_size = st.number_input("Batch Size", min_value=5, max_value=50, value=10)
            run_batch = st.button("Process Incoming Batch", type="primary")
            
        if run_batch:
            with st.spinner("Analyzing scans..."):
                # 1. Generate Batch
                random_indices = np.random.choice(len(df_ref), batch_size, replace=False)
                batch_results = []
                
                for i in random_indices:
                    p_data = df_ref.iloc[i]
                    p_scaled = fe.transform(pd.DataFrame([p_data]))
                    p_prob = cm.predict_proba(p_scaled)[0][1]
                    p_status, p_factors, p_action = rs.get_triage_data(p_prob, p_scaled[0], p_data, df_ref.columns)
                    
                    batch_results.append({
                        "Patient ID": f"P-{9000+i}",
                        "Risk Score": p_prob,
                        "Status": p_status,
                        "Action Required": p_action,
                        "Key Factors": p_factors
                    })
                
                batch_df = pd.DataFrame(batch_results)
                
                # 2. Summary Stats
                n_crit = len(batch_df[batch_df['Status'] == "Critical"])
                n_urg = len(batch_df[batch_df['Status'] == "Urgent"])
                
                st.markdown("#### 📈 Batch Summary")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Scans", batch_size)
                k2.metric("Critical Cases", n_crit, delta=f"{n_crit/batch_size:.0%}", delta_color="inverse")
                k3.metric("Urgent Cases", n_urg, delta_color="off")
                k4.metric("Clear/Low Risk", batch_size - n_crit - n_urg)
                
                # 3. Detailed Table with Styling
                st.markdown("#### 🏥 Triage Worklist")
                
                # Custom column config for visual flair
                st.dataframe(
                    batch_df,
                    column_config={
                        "Risk Score": st.column_config.ProgressColumn(
                            "Risk Score",
                            help="Probability of Malignancy",
                            format="%.2f",
                            min_value=0,
                            max_value=1,
                        ),
                        "Status": st.column_config.TextColumn(
                            "Triage Status",
                            validate="^(Critical|Urgent|Low Risk)$"
                        ),
                    },
                    use_container_width=True,
                    hide_index=True
                )

else:
    st.error("System Malfunction: Models could not be loaded. Please contact IT Support.")
    
    