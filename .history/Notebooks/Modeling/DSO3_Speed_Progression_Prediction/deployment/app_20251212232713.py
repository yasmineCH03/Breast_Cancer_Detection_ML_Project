# ============================================================================
# API FLASK - METABRIC PREDICTION
# Fichier : deployment/app.py
# ============================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

# Initialiser Flask
app = Flask(__name__)
CORS(app)  # Permettre requêtes depuis frontend

# Chemin vers les modèles
MODEL_DIR = 'models'

# Charger modèles au démarrage
print("🚀 Démarrage API METABRIC...")
print(f"📂 Chargement modèles depuis :  {os.path.abspath(MODEL_DIR)}")

# APRÈS (version robuste)
try:
    model = joblib.load(f'{MODEL_DIR}/metabric_model.pkl')
    scaler = joblib.load(f'{MODEL_DIR}/metabric_scaler.pkl')
    features = joblib.load(f'{MODEL_DIR}/metabric_features.pkl')
    
    print(f"✅ Modèle chargé : {type(model)}")
    print(f"✅ Scaler chargé : {type(scaler)}")
    print(f"✅ Features :  {len(features)}")
    
    # Charger metadata (optionnel)
    import json
    import os
    
    metadata_path = f'{MODEL_DIR}/metadata.json'
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json. load(f)
        print(f"✅ Metadata chargée : {metadata['model_name']}")
        print(f"✅ Performance : R²={metadata['performance']['r2_aggressiveness']:. 4f}")
    else:
        # Créer metadata par défaut
        print("⚠️ metadata.json non trouvé, utilisation valeurs par défaut")
        metadata = {
            'model_name': 'METABRIC Model',
            'date_training':  'Unknown',
            'n_features': len(features),
            'feature_names': features,
            'performance': {
                'r2_aggressiveness':  0.0,
                'r2_growth_rate': 0.0,
                'accuracy_evolution_6m': 0.0
            },
            'dataset': {
                'name': 'METABRIC',
                'n_samples_train': 0,
                'n_samples_test': 0
            }
        }
    
except Exception as e:
    print(f"❌ Erreur chargement : {e}")
    raise

# ============================================================================
# ROUTES API
# ============================================================================

@app.route('/')
def home():
    """Page d'accueil API"""
    return jsonify({
        'status': 'OK',
        'name': 'METABRIC Prediction API',
        'version':  '1.0',
        'model':  metadata['model_name'],
        'features_count': len(features),
        'endpoints': {
            '/':  'GET - Page accueil',
            '/health': 'GET - Statut API',
            '/features': 'GET - Liste features requises',
            '/predict': 'POST - Faire prédiction'
        }
    })

@app.route('/health')
def health():
    """Vérifier santé API"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features_count': len(features)
    })

@app.route('/features')
def get_features():
    """Lister features requises"""
    return jsonify({
        'features': features,
        'count': len(features),
        'example': {
            'age_at_diagnosis': 54.5,
            'tumor_size': 23.0,
            'tumor_stage_encoded': 2.0,
            # ... (montrer structure)
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint principal de prédiction"""
    try:
        # 1. Récupérer données JSON
        data = request.get_json()
        
        if not data: 
            return jsonify({'error': 'Pas de données fournies'}), 400
        
        # 2. Valider présence features
        missing = [f for f in features if f not in data]
        if missing: 
            return jsonify({
                'error': 'Features manquantes',
                'missing': missing,
                'received': list(data.keys())
            }), 400
        
        # 3. Créer DataFrame
        input_df = pd.DataFrame([data], columns=features)
        
        # 4. Standardiser
        input_scaled = scaler. transform(input_df)
        
        # 5. Prédire
        prediction = model. predict(input_scaled)[0]
        
        # 6. Formater résultats
        aggr_score = float(prediction[0])
        growth_rate = float(prediction[1])
        evol_raw = float(prediction[2])
        evol_class = int(np.round(evol_raw).clip(0, 2))
        
        evolution_labels = ['Stable', 'Modéré', 'Rapide']
        risk_levels = ['Faible', 'Modéré', 'Élevé']
        
        # Déterminer niveau risque global
        if aggr_score > 7 or evol_class == 2:
            risk_level = 'Élevé'
        elif aggr_score > 5 or evol_class == 1:
            risk_level = 'Modéré'
        else:
            risk_level = 'Faible'
        
        result = {
            'aggressiveness_score': round(aggr_score, 2),
            'growth_rate':  round(growth_rate, 2),
            'evolution_6m': evol_class,
            'evolution_6m_label': evolution_labels[evol_class],
            'risk_level': risk_level,
            'details': {
                'evolution_raw': round(evol_raw, 2),
                'npi':  round(float(data. get('nottingham_prognostic_index', 0)), 2),
                'triple_negative': bool(
                    data.get('er_status_binary', 1) == 0 and
                    data.get('her2_status_binary', 1) == 0 and
                    data.get('pr_status_binary', 1) == 0
                )
            }
        }
        
        return jsonify({
            'status': 'success',
            'predictions': result,
            'model':  metadata['model_name']
        })
    
    except Exception as e: 
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================================================
# LANCER SERVEUR
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏥 API METABRIC DÉMARRÉE")
    print("="*60)
    print("📍 URL : http://localhost:5000")
    print("📖 Documentation : http://localhost:5000/")
    print("🧪 Test :  http://localhost:5000/health")
    print("="*60 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )