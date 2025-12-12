# ============================================================================
# TEST API - PRÉDICTION
# ============================================================================

import requests
import json

API_URL = "http://localhost:5000"

print("="*70)
print("🧪 TEST API METABRIC")
print("="*70)

# 1. Test Health
print("\n1️⃣ Test /health")
try:
    response = requests.get(f"{API_URL}/health")
    print(f"   Status :  {response.status_code}")
    print(f"   Réponse : {response.json()}")
except Exception as e: 
    print(f"   ❌ Erreur : {e}")

# 2. Test Features
print("\n2️⃣ Test /features")
try:
    response = requests. get(f"{API_URL}/features")
    data = response.json()
    print(f"   Nombre features :  {data['count']}")
    print(f"   Premières features : {data['features'][: 5]}")
except Exception as e:
    print(f"   ❌ Erreur :  {e}")

# 3. Test Prédiction
print("\n3️⃣ Test /predict")

# Données patient test
# 3. Test Prédiction
print("\n3️⃣ Test /predict")

# Données patient test
patient_data = {
    # Features de base
    'age_at_diagnosis': 54.5,
    'tumor_size': 23.0,
    'tumor_stage_encoded': 2.0,
    'neoplasm_histologic_grade_encoded':  3.0,
    'cellularity_encoded': 2.0,
    'lymph_nodes_examined_positive': 0.0,
    'nottingham_prognostic_index': 4.04,
    'mutation_count':  5.0,
    'er_status_binary': 1,
    'her2_status_binary': 0,
    'pr_status_binary': 1,
    'overall_survival_months': 115.6,
    'overall_survival_binary': 1,
    'death_from_cancer_binary':  0,
    'pam50_+_claudin-low_subtype_encoded': 3,
    'integrative_cluster_encoded': 5,
    '3-gene_classifier_subtype_encoded': 2,
    
    # Features calculées (AJOUTÉES)
    'hormone_receptor_score': 2,  # ER=1 + PR=1 = 2
    'triple_negative': 0,  # ER=1, donc pas triple négatif
    'size_category': 1,  # 23mm → catégorie 1 (10-30mm)
    'grade_stage_interaction': 6.0,  # grade(3) * stage(2) = 6
    'high_risk': 0  # NPI=4. 04 < 5.4 → pas haut risque
}

try:
    response = requests.post(
        f"{API_URL}/predict",
        json=patient_data,
        headers={'Content-Type':  'application/json'}
    )
    
    print(f"   Status : {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n   ✅ PRÉDICTION RÉUSSIE :")
        print(f"   {json.dumps(result, indent=4)}")
        
        pred = result['predictions']
        print(f"\n   📊 RÉSUMÉ :")
        print(f"   Aggressiveness Score : {pred['aggressiveness_score']}/10")
        print(f"   Growth Rate          : {pred['growth_rate']}%/mois")
        print(f"   Evolution 6M         : {pred['evolution_6m_label']}")
        print(f"   Risque Global        : {pred['risk_level']}")
    else:
        print(f"   ❌ Erreur :  {response.json()}")
        
except Exception as e:
    print(f"   ❌ Erreur : {e}")

print("\n" + "="*70)
print("✅ TEST TERMINÉ")
print("="*70)