# ============================================================================
# TEST MODÈLE SAUVEGARDÉ - SCRIPT INDÉPENDANT
# Fichier : deployment/test_model. py
# ============================================================================

import joblib
import pandas as pd
import numpy as np
import sys
import os

print("="*70)
print("🧪 TEST DU MODÈLE METABRIC SAUVEGARDÉ")
print("="*70)

# Vérifier que les fichiers existent
MODEL_DIR = 'models'

required_files = [
    f'{MODEL_DIR}/metabric_model.pkl',
    f'{MODEL_DIR}/metabric_scaler.pkl',
    f'{MODEL_DIR}/metabric_features.pkl'
]

print("\n📂 Vérification des fichiers...")
for file_path in required_files: 
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path) / 1024  # En KB
        print(f"✓ {file_path} ({file_size:.1f} KB)")
    else:
        print(f"❌ ERREUR : {file_path} introuvable !")
        sys.exit(1)

# Charger le modèle
print("\n📦 Chargement du modèle...")
try:
    model = joblib.load(f'{MODEL_DIR}/metabric_model.pkl')
    scaler = joblib.load(f'{MODEL_DIR}/metabric_scaler.pkl')
    features = joblib.load(f'{MODEL_DIR}/metabric_features.pkl')
    
    print(f"✓ Modèle chargé :  {type(model)}")
    print(f"✓ Scaler chargé : {type(scaler)}")
    print(f"✓ Features :  {len(features)} colonnes\n")
except Exception as e:
    print(f"❌ Erreur lors du chargement : {e}")
    sys.exit(1)

# Créer un patient test
print("="*70)
print("👤 PATIENT TEST")
print("="*70)

patient_test = {
    'age_at_diagnosis': 54.5,
    'tumor_size': 23.0,
    'tumor_stage_encoded': 2.0,
    'neoplasm_histologic_grade_encoded': 3.0,
    'cellularity_encoded': 2.0,
    'lymph_nodes_examined_positive': 0.0,
    'nottingham_prognostic_index':  4.04,
    'mutation_count':  5.0,
    'er_status_binary': 1,
    'her2_status_binary': 0,
    'pr_status_binary': 1,
    'overall_survival_months': 115.6,
    'overall_survival_binary': 1,
    'death_from_cancer_binary': 0,
    'pam50_+_claudin-low_subtype_encoded': 3,
    'integrative_cluster_encoded': 5,
    '3-gene_classifier_subtype_encoded': 2
}

# Afficher données patient
print("\n📋 Données patient :")
for key, value in patient_test.items():
    print(f"  {key: 45s} : {value}")
# Vérifier features manquantes
missing = [f for f in features if f not in patient_test]
if missing: 
    print(f"\n⚠️ Features manquantes : {missing}")
    print("Ajout valeurs par défaut (0)...")
    for feat in missing:
        patient_test[feat] = 0

# Créer DataFrame
patient_df = pd.DataFrame([patient_test], columns=features)

# Standardiser
print("\n🔧 Standardisation...")
patient_scaled = scaler.transform(patient_df)
print(f"✓ Shape après standardisation : {patient_scaled.shape}")

# Prédire
print("\n🔬 Prédiction en cours...")
prediction = model.predict(patient_scaled)[0]

# Afficher résultats
print("\n" + "="*70)
print("📊 RÉSULTATS PRÉDICTION")
print("="*70)

aggr_score = prediction[0]
growth_rate = prediction[1]
evol_raw = prediction[2]
evol_class = int(np.round(evol_raw).clip(0, 2))

evolution_labels = ['Stable', 'Modéré', 'Rapide']
risk_levels = ['Faible', 'Modéré', 'Élevé']

print(f"\n1️⃣ Aggressiveness Score : {aggr_score:. 2f}/10")
print(f"2️⃣ Growth Rate          : {growth_rate:.2f}%/mois")
print(f"3️⃣ Evolution 6M (brut)  : {evol_raw:. 2f}")
print(f"4️⃣ Evolution 6M (classe): {evol_class}")
print(f"5️⃣ Evolution 6M (label) : {evolution_labels[evol_class]}")

# Déterminer risque global
if aggr_score > 7 or evol_class == 2:
    risk_level = 'Élevé'
    risk_color = '🔴'
elif aggr_score > 5 or evol_class == 1:
    risk_level = 'Modéré'
    risk_color = '🟡'
else:
    risk_level = 'Faible'
    risk_color = '🟢'

print(f"\n{risk_color} RISQUE GLOBAL :  {risk_level}")

# Vérifier cohérence
print("\n" + "="*70)
print("✅ VÉRIFICATION COHÉRENCE")
print("="*70)

coherent = True

if aggr_score > 7 and evol_class == 0:
    print("⚠️ Incohérence :  Score élevé mais évolution stable")
    coherent = False
elif aggr_score < 5 and evol_class == 2:
    print("⚠️ Incohérence : Score faible mais évolution rapide")
    coherent = False
else:
    print("✓ Prédiction cohérente")

# Informations supplémentaires
print("\n" + "="*70)
print("📌 INFORMATIONS COMPLÉMENTAIRES")
print("="*70)

# NPI
npi = patient_test. get('nottingham_prognostic_index', 0)
print(f"\nNottingham Prognostic Index : {npi}")
if npi <= 3.4:
    print("  → Bon pronostic")
elif npi <= 5.4:
    print("  → Pronostic modéré")
else:
    print("  → Pronostic défavorable")

# Triple Négatif
is_triple_neg = (
    patient_test['er_status_binary'] == 0 and
    patient_test['her2_status_binary'] == 0 and
    patient_test['pr_status_binary'] == 0
)
print(f"\nTriple Négatif : {'OUI ⚠️' if is_triple_neg else 'NON ✓'}")

# Profil moléculaire
if is_triple_neg:
    profile = 'Triple Négatif'
elif patient_test['her2_status_binary'] == 1:
    profile = 'HER2+'
elif patient_test['er_status_binary'] == 1:
    profile = 'Luminal (ER+/HER2-)'
else:
    profile = 'Indéterminé'

print(f"Profil moléculaire : {profile}")

print("\n" + "="*70)
if coherent:
    print("✅ TEST RÉUSSI - Modèle fonctionne correctement !")
else:
    print("⚠️ TEST TERMINÉ - Vérifier incohérences")
print("="*70)