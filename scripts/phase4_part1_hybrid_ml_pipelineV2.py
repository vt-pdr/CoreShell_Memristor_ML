import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

warnings.filterwarnings('ignore')

# =====================================================================
# 1. LOAD & PREPARE FEATURE MATRIX
# =====================================================================
PROJECT_ROOT = (
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if "scripts" in os.path.dirname(os.path.abspath(__file__))
    else os.getcwd()
)
input_file = os.path.join(PROJECT_ROOT, "data", "processed", "ml_feature_matrix.csv")

print(f"Loading feature matrix from '{input_file}'...")
df = pd.read_csv(input_file)

features = [
    'Core_Eg_nano_3nm_eV',
    'Shell_Eg_bulk_eV',
    'Total_Confinement_eV',
    'Dielectric_Contrast_Ratio',
    'Defect_Gradient_eV',
    'Confinement_Asymmetry',
]

X = df[features]

# =====================================================================
# 2. STAGE 1: UNSUPERVISED FILTERING (Robust K-Means)
# =====================================================================
print("\n[STAGE 1] Running K-Means Filtering...")
scaler_global = RobustScaler()
X_scaled = scaler_global.fit_transform(X)

rng = np.random.RandomState(42)
subset_size = min(20000, X_scaled.shape[0])
subset_indices = rng.choice(X_scaled.shape[0], size=subset_size, replace=False)
X_subset = X_scaled[subset_indices]

best_k = 4
best_score = -1
for k in range(3, 8):
    temp_kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    temp_labels = temp_kmeans.fit_predict(X_subset)
    score = silhouette_score(X_subset, temp_labels)
    if score > best_score:
        best_score = score
        best_k = k

kmeans = KMeans(
    n_clusters=best_k,
    init='k-means++',
    n_init=20,
    max_iter=500,
    algorithm='elkan',
    random_state=42,
)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Identify Elite Cluster
elite_cluster_id = df.groupby('Cluster')['Total_Confinement_eV'].mean().idxmax()
elite_df = df[df['Cluster'] == elite_cluster_id].copy()
print(f" -> Isolated {len(elite_df):,} Elite candidates from {len(df):,} total pairs.")

# =====================================================================
# 3. STAGE 2: TARGET GENERATION (Physical Centroid Proximity)
# =====================================================================
print("\n[STAGE 2] Calculating Centroid Distance Target Score...")
elite_indices = elite_df.index
X_elite_scaled = X_scaled[elite_indices]
centroid = X_elite_scaled.mean(axis=0)

distances = np.linalg.norm(X_elite_scaled - centroid, axis=1)
asymmetry_penalty = np.where(np.abs(elite_df['Confinement_Asymmetry']) > 0.35, 0.15, 0.0)
elite_df['Elite_Balance_Score'] = (1 / (1 + distances)) - asymmetry_penalty

# =====================================================================
# 4. STAGE 3: DUAL-TREE MODEL TRAINING (XGBoost + ExtraTrees)
# =====================================================================
print("\n[STAGE 3] Training Dual-Tree Ensemble (XGBoost + ExtraTrees)...")
X_elite = elite_df[features].values
y_elite = elite_df['Elite_Balance_Score'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_elite, y_elite, test_size=0.2, random_state=42
)

# 1. Primary Model: XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(X_train, y_train)

# 2. Secondary Model: ExtraTrees Regressor
et_model = ExtraTreesRegressor(
    n_estimators=400,
    max_depth=12,
    min_samples_leaf=3,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
)
et_model.fit(X_train, y_train)

# Test Performance
y_pred_xgb = xgb_model.predict(X_test)
y_pred_et = et_model.predict(X_test)

print(f" -> XGBoost    : R² = {r2_score(y_test, y_pred_xgb):.4f} | RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_xgb)):.4f}")
print(f" -> ExtraTrees : R² = {r2_score(y_test, y_pred_et):.4f} | RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_et)):.4f}")

# =====================================================================
# 5. STAGE 4: OUT-OF-FOLD CROSS-VERIFICATION & CONSENSUS
# =====================================================================
print("\n[STAGE 4] Out-of-Fold Cross-Verification (5-Fold CV)...")
cv = KFold(n_splits=5, shuffle=True, random_state=42)

oof_xgb = cross_val_predict(xgb_model, X_elite, y_elite, cv=cv, n_jobs=-1)
oof_et = cross_val_predict(et_model, X_elite, y_elite, cv=cv, n_jobs=-1)

elite_df['XGB_Quality'] = oof_xgb
elite_df['ET_Quality'] = oof_et

# Absolute Delta Agreement Metric
diff = np.abs(oof_xgb - oof_et)
mean_pred = (oof_xgb + oof_et) / 2
elite_df['Verification_Agreement_%'] = np.clip(100 * (1 - (diff / np.maximum(mean_pred, 1e-6))), 0, 100)

# Dual-Tree Consensus (60% XGB / 40% ExtraTrees)
elite_df['Consensus_Quality'] = (0.60 * oof_xgb) + (0.40 * oof_et)
final_ranked = elite_df.sort_values(by='Consensus_Quality', ascending=False)

# =====================================================================
# 6. STAGE 5: SAVE ARTIFACTS & EXPORT DATASET
# =====================================================================
models_dir = os.path.join(PROJECT_ROOT, "models")
data_dir = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

joblib.dump(xgb_model, os.path.join(models_dir, "xgb_model_v1.joblib"))
joblib.dump(et_model, os.path.join(models_dir, "extra_trees_model_v1.joblib"))
joblib.dump(scaler_global, os.path.join(models_dir, "scaler_global_v1.joblib"))

output_file = os.path.join(data_dir, "phase4a_dualtree_memristors.csv")
export_cols = [
    'Core_Formula',
    'Shell_Formula',
    'Consensus_Quality',
    'XGB_Quality',
    'ET_Quality',
    'Verification_Agreement_%',
] + features

final_ranked[export_cols].head(500).to_csv(output_file, index=False)
print(f"SUCCESS! Dual-Tree candidates saved to: '{output_file}'")