import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# 1. LOAD & PREPARE FEATURE MATRIX
# =====================================================================
# Auto-resolve paths based on standard project structure
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) if "scripts" in os.path.dirname(os.path.abspath(__file__)) else os.getcwd()
input_file = os.path.join(PROJECT_ROOT, "data", "processed", "ml_feature_matrix.csv")

print(f"Loading feature matrix from '{input_file}'...")
df = pd.read_csv(input_file)

features = [
    'Core_Eg_nano_3nm_eV', 'Shell_Eg_bulk_eV',
    'Total_Confinement_eV', 'Dielectric_Contrast_Ratio', 
    'Defect_Gradient_eV', 'Confinement_Asymmetry'
]

X = df[features]

# =====================================================================
# 2. STAGE 1: UNSUPERVISED FILTERING (Advanced K-Means)
# =====================================================================
print("\n[STAGE 1] Running Advanced K-Means (Robust Scaling + Dynamic K)...")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print("  -> Optimizing cluster count (k) via Silhouette analysis...")
best_k = 4
best_score = -1

subset_indices = np.random.choice(X_scaled.shape[0], size=min(20000, X_scaled.shape[0]), replace=False)
X_subset = X_scaled[subset_indices]

for k in range(3, 8):
    temp_kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    temp_labels = temp_kmeans.fit_predict(X_subset)
    score = silhouette_score(X_subset, temp_labels)
    if score > best_score:
        best_score = score
        best_k = k

print(f"  -> Optimal clusters determined: k={best_k} (Silhouette Score: {best_score:.3f})")

kmeans = KMeans(n_clusters=best_k, init='k-means++', n_init=20, max_iter=500, algorithm='elkan', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Identify Elite Cluster
elite_cluster_id = df.groupby('Cluster')['Total_Confinement_eV'].mean().idxmax()
elite_df = df[df['Cluster'] == elite_cluster_id].copy()
print(f"  -> Isolated {len(elite_df):,} Elite candidates from {len(df):,} total pairs.")

# =====================================================================
# 3. STAGE 2: TARGET GENERATION (Distance to Ideal Centroid)
# =====================================================================
print("\n[STAGE 2] Calculating physical proximity to Elite Cluster Centroid...")
elite_indices = elite_df.index
X_elite_scaled = X_scaled[elite_indices]
centroid = X_elite_scaled.mean(axis=0)

distances = np.linalg.norm(X_elite_scaled - centroid, axis=1)
elite_df['Elite_Balance_Score'] = 1 / (1 + distances)

# =====================================================================
# 4. STAGE 3: TRI-MODEL TRAINING (XGBoost + RF + KNN)
# =====================================================================
print("\n[STAGE 3] Training Deep Hybrid ML System (XGBoost + Random Forest + KNN)...")
X_elite = elite_df[features]
y_elite = elite_df['Elite_Balance_Score']

scaler_elite = RobustScaler()
X_elite_scaled_norm = scaler_elite.fit_transform(X_elite)

X_train, X_test, y_train, y_test, X_train_s, X_test_s = train_test_split(
    X_elite, y_elite, X_elite_scaled_norm, test_size=0.2, random_state=42
)

# 1. Primary Model: XGBoost (1200 Trees)
xgb_model = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.02, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train, y_train)

# 2. Secondary Ensemble: Random Forest (500 Trees)
rf_model = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 3. Distance Verifier: KNN (k=10)
knn_model = KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=-1)
knn_model.fit(X_train_s, y_train)

# Evaluations
y_pred_xgb = xgb_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test_s)

print(f"\nModel Performance Metrics:")
print(f"  -> XGBoost (1200 Trees) : R² = {r2_score(y_test, y_pred_xgb):.4f} | RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_xgb)):.4f}")
print(f"  -> Random Forest (500)  : R² = {r2_score(y_test, y_pred_rf):.4f} | RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.4f}")
print(f"  -> KNN Verifier (k=10)  : R² = {r2_score(y_test, y_pred_knn):.4f} | RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_knn)):.4f}")

# =====================================================================
# 5. SIDE VERIFICATION & CONSENSUS SCORING
# =====================================================================
print("\n[STAGE 4] Performing Cross-Model Verification...")
elite_df['XGB_Quality'] = xgb_model.predict(X_elite)
elite_df['RF_Quality'] = rf_model.predict(X_elite)
elite_df['KNN_Quality'] = knn_model.predict(X_elite_scaled_norm)

# Agreement between extreme models (Tree vs Distance)
diff = np.abs(elite_df['XGB_Quality'] - elite_df['KNN_Quality'])
elite_df['Verification_Agreement_%'] = np.clip(100 * (1 - (diff / elite_df['XGB_Quality'])), 0, 100)

# Tri-Model Weighted Consensus
elite_df['Consensus_Quality'] = (0.50 * elite_df['XGB_Quality']) + (0.30 * elite_df['RF_Quality']) + (0.20 * elite_df['KNN_Quality'])

final_ranked = elite_df.sort_values(by='Consensus_Quality', ascending=False)

output_file = os.path.join(PROJECT_ROOT, "data", "processed", "hybrid_validated_memristors.csv")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

export_cols = ['Core_Formula', 'Shell_Formula', 'Consensus_Quality', 'XGB_Quality', 'RF_Quality', 'KNN_Quality', 'Verification_Agreement_%'] + features
final_ranked[export_cols].head(500).to_csv(output_file, index=False)

print(f"\n==================================================")
print(f"SUCCESS! Top 500 consensus-validated devices saved to:\n'{output_file}'")
print("==================================================")
print("\nTop 15 Tri-Verified Candidates:")
display_cols = ['Core_Formula', 'Shell_Formula', 'Consensus_Quality', 'Verification_Agreement_%']
print(final_ranked[display_cols].head(15).to_string(index=False))