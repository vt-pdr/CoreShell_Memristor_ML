import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# 1. LOAD & PREPARE FEATURE MATRIX
# =====================================================================
input_file = "ml_feature_matrix.csv"
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
# 4. STAGE 3: DUAL MODEL TRAINING (XGBoost + KNN Verification)
# =====================================================================
print("\n[STAGE 3] Training XGBoost (300 Trees) & KNN Side-Verifier...")
X_elite = elite_df[features]
y_elite = elite_df['Elite_Balance_Score']

# Separate scaler for elite feature space verification
scaler_elite = RobustScaler()
X_elite_scaled_norm = scaler_elite.fit_transform(X_elite)

X_train, X_test, y_train, y_test, X_train_s, X_test_s = train_test_split(
    X_elite, y_elite, X_elite_scaled_norm, test_size=0.2, random_state=42
)

# Primary Model: XGBoost Regressor
xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

# Verification Model: Distance-Weighted KNN Regressor (k=10)
knn_model = KNeighborsRegressor(n_neighbors=10, weights='distance')
knn_model.fit(X_train_s, y_train)

# Performance Evaluations
y_pred_xgb = xgb_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test_s)

print(f"\nPrimary Model (XGBoost) Performance:")
print(f"  -> R² Score: {r2_score(y_test, y_pred_xgb):.4f} | RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_xgb)):.4f}")

print(f"Verifier Model (KNN) Performance:")
print(f"  -> R² Score: {r2_score(y_test, y_pred_knn):.4f} | RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_knn)):.4f}")

# =====================================================================
# 5. SIDE VERIFICATION & CONSENSUS SCORING
# =====================================================================
print("\n[STAGE 4] Performing Cross-Model Verification...")
elite_df['XGB_Quality'] = xgb_model.predict(X_elite)
elite_df['KNN_Quality'] = knn_model.predict(X_elite_scaled_norm)

# Verification metric: Model agreement percentage
diff = np.abs(elite_df['XGB_Quality'] - elite_df['KNN_Quality'])
elite_df['Verification_Agreement_%'] = np.clip(100 * (1 - (diff / elite_df['XGB_Quality'])), 0, 100)

# Weighted Consensus Score: 70% XGBoost + 30% KNN
elite_df['Consensus_Quality'] = (0.7 * elite_df['XGB_Quality']) + (0.3 * elite_df['KNN_Quality'])

final_ranked = elite_df.sort_values(by='Consensus_Quality', ascending=False)

output_file = "hybrid_validated_memristors.csv"
export_cols = ['Core_Formula', 'Shell_Formula', 'Consensus_Quality', 'XGB_Quality', 'KNN_Quality', 'Verification_Agreement_%', 'Total_Confinement_eV'] + features
final_ranked[export_cols].head(100).to_csv(output_file, index=False)

print(f"\n==================================================")
print(f"SUCCESS! Top 100 consensus-validated devices saved to '{output_file}'.")
print("==================================================")
print("\nTop 20 Dual-Verified Candidates:")
display_cols = ['Core_Formula', 'Shell_Formula', 'Consensus_Quality', 'XGB_Quality', 'KNN_Quality', 'Verification_Agreement_%']
print(final_ranked[display_cols].head(20).to_string(index=False))