import numpy as np
import pandas as pd
import os

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("Initializing Phase 10: Multi-Model Hybrid Inverse-Design AI Engine...\n")

# ==========================================
# 1. LOAD SIMULATED DATASET
# ==========================================
input_path = os.path.join("data", "processed", "simulated_device_metrics.csv")
if not os.path.exists(input_path):
    input_path = "simulated_device_metrics.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Cannot find '{input_path}'. Execute Phase 9 first.")

df = pd.read_csv(input_path)
df = df[df['Simulated_ON_OFF_Ratio_2V'] > 0].copy()

# Target Variable in Log10 Space
df['Log_ON_OFF'] = np.log10(df['Simulated_ON_OFF_Ratio_2V'])

# ==========================================
# 2. FEATURE SELECTION (ALL NUMERICAL FEATURES)
# ==========================================
exclude_cols = [
    'Simulated_ON_OFF_Ratio_2V', 'Log_ON_OFF', 'J_OFF_2V_A_m2', 'J_ON_2V_A_m2',
    'Core_Material', 'Shell_Material', 'Core_Formula', 'Shell_Formula', 
    'Synthesis_Regime', 'Thermodynamically_Stable'
]

feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

print(f"Selected {len(feature_cols)} input features for machine learning:")
for f in feature_cols:
    print(f" - {f}")
print()

X = df[feature_cols].fillna(df[feature_cols].mean())
y = df['Log_ON_OFF']

# Feature Standardization for distance metrics (KNN, K-Means, Agglomerative)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. UNSUPERVISED CLUSTERING SEGMENTATION
# ==========================================
print("Executing Cluster Regimes Analysis...")

# Technique 1: K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# Technique 2: Agglomerative Hierarchical Clustering
agg_cluster = AgglomerativeClustering(n_clusters=4)
df['Agglo_Cluster'] = agg_cluster.fit_predict(X_scaled)

print(" -> Partitioned design space into 4 regime clusters via K-Means and Hierarchical Clustering.\n")

# ==========================================
# 4. MULTI-MODEL SURROGATE REGRESSION ENSEMBLE
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Training Machine Learning Regressors...")

# Model 1: 500-Tree Random Forest Regressor
rf_500 = RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42)
rf_500.fit(X_train, y_train)

# Model 2: K-Nearest Neighbors (KNN) Regressor
knn_model = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_model.fit(X_train, y_train)

# Model 3: Gradient Boosting Regressor (GBR)
gbr_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
gbr_model.fit(X_train, y_train)

# Predictions on Test Set
pred_rf_test = rf_500.predict(X_test)
pred_knn_test = knn_model.predict(X_test)
pred_gbr_test = gbr_model.predict(X_test)

print("\n=== MODEL PERFORMANCE ACCURACY BENCHMARKS ===")
print(f"500-Tree Random Forest  | R^2: {r2_score(y_test, pred_rf_test):.5f} | MAE: {mean_absolute_error(y_test, pred_rf_test):.4f}")
print(f"KNN Regressor           | R^2: {r2_score(y_test, pred_knn_test):.5f} | MAE: {mean_absolute_error(y_test, pred_knn_test):.4f}")
print(f"Gradient Boosting (GBR) | R^2: {r2_score(y_test, pred_gbr_test):.5f} | MAE: {mean_absolute_error(y_test, pred_gbr_test):.4f}")

# Full Space Ensemble Inference & Verification
df['Pred_RF_500'] = rf_500.predict(X_scaled)
df['Pred_KNN'] = knn_model.predict(X_scaled)
df['Pred_GBR'] = gbr_model.predict(X_scaled)

# Weighted Consensus Voting Ensemble
df['Ensemble_Log_ON_OFF'] = (0.50 * df['Pred_RF_500']) + (0.30 * df['Pred_GBR']) + (0.20 * df['Pred_KNN'])

# Spatial KNN Verification Disagreement
df['KNN_RF_Disagreement_MAE'] = abs(df['Pred_RF_500'] - df['Pred_KNN'])
print(f"\nAverage Model Verification Disagreement (RF vs KNN): {df['KNN_RF_Disagreement_MAE'].mean():.4f} log units\n")

# ==========================================
# 5. INVERSE DESIGN SEARCH & SELECTION
# ==========================================
# Search Filters
min_retention_years = 10.0
consensus_max_disagreement = 1.0

viable = df[
    (df['Retention_Time_Years'] >= min_retention_years) & 
    (df['KNN_RF_Disagreement_MAE'] <= consensus_max_disagreement)
].copy()

if viable.empty:
    print("\nWarning: No candidates met the 10-year retention threshold! Falling back to the absolute best available ON/OFF ratios.")
    viable = df.copy()

# Sort by highest predicted ON/OFF ratio ensemble score
top_candidates = viable.sort_values(by='Ensemble_Log_ON_OFF', ascending=False).head(3)

print("==================================================")
print("TOP 3 VERIFIED HYBRID INVERSE-DESIGN RECOMMENDATIONS")
print("==================================================")

for rank, (_, row) in enumerate(top_candidates.iterrows(), 1):
    core_formula = row.get('Core_Formula', row.get('Core_Material', 'Unknown Core'))
    shell_formula = row.get('Shell_Formula', row.get('Shell_Material', 'Unknown Shell'))
    
    print(f"\nRANK #{rank} RECOMMENDATION:")
    print(f"  Core Material              : {core_formula}")
    print(f"  Shell Material             : {shell_formula}")
    print(f"  Manufacturing Recipe:")
    print(f"    - Shell Thickness        : {row['Design_Thickness_nm']:.2f} nm")
    print(f"    - Trap Depth             : {row['Design_Trap_Depth_eV']:.2f} eV")
    print(f"  Unsupervised Cluster Segmentations:")
    print(f"    - K-Means Cluster ID     : Cluster {row['KMeans_Cluster']}")
    print(f"    - Agglomerative ID       : Cluster {row['Agglo_Cluster']}")
    print(f"  Multi-Model Surrogate Predictions:")
    print(f"    - RF (500 Trees) Log Ratio: {row['Pred_RF_500']:.2f}")
    print(f"    - KNN Validated Log Ratio : {row['Pred_KNN']:.2f}")
    print(f"    - GBR Predicted Log Ratio : {row['Pred_GBR']:.2f}")
    print(f"    - Ensemble Combined Ratio : 10^{row['Ensemble_Log_ON_OFF']:.2f}")
    print(f"    - Theoretical Retention   : {row['Retention_Time_Years']:.2e} Years")

# Save Inverse Design Outputs
os.makedirs(os.path.join("data", "final"), exist_ok=True)
output_path = os.path.join("data", "final", "inverse_design_recommendations.csv")
top_candidates.to_csv(output_path, index=False)
print(f"\nSUCCESS: Verified recommendations saved to '{output_path}'.")