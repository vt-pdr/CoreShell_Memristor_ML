import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# 1. LOAD ML FEATURE MATRIX
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
# 2. FEATURE SCALING (Crucial for Clustering)
# =====================================================================
print("Scaling physical features for multidimensional clustering...")
# K-Means calculates distance, so a bandgap of 5 eV and a dielectric ratio of 1000 
# will confuse it unless we standardize everything to the same scale.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================================
# 3. K-MEANS CLUSTERING
# =====================================================================
n_clusters = 4
print(f"Running K-Means algorithm to group devices into {n_clusters} performance tiers...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# =====================================================================
# 4. IDENTIFY THE "ELITE" CLUSTER
# =====================================================================
# We want the cluster with the highest average Total Confinement
cluster_stats = df.groupby('Cluster')['Total_Confinement_eV'].mean()
elite_cluster_id = cluster_stats.idxmax()

print(f"\nCluster Analysis (Average Total Confinement per tier):")
for cluster_id, mean_conf in cluster_stats.items():
    tier = "ELITE TIER" if cluster_id == elite_cluster_id else "Lower Tier"
    print(f"  -> Cluster {cluster_id}: {mean_conf:.2f} eV ({tier})")

elite_devices = df[df['Cluster'] == elite_cluster_id].copy()

# Sort the elite devices by their actual confinement to find the absolute best
elite_devices = elite_devices.sort_values(by='Total_Confinement_eV', ascending=False)

# =====================================================================
# 5. EXPORT RESULTS
# =====================================================================
output_file = "elite_memristor_clusters.csv"
elite_devices.head(100).to_csv(output_file, index=False)

print(f"\n==================================================")
print(f"SUCCESS! Isolated {len(elite_devices):,} Elite Candidates.")
print(f"Top 100 organic devices saved to '{output_file}'.")
print("==================================================")
print("\nTop 5 Discovered Core-Shell Pairs (No Data Leakage!):")
print(elite_devices[['Core_Formula', 'Shell_Formula', 'Total_Confinement_eV']].head())