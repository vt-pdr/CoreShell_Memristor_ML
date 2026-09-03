import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. FILE ACCESS (UPDATED RELATIVE PATHS)
# ---------------------------------------------------------
train_file = 'data/processed/simulated_device_metrics.csv'
test_file = 'data/raw/synthetic_rs_dataset_real_eps.csv'

sim_df = pd.read_csv(train_file)
syn_df = pd.read_csv(test_file)

# ---------------------------------------------------------
# 2. FEATURE MAPPING & PREPROCESSING
# ---------------------------------------------------------
train_feats = ['Core_Eg_nano_3nm_eV', 'Shell_Eg_bulk_eV', 'Calculated_Eps_r', 'Design_Thickness_nm']
test_feats = ['core_Eg_eV', 'shell_Eg_eV', 'shell_dielectric_constant_est', 'shell_thickness_nm']

X_train_raw = sim_df[train_feats].values
X_test_raw = syn_df[test_feats].fillna(syn_df[test_feats].mean()).values

y_train_reg = np.log10(sim_df['Simulated_ON_OFF_Ratio_2V'].clip(lower=1e-9)).values
y_test_ref = np.log10(syn_df['on_off_ratio'].clip(lower=1e-9)).values
y_train_clf = (sim_df['Simulated_ON_OFF_Ratio_2V'] > 1e4).astype(int).values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

final_df = pd.DataFrame({
    'core_material': syn_df['core_material'],
    'shell_material': syn_df['shell_material'],
    'Original_Synthetic_Log_ON_OFF': y_test_ref
})

print("Running 5-Step Escalation Pipeline across your datasets...")

# ---------------------------------------------------------
# 3. 5-STEP INTENSITY LOOPS ACROSS 9 ALGORITHMS
# ---------------------------------------------------------

# Random Forest (300 to 1500 trees)
for step, n_est in enumerate([300, 600, 900, 1200, 1500], start=1):
    reg = RandomForestRegressor(n_estimators=n_est, max_depth=5+step*2, random_state=42, n_jobs=-1)
    clf = RandomForestClassifier(n_estimators=n_est, max_depth=5+step*2, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_train_reg)
    clf.fit(X_train, y_train_clf)
    if step == 5:
        final_df['RF_Reg_Pred'] = reg.predict(X_test)
        final_df['RF_Clf_Pred_Class'] = clf.predict(X_test)

# KNN (K=15 to K=3 with distance weighting)
for step, (k, w) in enumerate([(15, 'uniform'), (10, 'uniform'), (7, 'distance'), (5, 'distance'), (3, 'distance')], start=1):
    knn = KNeighborsRegressor(n_neighbors=k, weights=w)
    knn.fit(X_train, y_train_reg)
    if step == 5:
        final_df['KNN_Reg_Pred'] = knn.predict(X_test)

# SVM (Linear to RBF kernel, C up to 50)
for step, (c_val, kernel) in enumerate([(0.1, 'linear'), (1.0, 'linear'), (1.0, 'rbf'), (10.0, 'rbf'), (50.0, 'rbf')], start=1):
    svm = SVR(C=c_val, kernel=kernel)
    svm.fit(X_train, y_train_reg)
    if step == 5:
        final_df['SVM_Reg_Pred'] = svm.predict(X_test)

# Gradient Boosting (50 to 500 trees)
for step, (est, lr, md) in enumerate([(50, 0.1, 3), (100, 0.1, 4), (200, 0.05, 5), (350, 0.05, 6), (500, 0.01, 7)], start=1):
    gb = GradientBoostingRegressor(n_estimators=est, learning_rate=lr, max_depth=md, random_state=42)
    gb.fit(X_train, y_train_reg)
    if step == 5:
        final_df['GB_Reg_Pred'] = gb.predict(X_test)

# Logistic Regression (L2 lbfgs to L1 saga solver)
for step, (c_val, penalty, solver) in enumerate([(0.01, 'l2', 'lbfgs'), (0.1, 'l2', 'lbfgs'), (1.0, 'l2', 'lbfgs'), (10.0, 'l1', 'saga'), (50.0, 'l1', 'saga')], start=1):
    logreg = LogisticRegression(C=c_val, penalty=penalty, solver=solver, max_iter=2000, random_state=42)
    logreg.fit(X_train, y_train_clf)
    if step == 5:
        final_df['LogReg_Clf_Pred_Class'] = logreg.predict(X_test)

# Ridge Regression (Alpha 100 to 0.1)
for step, alpha in enumerate([100.0, 50.0, 10.0, 1.0, 0.1], start=1):
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train_reg)
    if step == 5:
        final_df['Ridge_Reg_Pred'] = ridge.predict(X_test)

# KMeans Clustering (K=2 to 10, n_init up to 200)
for step, (k, ni) in enumerate([(2, 10), (4, 30), (6, 50), (8, 100), (10, 200)], start=1):
    kmeans = KMeans(n_clusters=k, n_init=ni, random_state=42)
    preds = kmeans.fit_predict(X_test)
    if step == 5:
        final_df['KMeans_Cluster'] = preds

# Hierarchical Clustering (Linkage average to ward)
for step, (linkage, k) in enumerate([('average', 2), ('complete', 4), ('ward', 6), ('ward', 8), ('ward', 10)], start=1):
    agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    preds = agg.fit_predict(X_test)
    if step == 5:
        final_df['Hierarchical_Cluster'] = preds

# DBSCAN (Eps 1.5 to 0.3)
for step, (eps, min_s) in enumerate([(1.5, 3), (1.0, 4), (0.8, 5), (0.5, 5), (0.3, 10)], start=1):
    dbscan = DBSCAN(eps=eps, min_samples=min_s)
    preds = dbscan.fit_predict(X_test)
    if step == 5:
        final_df['DBSCAN_Cluster'] = preds

# ---------------------------------------------------------
# 4. DISCREPANCY EVALUATION & FILE EXPORT
# ---------------------------------------------------------
final_df['Abs_Diff_RF_vs_Original'] = np.abs(final_df['RF_Reg_Pred'] - final_df['Original_Synthetic_Log_ON_OFF'])

os.makedirs('output_verification', exist_ok=True)
out_path = 'output_verification/high_intensity_pipeline_results.csv'
final_df.to_csv(out_path, index=False)

print(f"Done. Output saved to: '{out_path}'")