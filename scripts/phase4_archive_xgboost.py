import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

# =====================================================================
# 1. LOAD ML FEATURE MATRIX
# =====================================================================
input_file = "ml_feature_matrix.csv"
print(f"Loading feature matrix from '{input_file}'...")
df = pd.read_csv(input_file)

# =====================================================================
# 2. DEFINE TARGET METRIC (Physics-Informed Figure of Merit)
# =====================================================================
# Higher FOM = better carrier confinement, field focusing, and vacancy drift
df['Target_FOM'] = (
    df['Total_Confinement_eV'] * 
    np.log1p(df['Dielectric_Contrast_Ratio'].clip(lower=0)) * 
    df['Defect_Gradient_eV'].clip(lower=0.1)
)

features = [
    'Core_Eg_nano_3nm_eV', 'Shell_Eg_bulk_eV',
    'dEc_eV', 'dEv_eV', 'Total_Confinement_eV',
    'Dielectric_Contrast_Ratio', 'Defect_Gradient_eV', 
    'Confinement_Asymmetry', 'Effective_System_Gap_eV'
]

X = df[features]
y = df['Target_FOM']

# =====================================================================
# 3. TRAIN/TEST SPLIT & MODEL TRAINING
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training XGBoost Regressor on {len(X_train):,} pairs...")
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nModel Evaluation Metrics:")
print(f"  -> R² Score: {r2:.4f}")
print(f"  -> RMSE:     {rmse:.4f}")

# =====================================================================
# 4. FEATURE IMPORTANCE ANALYSIS
# =====================================================================
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances (Top Drivers of Device Performance):")
for _, row in importance_df.iterrows():
    print(f"  * {row['Feature']:<28}: {row['Importance']*100:.2f}%")

# =====================================================================
# 5. RANK AND EXPORT TOP MEMRISTOR PAIRS
# =====================================================================
df['Predicted_FOM'] = model.predict(X)
ranked_df = df.sort_values(by='Predicted_FOM', ascending=False)

output_file = "top_predicted_memristors.csv"
ranked_df[['Core_Formula', 'Shell_Formula', 'Predicted_FOM'] + features].head(100).to_csv(output_file, index=False)

print(f"\n==================================================")
print(f"Top 100 candidate devices saved to '{output_file}'.")
print("==================================================")
print("\nTop 5 Core-Shell Pairs Identified:")
print(ranked_df[['Core_Formula', 'Shell_Formula', 'Predicted_FOM', 'Total_Confinement_eV']].head())