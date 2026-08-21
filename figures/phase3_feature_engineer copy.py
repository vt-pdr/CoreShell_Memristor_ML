import pandas as pd
import numpy as np

# =====================================================================
# 1. LOAD TYPE-I PAIRS
# =====================================================================
input_file = "type_1_memristor_pairs.csv"
print(f"Loading Type-I pairs from '{input_file}'...")

try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"Error: Could not find '{input_file}'.")
    exit()

print(f"Loaded {len(df):,} initial pairs.")

# =====================================================================
# 2. BULLETPROOF DATA IMPUTATION (With Physical Fallbacks)
# =====================================================================
print("Imputing missing physical parameters...")

# If APIs failed and left columns entirely blank, the median is NaN.
# We inject standard physical baselines to guarantee the matrix survives.
fallbacks = {
    'Core_Dielectric_Static': 10.0,      # Typical transition metal oxide dielectric
    'Shell_Dielectric_Static': 10.0,
    'Core_Formation_Energy_eV': -2.0,    # Typical stable formation energy
    'Shell_Formation_Energy_eV': -2.0
}

for col, fallback_val in fallbacks.items():
    if df[col].isnull().all():
        print(f"  -> WARNING: '{col}' is 100% empty. Injecting physical fallback: {fallback_val}")
        df[col] = fallback_val
    else:
        median_val = df[col].median()
        # Fallback just in case the median still returns NaN for some reason
        if pd.isna(median_val):
            median_val = fallback_val
        df[col] = df[col].fillna(median_val)

# =====================================================================
# 3. FEATURE ENGINEERING
# =====================================================================
print("Engineering physical features for ML ingestion...")

df['Dielectric_Contrast_Ratio'] = df['Core_Dielectric_Static'] / df['Shell_Dielectric_Static']
df['Defect_Gradient_eV'] = df['Shell_Formation_Energy_eV'] - df['Core_Formation_Energy_eV']
df['Confinement_Asymmetry'] = df['dEc_eV'] / (df['dEv_eV'] + 1e-6) # 1e-6 prevents dividing by zero
df['Effective_System_Gap_eV'] = df['Shell_Eg_bulk_eV'] 

# =====================================================================
# 4. CLEANUP & EXPORT
# =====================================================================
print("Selecting final feature columns...")

identifiers = ['Core_Formula', 'Shell_Formula']

# --- FIXED: Added Shell_Dielectric_Static to the final output ---
ml_features = [
    'Core_Eg_nano_3nm_eV', 'Shell_Eg_bulk_eV',
    'dEc_eV', 'dEv_eV', 'Total_Confinement_eV',
    'Shell_Dielectric_Static',  
    'Dielectric_Contrast_Ratio', 'Defect_Gradient_eV', 
    'Confinement_Asymmetry', 'Effective_System_Gap_eV'
]

final_ml_df = df[identifiers + ml_features].copy()

# Final sweep: replace infinities with NaN, then drop the tiny fraction of remaining bad rows
final_ml_df = final_ml_df.replace([np.inf, -np.inf], np.nan).dropna()

output_file = "ml_feature_matrix.csv"
final_ml_df.to_csv(output_file, index=False)

print("\n==================================================")
print(f"SUCCESS! Feature Matrix compiled with {len(final_ml_df):,} viable devices.")
print(f"Saved ready-to-train dataset to '{output_file}'.")
print("==================================================")