import os
import sys
import pandas as pd
import numpy as np

print("=" * 70)
print("     PHASE 3: AUDITED & REFACTORED ML FEATURE ENGINEERING ENGINE     ")
print("=" * 70)

# 1. PATH RESOLUTION 
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
except NameError:
    curr = os.getcwd()
    while curr != os.path.dirname(curr):
        if os.path.exists(os.path.join(curr, "data")):
            break
        curr = os.path.dirname(curr)
    PROJECT_ROOT = curr if os.path.exists(os.path.join(curr, "data")) else os.getcwd()

INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "type_1_memristor_pairs.csv")
RAW_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "core_shell_parameters_full.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ml_feature_matrix.csv")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file '{INPUT_FILE}' not found.")

print(f"Loading candidate pairs from: '{INPUT_FILE}'")
df = pd.read_csv(INPUT_FILE)

# 2. DEDUPLICATION
identifiers = ['Core_MP_ID', 'Shell_MP_ID', 'Core_Formula', 'Shell_Formula']
df = df.drop_duplicates(subset=['Core_MP_ID', 'Shell_MP_ID']).copy()

# 3. DYNAMIC MERGE LOGIC
form_core_col = 'Core_Formation_Energy_eV' if 'Core_Formation_Energy_eV' in df.columns else 'Core_Formation_Energy_per_atom_eV'
form_shell_col = 'Shell_Formation_Energy_eV' if 'Shell_Formation_Energy_eV' in df.columns else 'Shell_Formation_Energy_per_atom_eV'

if (form_core_col not in df.columns or form_shell_col not in df.columns) and os.path.exists(RAW_FILE):
    print("  -> Merging formation energy parameters from Phase 1 raw data...")
    raw_preview = pd.read_csv(RAW_FILE, nrows=1).columns
    raw_energy_col = 'Formation_Energy_per_atom_eV' if 'Formation_Energy_per_atom_eV' in raw_preview else 'Formation_Energy_eV'
    
    df_raw = pd.read_csv(RAW_FILE)[['MP_ID', raw_energy_col]].drop_duplicates(subset=['MP_ID'])
    
    if form_core_col not in df.columns:
        df = df.merge(
            df_raw.rename(columns={'MP_ID': 'Core_MP_ID', raw_energy_col: 'Core_Formation_Energy_eV'}),
            on='Core_MP_ID', how='left'
        )
        form_core_col = 'Core_Formation_Energy_eV'
        
    if form_shell_col not in df.columns:
        df = df.merge(
            df_raw.rename(columns={'MP_ID': 'Shell_MP_ID', raw_energy_col: 'Shell_Formation_Energy_eV'}),
            on='Shell_MP_ID', how='left'
        )
        form_shell_col = 'Shell_Formation_Energy_eV'

if ("per_atom" in form_core_col) != ("per_atom" in form_shell_col):
    raise ValueError(f"Unit mismatch: Cannot compute driving force between {form_core_col} and {form_shell_col}.")

# 4. SCHEMA INTEGRITY CHECK
required_cols = [
    'Core_MP_ID', 'Shell_MP_ID', 'Core_Formula', 'Shell_Formula',
    'Core_Eg_nano_3nm_eV', 'Shell_Eg_bulk_eV', 'dEc_eV', 'dEv_eV',
    'Core_Dielectric_Static', 'Shell_Dielectric_Static',
    form_core_col, form_shell_col
]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise KeyError(f"Missing required schema columns: {missing_cols}")

df = df.dropna(subset=required_cols).copy()
df = df[(df['Core_Eg_nano_3nm_eV'] > 0) & (df['Shell_Eg_bulk_eV'] > 0)].copy()

# 5. PHYSICAL FEATURE ENGINEERING
print("Engineering composite physical descriptors...")

denom_confinement = np.abs(df['dEc_eV']) + np.abs(df['dEv_eV'])
df['Confinement_Asymmetry'] = np.where(
    denom_confinement > 1e-6,
    (df['dEc_eV'] - df['dEv_eV']) / denom_confinement,
    0.0
)

# Standard features
df['Log_Dielectric_Ratio'] = np.log(df['Core_Dielectric_Static'] / df['Shell_Dielectric_Static'])
df['Log_Bandgap_Ratio'] = np.log(df['Core_Eg_nano_3nm_eV'] / df['Shell_Eg_bulk_eV'])
df['Thermodynamic_Driving_Force_eV'] = df[form_shell_col] - df[form_core_col]

# --- PHASE 4 REQUIRED FEATURES ---
df['Total_Confinement_eV'] = df['dEc_eV'] + df['dEv_eV']
df['Dielectric_Contrast_Ratio'] = df['Core_Dielectric_Static'] / df['Shell_Dielectric_Static']
df['Defect_Gradient_eV'] = df['Thermodynamic_Driving_Force_eV'] 

# 6. FEATURE MATRIX COMPILATION
ml_features = [
    'Core_Eg_nano_3nm_eV',
    'Shell_Eg_bulk_eV',               # Added for Phase 4
    'Total_Confinement_eV',           # Added for Phase 4
    'Dielectric_Contrast_Ratio',      # Added for Phase 4
    'Defect_Gradient_eV',             # Added for Phase 4
    'dEc_eV',
    'dEv_eV',
    'Core_Dielectric_Static',
    'Shell_Dielectric_Static',
    'Log_Dielectric_Ratio',
    'Thermodynamic_Driving_Force_eV',
    'Confinement_Asymmetry',
    'Log_Bandgap_Ratio'
]

final_df = df[identifiers + ml_features].copy()
final_df = final_df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

if len(final_df) == 0:
    raise ValueError("Feature compilation resulted in 0 valid rows.")

X = final_df[ml_features].values
X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-12)
rank = np.linalg.matrix_rank(X_std)
total_features = len(ml_features)
rank_status = "Full Rank" if rank == total_features else f"Rank Deficient ({rank}/{total_features})"

os.makedirs(OUTPUT_DIR, exist_ok=True)
final_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print(f"SUCCESS: Compiled clean feature matrix with {len(final_df):,} devices.")
print(f"Matrix Rank: {rank} / {total_features} ({rank_status})")
print(f"Saved output to: '{OUTPUT_FILE}'")
print("=" * 70)