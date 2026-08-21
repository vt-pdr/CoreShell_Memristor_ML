import pandas as pd
import numpy as np

# =====================================================================
# 1. LOAD DATASET
# =====================================================================
input_file = "core_shell_parameters_full.csv"
print(f"Loading materials data from {input_file}...")

try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"Error: Could not find '{input_file}'. Run Phase 1 first.")
    exit()

# Drop rows that are missing crucial band alignment data
df = df.dropna(subset=['CBM_bulk_eV', 'VBM_bulk_eV', 'Eg_nano_3nm_eV'])
print(f"Loaded {len(df)} valid materials for pairing.")

# =====================================================================
# 2. GENERATE ALL POSSIBLE PAIRS (CARTESIAN PRODUCT)
# =====================================================================
print("\nCross-referencing all possible Core-Shell combinations...")

# Add prefixes to distinguish Core vs Shell properties
df_core = df.add_prefix('Core_')
df_shell = df.add_prefix('Shell_')

# Create a temporary merge key to force a cartesian product (every combo)
df_core['merge_key'] = 1
df_shell['merge_key'] = 1

# Merge and drop the temporary key
pairs = pd.merge(df_core, df_shell, on='merge_key').drop('merge_key', axis=1)

# Remove identical material pairings (e.g., TiO2 core with TiO2 shell)
pairs = pairs[pairs['Core_Formula'] != pairs['Shell_Formula']]
print(f"Generated {len(pairs):,} unique material pairs.")

# =====================================================================
# 3. CALCULATE BAND OFFSETS & FILTER TYPE-I
# =====================================================================
print("Calculating Conduction and Valence Band Offsets...")

# Conduction Band Offset (CBO) = Shell CBM - Core CBM
pairs['dEc_eV'] = pairs['Shell_CBM_bulk_eV'] - pairs['Core_CBM_bulk_eV']

# Valence Band Offset (VBO) = Core VBM - Shell VBM 
pairs['dEv_eV'] = pairs['Core_VBM_bulk_eV'] - pairs['Shell_VBM_bulk_eV']

print("Filtering for strict Type-I (Straddling) Band Alignment...")
# For Type-I, both offsets must be positive (Shell encompasses Core)
type_1_pairs = pairs[(pairs['dEc_eV'] > 0) & (pairs['dEv_eV'] > 0)].copy()

# Sort by the most confining shells (highest combined barriers)
type_1_pairs['Total_Confinement_eV'] = type_1_pairs['dEc_eV'] + type_1_pairs['dEv_eV']
type_1_pairs = type_1_pairs.sort_values(by='Total_Confinement_eV', ascending=False)

# =====================================================================
# 4. EXPORT RESULTS
# =====================================================================
# Select the most important columns for the final dataset
output_cols = [
    'Core_Formula', 'Shell_Formula', 
    'Core_Eg_nano_3nm_eV', 'Shell_Eg_bulk_eV',
    'Core_Formation_Energy_eV', 'Shell_Formation_Energy_eV',
    'Core_Dielectric_Static', 'Shell_Dielectric_Static',
    'dEc_eV', 'dEv_eV', 'Total_Confinement_eV'
]

final_df = type_1_pairs[output_cols]

output_file = "type_1_memristor_pairs.csv"
final_df.to_csv(output_file, index=False)

print("\n==================================================")
print(f"SUCCESS! Found {len(final_df):,} viable Type-I Core-Shell pairs.")
print(f"Data saved to '{output_file}'.")
print("==================================================")
print("\nTop 5 Most Confining Pairs:")
print(final_df[['Core_Formula', 'Shell_Formula', 'dEc_eV', 'dEv_eV', 'Total_Confinement_eV']].head())