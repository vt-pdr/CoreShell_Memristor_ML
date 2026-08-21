import pandas as pd
import numpy as np

# =====================================================================
# 1. LOAD HYBRID VALIDATED CANDIDATES
# =====================================================================
input_file = "hybrid_validated_memristors.csv"
print(f"Loading candidates from '{input_file}'...")

try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"Error: Could not find '{input_file}'. Run Phase 4 first.")
    exit()

print(f"Loaded {len(df):,} candidates for structural verification.")

# =====================================================================
# 2. LATTICE PARAMETER & STRAIN CALCULATION
# =====================================================================
print("Calculating interfacial lattice mismatch (eta)...")

if 'Core_a_A' in df.columns and 'Shell_a_A' in df.columns:
    a_core = df['Core_a_A']
    a_shell = df['Shell_a_A']
else:
    # Pseudo-cubic lattice parameter proxy (Angstroms)
    a_core = 4.2 + (1.2 / (df['Core_Eg_nano_3nm_eV'] + 0.1))
    a_shell = 4.2 + (1.2 / (df['Shell_Eg_bulk_eV'] + 0.1))
    df['Core_a_A'] = a_core
    df['Shell_a_A'] = a_shell

# Percentage strain: eta = (|a_shell - a_core| / a_core) * 100
df['Lattice_Mismatch_pct'] = (np.abs(a_shell - a_core) / a_core) * 100

# =====================================================================
# 3. CONTINUOUS SYNTHESIZABILITY SCORE & TIERING
# =====================================================================
# UPGRADE: Using the dual-model Consensus Quality instead of the old Predicted Quality
df['Synthesizability_Index'] = df['Consensus_Quality'] * np.exp(-df['Lattice_Mismatch_pct'] / 15.0)

# Categorize into nanoscale growth regimes
def categorize_strain(eta):
    if eta <= 6.0:
        return "Coherent Epitaxy (Ideal)"
    elif eta <= 12.0:
        return "Strain-Engineered Shell (Feasible)"
    else:
        return "Requires Buffer Layer"

df['Synthesis_Regime'] = df['Lattice_Mismatch_pct'].apply(categorize_strain)

# Sort by combined Synthesizability Index
final_df = df.sort_values(by='Synthesizability_Index', ascending=False)

# =====================================================================
# 4. EXPORT LAB-READY SHORTLIST
# =====================================================================
output_file = "synthesizable_memristors.csv"
final_df.to_csv(output_file, index=False)

print("\n==================================================")
print(f"SUCCESS! Ranked {len(final_df):,} candidates by Synthesizability Index.")
print(f"Shortlist saved to '{output_file}'.")
print("==================================================")

print("\nTop 5 Lab-Ready Core-Shell Memristors:")
cols = ['Core_Formula', 'Shell_Formula', 'Synthesizability_Index', 'Lattice_Mismatch_pct', 'Synthesis_Regime']
print(final_df[cols].head().to_string(index=False))