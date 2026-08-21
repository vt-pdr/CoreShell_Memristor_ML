import pandas as pd
import re

# =====================================================================
# 1. LOAD ML FEATURE MATRIX
# =====================================================================
input_file = "ml_feature_matrix.csv"
print(f"Loading '{input_file}' for memristor domain filtering...")

try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"Error: Could not find '{input_file}'. Ensure Phase 3 has run.")
    exit()

initial_count = len(df)

# =====================================================================
# 2. INORGANIC MEMRISTIVE DOMAIN FILTER
# =====================================================================
# Disallowed elements (Hydrates, organics, volatile halides, toxic/unstable salts)
disallowed_elements = {'H', 'Cl', 'Br', 'I', 'F'}

# Required inorganic anions for solid-state memristive switching
valid_anions = {'O', 'N', 'S', 'Se', 'Te', 'P', 'C'}

def is_solid_state_memristor(formula):
    if not isinstance(formula, str):
        return False
    
    # Extract element symbols
    elements = set(re.findall(r'[A-Z][a-z]?', formula))
    
    # Exclude formulas with hydrogen, perchlorates, or volatile halides
    if elements.intersection(disallowed_elements):
        return False
        
    # Must contain at least one valid inorganic anion (Oxide, Nitride, Chalcogenide, Pnictide)
    if not elements.intersection(valid_anions):
        return False
        
    return True

# Apply filtering to both Core and Shell compounds
valid_mask = df['Core_Formula'].apply(is_solid_state_memristor) & \
             df['Shell_Formula'].apply(is_solid_state_memristor)

filtered_df = df[valid_mask].copy()

# =====================================================================
# 3. SAVE FILTERED FEATURE MATRIX
# =====================================================================
output_file = "ml_feature_matrix.csv"  # Overwrites feature matrix with clean data
filtered_df.to_csv(output_file, index=False)

print("\n==================================================")
print(f"SUCCESS! Domain Filtering Complete:")
print(f"  -> Filtered Out : {initial_count - len(filtered_df):,} non-solid-state candidates.")
print(f"  -> Retained     : {len(filtered_df):,} solid-state inorganic pairs.")
print(f"Cleaned dataset saved to '{output_file}'.")
print("==================================================")