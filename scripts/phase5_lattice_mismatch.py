import os
import re
import sys
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# =====================================================================
# DYNAMIC ENVIRONMENT & DATA PATH RESOLUTION
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
PROJECT_ROOT = (
    os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    if "scripts" in SCRIPT_DIR
    else SCRIPT_DIR
)

data_dir = os.path.join(PROJECT_ROOT, "data", "processed")
primary_input = os.path.join(data_dir, "final_master_verified_memristors.csv")
fallback_input = os.path.join(data_dir, "phase4b_multiparadigm_memristors.csv")

if os.path.exists(primary_input):
    input_file = primary_input
    print(f"[PHASE 5] Loading candidates from primary output: '{input_file}'")
elif os.path.exists(fallback_input):
    input_file = fallback_input
    print(f"[PHASE 5] [WARNING] Primary input not found. Falling back to: '{input_file}'")
else:
    print(f"[ERROR] Could not find Phase 4 output files in '{data_dir}'. Run Phase 4 first.")
    sys.exit(1)

try:
    df = pd.read_csv(input_file)
except Exception as e:
    print(f"[ERROR] Failed to read input file '{input_file}': {e}")
    sys.exit(1)

initial_count = len(df)
if initial_count == 0:
    print("[ERROR] Input dataset is empty. Pipeline terminated.")
    sys.exit(1)

print(f" -> Loaded {initial_count:,} candidates from Phase 4 pipeline.")

eps = np.finfo(float).eps

# Dynamic Domain Constants (Semiconductor Epitaxy Standard Boundaries)
COHERENT_THRESHOLD_PCT = 2.0
SEMICOHERENT_THRESHOLD_PCT = 5.0

# =====================================================================
# 1. FUZZY & METADATA-SAFE COLUMN RESOLUTION
# =====================================================================
def resolve_column(df_columns, primary_candidates, keyword_patterns, exclude_patterns=None):
    if exclude_patterns is None:
        exclude_patterns = []
        
    # 1. Direct candidate matching
    for cand in primary_candidates:
        if cand in df_columns:
            return cand

    # 2. Case-insensitive & normalized regex pattern matching
    for col in df_columns:
        col_norm = col.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("å", "a")
        if any(re.search(ex, col_norm) for ex in exclude_patterns):
            continue
        if any(re.search(pat, col_norm) for pat in keyword_patterns):
            return col
    return None

print(" -> Resolving physical columns and performing metadata-safe k-NN imputation...")

core_lat_col = resolve_column(
    df.columns,
    ['Core_a_A', 'Core_Lattice_A', 'a_core', 'Core_a', 'Core_a_Å', 'Core_a_Angstrom', 'a_core_A'],
    [r'core_a', r'a_core', r'core.*lattice', r'core.*lat', r'core.*angstrom'],
    exclude_patterns=[r'formula', r'composition', r'element', r'name', r'id']
)

shell_lat_col = resolve_column(
    df.columns,
    ['Shell_a_A', 'Shell_Lattice_A', 'a_shell', 'Shell_a', 'Shell_a_Å', 'Shell_a_Angstrom', 'a_shell_A'],
    [r'shell_a', r'a_shell', r'shell.*lattice', r'shell.*lat', r'shell.*angstrom'],
    exclude_patterns=[r'formula', r'composition', r'element', r'name', r'id']
)

quality_col = resolve_column(
    df.columns,
    ['Master_Consensus_Score', 'Consensus_Quality', 'XGB_Quality'],
    [r'master_consensus', r'consensus', r'xgb_quality', r'quality_score']
)

conf_col = resolve_column(
    df.columns,
    ['Master_Confidence_Index_%', 'Confidence_Index', 'Confidence'],
    [r'master_confidence', r'confidence_index', r'confidence']
)

conf_eV_col = resolve_column(
    df.columns,
    ['Total_Confinement_eV', 'Confinement_eV'],
    [r'total_confinement', r'confinement_ev', r'confinement']
)

target_cols = [c for c in [core_lat_col, shell_lat_col, quality_col, conf_col, conf_eV_col] if c is not None]

if len(target_cols) < 5:
    print("\n[ERROR] Missing essential physical columns.")
    print(f"Matched ({len(target_cols)}/5): {target_cols}")
    print("Missing targets:")
    if not core_lat_col: print(" - Core Lattice Parameter column")
    if not shell_lat_col: print(" - Shell Lattice Parameter column")
    if not quality_col: print(" - Quality/Consensus Score column")
    if not conf_col: print(" - Confidence Index column")
    if not conf_eV_col: print(" - Confinement Energy column")
    print(f"\nAvailable CSV Headers in '{input_file}':\n{list(df.columns)}")
    sys.exit(1)

print(f" -> Column Mapping Resolved: Core Lattice='{core_lat_col}', Shell Lattice='{shell_lat_col}'")

# Coerce target columns to numeric
for col in target_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Exclude non-physical metadata/ID/index patterns from KNN distance matrix
metadata_patterns = ['id', 'index', 'unnamed', 'year', 'phase', 'formula', 'regime', 'label']
physical_numeric_cols = [
    c for c in df.select_dtypes(include=[np.number]).columns 
    if not any(pattern in c.lower() for pattern in metadata_patterns)
]

for col in target_cols:
    if col not in physical_numeric_cols:
        physical_numeric_cols.append(col)

numeric_df = df[physical_numeric_cols].copy()
numeric_df = numeric_df.dropna(how='all', axis=1)

missing_count = df[target_cols].isna().sum().sum()

if missing_count > 0:
    k_neighbors = int(np.clip(np.sqrt(len(df)), 1, 5))
    print(f" -> Imputing {missing_count} missing values using distance-weighted k-NN (k={k_neighbors})...")
    
    n_min = numeric_df.min()
    n_max = numeric_df.max()
    denom = np.where((n_max - n_min) > eps, n_max - n_min, 1.0)
    norm_numeric = (numeric_df - n_min) / denom
    
    imputer = KNNImputer(n_neighbors=k_neighbors, weights='distance')
    imputed_array = imputer.fit_transform(norm_numeric)
    
    imputed_df = pd.DataFrame(imputed_array, columns=numeric_df.columns, index=numeric_df.index)
    denorm_df = (imputed_df * denom) + n_min
    
    for col in target_cols:
        df[col] = denorm_df[col]

# Enforce strict physical non-negativity boundary on lattice constants
df[core_lat_col] = df[core_lat_col].clip(lower=eps)
df[shell_lat_col] = df[shell_lat_col].clip(lower=eps)

print(f" -> Physical recovery completed. Retained 100% of candidates ({len(df):,} / {initial_count:,}).")

a_core = df[core_lat_col]
a_shell = df[shell_lat_col]

# =====================================================================
# 2. PHYSICAL INTERFACIAL STRAIN & RELAXATION MODELING
# =====================================================================
print(" -> Modeling elastic strain accommodation (Matthews-Blakeslee kinetics)...")

df['Lattice_Mismatch_pct'] = (np.abs(a_shell - a_core) / np.maximum(a_core, eps)) * 100.0

mismatch_pos = df['Lattice_Mismatch_pct'][df['Lattice_Mismatch_pct'] > 0]
eta_0 = mismatch_pos.median() if not mismatch_pos.empty else df['Lattice_Mismatch_pct'].mean()
eta_0 = max(float(eta_0), eps)

df['Strain_Decay_Factor'] = np.exp(-df['Lattice_Mismatch_pct'] / eta_0)

# =====================================================================
# 3. 4-CRITERIA MCDM DECISION MATRIX & ENTROPY WEIGHT METHOD (EWM)
# =====================================================================
def minmax_scale(series):
    s_min, s_max = series.min(), series.max()
    if s_max - s_min > eps:
        return (series - s_min) / (s_max - s_min)
    return pd.Series(1.0, index=series.index)

c_ml = minmax_scale(df[quality_col])
c_conf = minmax_scale(df[conf_col])
c_strain = df['Strain_Decay_Factor']
c_eV = minmax_scale(df[conf_eV_col])

env_w_ml = os.getenv('WEIGHT_ML_QUALITY')
env_w_conf_idx = os.getenv('WEIGHT_ML_CONFIDENCE')
env_w_strain = os.getenv('WEIGHT_PHYSICAL_STRAIN')
env_w_conf_eV = os.getenv('WEIGHT_ELECTRONIC_CONFINEMENT')

if env_w_ml and env_w_conf_idx and env_w_strain and env_w_conf_eV:
    w_raw = np.array([float(env_w_ml), float(env_w_conf_idx), float(env_w_strain), float(env_w_conf_eV)])
    weights = w_raw / np.maximum(np.sum(w_raw), eps)
else:
    X = np.column_stack([c_ml.values, c_conf.values, c_strain.values, c_eV.values])
    n_samples, n_features = X.shape

    if n_samples > 1:
        col_min = np.min(X, axis=0, keepdims=True)
        col_max = np.max(X, axis=0, keepdims=True)
        r_diff = col_max - col_min

        r_matrix = np.zeros_like(X)
        for j in range(n_features):
            if r_diff[0, j] > eps:
                r_matrix[:, j] = (X[:, j] - col_min[0, j]) / r_diff[0, j]
            else:
                r_matrix[:, j] = 0.0

        r_sums = np.sum(r_matrix, axis=0, keepdims=True)
        p_matrix = np.zeros_like(r_matrix)
        for j in range(n_features):
            if r_sums[0, j] > eps:
                p_matrix[:, j] = r_matrix[:, j] / r_sums[0, j]

        k = 1.0 / np.log(n_samples)
        entropy = np.zeros(n_features)
        for j in range(n_features):
            if r_diff[0, j] <= eps or r_sums[0, j] <= eps:
                entropy[j] = 1.0
            else:
                p_col = p_matrix[:, j]
                p_pos = p_col[p_col > 0]
                entropy[j] = -k * np.sum(p_pos * np.log(p_pos))

        utility = 1.0 - entropy
        sum_utility = np.sum(utility)
        weights = utility / sum_utility if sum_utility > eps else np.ones(n_features) / float(n_features)
    else:
        weights = np.ones(n_features) / float(n_features)

df['Synthesizability_Index'] = (
    (weights[0] * c_ml) +
    (weights[1] * c_conf) +
    (weights[2] * c_strain) +
    (weights[3] * c_eV)
)

print(f" -> Derived EWM Weights (ML Quality / Confidence / Strain Accommodation / Confinement): "
      f"{weights[0]:.4f} / {weights[1]:.4f} / {weights[2]:.4f} / {weights[3]:.4f}")

# =====================================================================
# 4. EPITAXIAL GROWTH REGIME CLASSIFICATION
# =====================================================================
conditions = [
    df['Lattice_Mismatch_pct'] <= COHERENT_THRESHOLD_PCT,
    (df['Lattice_Mismatch_pct'] > COHERENT_THRESHOLD_PCT) & (df['Lattice_Mismatch_pct'] <= SEMICOHERENT_THRESHOLD_PCT),
    df['Lattice_Mismatch_pct'] > SEMICOHERENT_THRESHOLD_PCT
]

choices = [
    "Coherent Epitaxy (Low Strain Risk)",
    "Semi-Coherent (Moderate Strain Risk)",
    "Requires Buffer Layer (High Dislocation Risk)"
]

df['Synthesis_Regime'] = np.select(conditions, choices, default="Unclassified")
final_df = df.sort_values(by='Synthesizability_Index', ascending=False)

# =====================================================================
# 5. EXPORT & SUMMARY REPORTING
# =====================================================================
output_file = os.path.join(data_dir, "synthesizable_memristors.csv")
final_df.to_csv(output_file, index=False)

print("\n==================================================")
print(f"SUCCESS! Ranked {len(final_df):,} candidates by Synthesizability Index.")
print(f"Shortlist saved to: '{output_file}'")
print("==================================================")

regime_counts = final_df['Synthesis_Regime'].value_counts()
print("\nEpitaxial Growth Regime Breakdown:")
for regime, count in regime_counts.items():
    pct = (count / len(final_df)) * 100.0 if len(final_df) > 0 else 0.0
    print(f" -> {regime:<45}: {count:5d} candidates ({pct:.1f}%)")

preferred_cols = ['Core_Formula', 'Shell_Formula', 'Synthesizability_Index', 'Lattice_Mismatch_pct', 'Synthesis_Regime']
display_cols = [c for c in preferred_cols if c in final_df.columns]

preview_count = min(len(final_df), 10)
print(f"\nTop {preview_count} Lab-Ready Core-Shell Memristors:")
print(final_df[display_cols].head(preview_count).to_string(index=False))