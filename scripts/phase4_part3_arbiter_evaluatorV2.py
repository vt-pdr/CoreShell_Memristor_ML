import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# =====================================================================
# 1. LOAD CSV ARTIFACTS FROM PART 1 AND PART 2
# =====================================================================
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

PROJECT_ROOT = (
    os.path.abspath(os.path.join(script_dir, ".."))
    if "scripts" in script_dir
    else os.getcwd()
)

data_dir = os.path.join(PROJECT_ROOT, "data", "processed")
file_p1 = os.path.join(data_dir, "phase4a_dualtree_memristors.csv")
file_p2 = os.path.join(data_dir, "phase4b_multiparadigm_memristors.csv")

print("==================================================")
print("[PHASE 4 - PART 3] INTEGRATION & ARBITER EVALUATOR")
print("==================================================")
print(f"Loading Part 1 (Dual-Tree)     : '{file_p1}'")
print(f"Loading Part 2 (Multi-Paradigm): '{file_p2}'")

df_p1 = pd.read_csv(file_p1)
df_p2 = pd.read_csv(file_p2)

def get_column(df, variants):
    for var in variants:
        if var in df.columns:
            return var
    return None

def standardize_df(df):
    col_map = {col: col.strip().replace(" ", "_") for col in df.columns}
    df = df.rename(columns=col_map)
    
    core_col = get_column(df, ['Core_Formula', 'Core', 'Core_Composition'])
    shell_col = get_column(df, ['Shell_Formula', 'Shell', 'Shell_Composition'])
    if core_col and core_col != 'Core_Formula':
        df.rename(columns={core_col: 'Core_Formula'}, inplace=True)
    if shell_col and shell_col != 'Shell_Formula':
        df.rename(columns={shell_col: 'Shell_Formula'}, inplace=True)
        
    return df

df_p1 = standardize_df(df_p1)
df_p2 = standardize_df(df_p2)

df_p1['Material_Pair'] = df_p1['Core_Formula'].astype(str) + " / " + df_p1['Shell_Formula'].astype(str)
df_p2['Material_Pair'] = df_p2['Core_Formula'].astype(str) + " / " + df_p2['Shell_Formula'].astype(str)

score_col_p1 = get_column(df_p1, ['Consensus_Quality_DualTree', 'Consensus_Quality', 'DualTree_Score', 'Quality_Score'])
score_col_p2 = get_column(df_p2, ['Consensus_Quality_MultiParadigm', 'Consensus_Quality', 'MultiParadigm_Score', 'Quality_Score'])

df_p1_sorted = df_p1.sort_values(by=score_col_p1, ascending=False) if score_col_p1 else df_p1
df_p2_sorted = df_p2.sort_values(by=score_col_p2, ascending=False) if score_col_p2 else df_p2

# =====================================================================
# 2. EVALUATION METRIC 1: RANK STABILITY (JACCARD & SPEARMAN)
# =====================================================================
print("\n--- 1. RANK STABILITY & OVERLAP ANALYSIS ---")

for k in [50, 100, 250, 500]:
    top_p1 = set(df_p1_sorted.head(k)['Material_Pair'])
    top_p2 = set(df_p2_sorted.head(k)['Material_Pair'])
    intersection = len(top_p1.intersection(top_p2))
    union = len(top_p1.union(top_p2))
    jaccard = intersection / union if union > 0 else 0.0
    # FIX BUG 2: Correct reporting of set intersection vs union
    print(f" -> Top-{k:<3} Overlap : {intersection}/{union} unique pairs in union | Jaccard Index = {jaccard:.3f}")

merged = pd.merge(
    df_p1,
    df_p2,
    on='Material_Pair',
    suffixes=('_DualTree', '_MultiParadigm')
)

q_dt = get_column(merged, ['Consensus_Quality_DualTree', 'Consensus_Quality_x', 'DualTree_Score'])
q_mp = get_column(merged, ['Consensus_Quality_MultiParadigm', 'Consensus_Quality_y', 'MultiParadigm_Score'])
agr_dt = get_column(merged, ['Verification_Agreement_%_DualTree', 'Verification_Agreement_%_x', 'DualTree_Agreement_%'])
agr_mp = get_column(merged, ['Verification_Agreement_%_MultiParadigm', 'Verification_Agreement_%_y', 'MultiParadigm_Agreement_%'])

if len(merged) > 1 and q_dt in merged.columns and q_mp in merged.columns:
    spearman_rho, p_val = spearmanr(merged[q_dt], merged[q_mp])
    print(f"\n -> Cross-Pipeline Spearman Rank Correlation (ρ): {spearman_rho:.4f} (p-val: {p_val:.2e})")
else:
    spearman_rho = 1.0

# =====================================================================
# 3. EVALUATION METRIC 2: SCORE DIVERGENCE & HALLUCINATION DETECTION
# =====================================================================
print("\n--- 2. SCORE DIVERGENCE & MODEL HALLUCINATION DETECTION ---")

merged['Score_Delta'] = np.abs(merged[q_dt] - merged[q_mp])

merged['Hallucination_Flag'] = (
    ((merged[q_dt] >= 0.75) | (merged[q_mp] >= 0.75)) & 
    (merged['Score_Delta'] > 0.12)
)

hallucinated_count = merged['Hallucination_Flag'].sum()
print(f" -> Average Inter-Pipeline Delta Score (Δ) : {merged['Score_Delta'].mean():.4f}")
print(f" -> Tree Boundary Hallucinations Flagged   : {hallucinated_count} candidates")

if hallucinated_count > 0:
    print("\n   Top Flagged Hallucination Candidates (Filtered out of final output):")
    flagged_cols = [c for c in ['Material_Pair', q_dt, q_mp, 'Score_Delta'] if c in merged.columns]
    flagged_df = merged[merged['Hallucination_Flag']][flagged_cols].head(5)
    print(flagged_df.to_string(index=False))

# =====================================================================
# 4. MASTER CONFIDENCE INDEX (MCI) INTEGRATION & PHYSICAL VALIDATION
# =====================================================================
print("\n--- 3. UNIFIED MASTER CANDIDATE SYNTHESIS ---")

merged['Master_Consensus_Score'] = (0.50 * merged[q_dt]) + (0.50 * merged[q_mp])

if agr_dt and agr_mp and agr_dt in merged and agr_mp in merged:
    raw_agr = (merged[agr_dt] + merged[agr_mp]) / 2
    # FIX BUG 4: Normalize agreement percentage scale if in decimal form
    merged['Avg_Model_Agreement_%'] = np.where(raw_agr <= 1.0, raw_agr * 100.0, raw_agr)
else:
    merged['Avg_Model_Agreement_%'] = 100.0

merged['Master_Confidence_Index_%'] = np.clip(
    merged['Avg_Model_Agreement_%'] * (1 - merged['Score_Delta']), 0, 100
)

# FIX BUG 1: Reconstruct unified formula columns before fallback check
merged['Core_Formula'] = merged['Material_Pair'].apply(lambda x: str(x).split(' / ')[0])
merged['Shell_Formula'] = merged['Material_Pair'].apply(lambda x: str(x).split(' / ')[1])

final_master = merged[~merged['Hallucination_Flag']].sort_values(
    by='Master_Consensus_Score', ascending=False
).copy()

lattice_keywords = ['core_a', 'shell_a', 'lattice', 'lat_a']
has_lattice = any(any(k in c.lower() for k in lattice_keywords) for c in final_master.columns)

if not has_lattice:
    for fallback_file in ["phase3_feature_engineered_memristors.csv", "phase2_band_aligned_memristors.csv"]:
        fallback_path = os.path.join(data_dir, fallback_file)
        if os.path.exists(fallback_path):
            fb_df = pd.read_csv(fallback_path)
            fb_df.rename(columns={c: c.strip().replace(" ", "_") for c in fb_df.columns}, inplace=True)
            fb_lat_cols = [c for c in fb_df.columns if any(k in c.lower() for k in lattice_keywords)]
            if fb_lat_cols and 'Core_Formula' in fb_df.columns and 'Shell_Formula' in fb_df.columns:
                sub_fb = fb_df[['Core_Formula', 'Shell_Formula'] + fb_lat_cols].drop_duplicates()
                final_master = final_master.merge(sub_fb, on=['Core_Formula', 'Shell_Formula'], how='left')
                print(f" -> Recovered physical lattice parameters from '{fallback_file}'")
                break

# FIX BUG 5: Physical Validation - Calculate Lattice Mismatch %
core_lat_col = get_column(final_master, ['Core_a_A', 'Core_a', 'Core_Lattice_A', 'Core_a_Å', 'a_core'])
shell_lat_col = get_column(final_master, ['Shell_a_A', 'Shell_a', 'Shell_Lattice_A', 'Shell_a_Å', 'a_shell'])

if core_lat_col and shell_lat_col:
    final_master['Lattice_Mismatch_%'] = (
        np.abs(final_master[core_lat_col] - final_master[shell_lat_col]) / final_master[core_lat_col]
    ) * 100.0
    # Flag unviable epitaxy (> 7.0% mismatch)
    final_master['Unviable_Lattice_Strain'] = final_master['Lattice_Mismatch_%'] > 7.0
    print(f" -> Lattice strain check complete. Unviable strain flagged: {final_master['Unviable_Lattice_Strain'].sum()} pairs.")

# =====================================================================
# 5. EXPORT FINAL MASTER DATASET FOR PHASE 5
# =====================================================================
desired_features = [
    'Core_a_A', 'Shell_a_A', 'Core_a', 'Shell_a', 'Core_Lattice_A', 'Shell_Lattice_A',
    'Core_a_Å', 'Shell_a_Å', 'a_core', 'a_shell', 'Lattice_Mismatch_%',
    'Core_Eg_nano_3nm_eV', 'Shell_Eg_bulk_eV', 'Total_Confinement_eV',
    'Dielectric_Contrast_Ratio', 'Defect_Gradient_eV', 'Confinement_Asymmetry'
]

found_cols = []
for c in list(df_p1.columns) + list(df_p2.columns) + list(final_master.columns):
    c_clean = c.replace('_DualTree', '').replace('_MultiParadigm', '')
    if c_clean in desired_features or any(k in c_clean.lower() for k in lattice_keywords):
        if c_clean not in ['Material_Pair', 'Core_Formula', 'Shell_Formula'] and c_clean not in found_cols:
            found_cols.append(c_clean)

# FIX BUG 3: Robust feature coalescing for both shared and unique single-pipeline columns
for col in found_cols:
    dt_col = f"{col}_DualTree"
    mp_col = f"{col}_MultiParadigm"
    if col in final_master.columns:
        continue
    elif dt_col in final_master.columns and mp_col in final_master.columns:
        final_master[col] = final_master[dt_col].fillna(final_master[mp_col])
    elif dt_col in final_master.columns:
        final_master[col] = final_master[dt_col]
    elif mp_col in final_master.columns:
        final_master[col] = final_master[mp_col]

export_col_map = {
    'Material_Pair': 'Material_Pair',
    'Core_Formula': 'Core_Formula',
    'Shell_Formula': 'Shell_Formula',
    'Master_Consensus_Score': 'Master_Consensus_Score',
    'Master_Confidence_Index_%': 'Master_Confidence_Index_%',
    'Score_Delta': 'Score_Delta',
    q_dt: 'DualTree_Score',
    q_mp: 'MultiParadigm_Score',
    agr_dt: 'DualTree_Agreement_%',
    agr_mp: 'MultiParadigm_Agreement_%'
}

for col in found_cols:
    if col in final_master.columns and col not in export_col_map.values():
        export_col_map[col] = col

export_col_map = {k: v for k, v in export_col_map.items() if k is not None and k in final_master.columns}

final_export = final_master[list(export_col_map.keys())].rename(columns=export_col_map).reset_index(drop=True)

output_master_file = os.path.join(data_dir, "final_master_verified_memristors.csv")
final_export.head(500).to_csv(output_master_file, index=False)

print("\n==================================================")
print("ARBITER EVALUATION COMPLETE!")
print(f"Master Cross-Verified File Saved to:\n '{output_master_file}'")
print("==================================================")
print("\nTop 10 Master Candidates:")
summary_display = [
    c for c in ['Material_Pair', 'Core_Formula', 'Shell_Formula', 'Master_Consensus_Score', 
                'Master_Confidence_Index_%', 'Score_Delta', 'Lattice_Mismatch_%'] 
    if c in final_export.columns
]
print(final_export[summary_display].head(10).to_string(index=False))