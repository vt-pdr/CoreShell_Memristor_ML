import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

print("=" * 70)
print("[PHASE 6] Advanced Literature Benchmarking & Performance Tiering")
print("=" * 70)

# ==============================================================================
# CONFIGURATION & PARAMETERS
# ==============================================================================
PATHS = {
    "synth": Path("data/processed/synthesizable_memristors.csv"),
    "master": Path("data/processed/final_master_verified_memristors.csv"),
    "output": Path("data/processed/benchmarked_memristors.csv"),
}

# Industry Literature Baselines
BENCHMARKS = {
    'TiO2 / Al2O3': {'confinement': 3.20, 'synthesizability': 0.65, 'dielectric_contrast': 2.10},
    'HfO2 / SiO2':  {'confinement': 4.10, 'synthesizability': 0.70, 'dielectric_contrast': 1.80},
    'ZnO / Al2O3':  {'confinement': 3.50, 'synthesizability': 0.68, 'dielectric_contrast': 2.30}
}

# Metric Weights (Must sum to 1.0)
WEIGHTS = {
    'confinement': 0.40,
    'synthesizability': 0.35,
    'dielectric': 0.25
}

# Performance Tier Criteria
TIER_THRESHOLDS = {
    'tier1_min_cbi': 1.15,
    'tier2_min_cbi': 0.95
}

# Validate Weights Integrity
if not np.isclose(sum(WEIGHTS.values()), 1.0):
    raise ValueError(f"Weights must sum to 1.0. Current sum: {sum(WEIGHTS.values()):.4f}")

# ==============================================================================
# 1. LOAD & VALIDATE INPUT DATA
# ==============================================================================
if not PATHS["synth"].exists():
    raise FileNotFoundError(f"Missing required Phase 5 output: '{PATHS['synth']}'")

df_synth = pd.read_csv(PATHS["synth"])
logging.info(f"Loaded {len(df_synth)} synthesizable candidates from Phase 5.")

# Clean string whitespaces to prevent join mismatches (Fix Flaw 10)
if 'Material_Pair' in df_synth.columns:
    df_synth['Material_Pair'] = df_synth['Material_Pair'].astype(str).str.strip()
else:
    raise KeyError("Input file is missing required column: 'Material_Pair'")

# ==============================================================================
# 2. MERGE MISSING DATA & NA VALUES FROM MASTER FILE
# ==============================================================================
target_cols = ['Total_Confinement_eV', 'Dielectric_Contrast_Ratio', 'Synthesis_Regime']

if PATHS["master"].exists():
    df_master = pd.read_csv(PATHS["master"])
    df_master['Material_Pair'] = df_master['Material_Pair'].astype(str).str.strip()
    
    # Identify available master columns
    master_cols = ['Material_Pair'] + [c for c in target_cols if c in df_master.columns]
    
    # Group master records to preserve valid non-null entries (Fix Flaw 8 & 14)
    df_master_clean = df_master[master_cols].groupby('Material_Pair', as_index=False).first()
    
    # Merge master properties (Fix Flaw 7 & 11)
    df_synth = pd.merge(df_synth, df_master_clean, on='Material_Pair', how='left', suffixes=('', '_master'))
    
    # Fill missing values from master file
    for col in target_cols:
        master_col = f"{col}_master"
        if master_col in df_synth.columns:
            if col in df_synth.columns:
                df_synth[col] = df_synth[col].fillna(df_synth[master_col])
            else:
                df_synth[col] = df_synth[master_col]
            df_synth.drop(columns=[master_col], inplace=True)

# Calculate Reference Literature Averages
avg_ref_conf = float(np.mean([b['confinement'] for b in BENCHMARKS.values()]))
avg_ref_synth = float(np.mean([b['synthesizability'] for b in BENCHMARKS.values()]))
avg_ref_diel = float(np.mean([b['dielectric_contrast'] for b in BENCHMARKS.values()]))

logging.info(
    f"Benchmark Baselines (Avg): Confinement={avg_ref_conf:.2f} eV | "
    f"Synth Index={avg_ref_synth:.2f} | Dielectric Ratio={avg_ref_diel:.2f}"
)

# Dynamic Baseline Imputation for missing values across ALL metrics (Fix Flaw 1 & 12)
impute_map = {
    'Total_Confinement_eV': avg_ref_conf,
    'Dielectric_Contrast_Ratio': avg_ref_diel,
    'Synthesizability_Index': avg_ref_synth
}

for col, ref_val in impute_map.items():
    if col not in df_synth.columns:
        df_synth[col] = ref_val
    null_count = df_synth[col].isnull().sum()
    if null_count > 0:
        logging.warning(f"Imputing {null_count} missing values in '{col}' with literature baseline average ({ref_val:.4f}).")
        df_synth[col] = df_synth[col].fillna(ref_val)

# ==============================================================================
# 3. CALCULATE GAINS & COMPOSITE BENCHMARK INDEX (FULL PRECISION)
# ==============================================================================
# Preserve full precision floats during CBI calculation (Fix Flaw 2)
df_synth['Confinement_Gain_vs_Lit'] = df_synth['Total_Confinement_eV'] / avg_ref_conf
df_synth['Synth_Gain_vs_Lit'] = df_synth['Synthesizability_Index'] / avg_ref_synth
df_synth['Dielectric_Gain_vs_Lit'] = df_synth['Dielectric_Contrast_Ratio'] / avg_ref_diel

df_synth['Composite_Benchmark_Index'] = (
    WEIGHTS['confinement'] * df_synth['Confinement_Gain_vs_Lit'] +
    WEIGHTS['synthesizability'] * df_synth['Synth_Gain_vs_Lit'] +
    WEIGHTS['dielectric'] * df_synth['Dielectric_Gain_vs_Lit']
)

# ==============================================================================
# 4. VECTORIZED TIER CLASSIFICATION (EXACT REGIME MATCHING)
# ==============================================================================
# Standardize regime strings and ensure column assignment (Fix Flaw 13)
if 'Synthesis_Regime' not in df_synth.columns:
    df_synth['Synthesis_Regime'] = ''
else:
    df_synth['Synthesis_Regime'] = df_synth['Synthesis_Regime'].fillna('').astype(str)

regime_series = df_synth['Synthesis_Regime']

# Match coherent regimes while excluding incoherent/high-strain terms (Fix Flaw 6)
is_coherent = regime_series.str.contains('Coherent|Semi-Coherent', case=False, na=False)
is_incoherent = regime_series.str.contains('Incoherent|Non-Coherent|Buffer Required', case=False, na=False)
has_valid_regime = is_coherent & (~is_incoherent)

# Vectorized Tier Assignment (Fix Flaw 3)
tier_conditions = [
    (df_synth['Composite_Benchmark_Index'] >= TIER_THRESHOLDS['tier1_min_cbi']) & has_valid_regime,
    (df_synth['Composite_Benchmark_Index'] >= TIER_THRESHOLDS['tier2_min_cbi'])
]

tier_labels = [
    'Tier 1: Next-Gen Breakthrough',
    'Tier 2: Competitive Alternative'
]

df_synth['Benchmark_Tier'] = np.select(tier_conditions, tier_labels, default='Tier 3: Sub-optimal Baseline')

# ==============================================================================
# 5. SORTING, ROUNDING & EXPORT
# ==============================================================================
# Sort by exact CBI float score BEFORE rounding to preserve rank fidelity (Fix Flaw 15)
df_benchmarked = df_synth.sort_values(by='Composite_Benchmark_Index', ascending=False).reset_index(drop=True)

# Round metrics strictly at export stage
round_cols = {
    'Confinement_Gain_vs_Lit': 3,
    'Synth_Gain_vs_Lit': 3,
    'Dielectric_Gain_vs_Lit': 3,
    'Composite_Benchmark_Index': 4
}
df_benchmarked = df_benchmarked.round(round_cols)

# Ensure output path exists (Fix Flaw 5)
PATHS["output"].parent.mkdir(parents=True, exist_ok=True)
df_benchmarked.to_csv(PATHS["output"], index=False)

print("\n" + "=" * 70)
print(f"SUCCESS! Benchmarked {len(df_benchmarked)} candidates.")
print(f"Results exported to: '{PATHS['output']}'")
print("=" * 70)

# ==============================================================================
# 6. SUMMARY DISPLAY
# ==============================================================================
print("\nPerformance Tier Breakdown:")
print(df_benchmarked['Benchmark_Tier'].value_counts().to_string())

# Filter strictly for Tier 1 Breakthrough Memristors (Fix Flaw 9)
tier1_df = df_benchmarked[df_benchmarked['Benchmark_Tier'] == 'Tier 1: Next-Gen Breakthrough']
print(f"\nTop Next-Gen Breakthrough Memristors (Tier 1 Count: {len(tier1_df)}):")
cols_to_show = [c for c in ['Material_Pair', 'Composite_Benchmark_Index', 'Synthesizability_Index', 'Total_Confinement_eV', 'Synthesis_Regime'] if c in df_benchmarked.columns]

if len(tier1_df) > 0:
    print(tier1_df[cols_to_show].head(5).to_string(index=False))
else:
    print("No candidates satisfied Tier 1 Breakthrough criteria (CBI >= 1.15 + Coherent Regime).")