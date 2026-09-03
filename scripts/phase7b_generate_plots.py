import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------
# Directory & Style Setup
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output_plots" / "phase7b_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'figure.autolayout': True
})

def run_phase7b_visualization_pipeline():
    # Load all processed data files safely
    df_ml = pd.read_csv(DATA_DIR / "ml_feature_matrix.csv") if (DATA_DIR / "ml_feature_matrix.csv").exists() else None
    df_p4b = pd.read_csv(DATA_DIR / "phase4b_multiparadigm_memristors.csv") if (DATA_DIR / "phase4b_multiparadigm_memristors.csv").exists() else None
    df_synth = pd.read_csv(DATA_DIR / "synthesizable_memristors.csv") if (DATA_DIR / "synthesizable_memristors.csv").exists() else None
    df_sim = pd.read_csv(DATA_DIR / "simulated_device_metrics.csv") if (DATA_DIR / "simulated_device_metrics.csv").exists() else None
    df_bm = pd.read_csv(DATA_DIR / "benchmarked_memristors.csv") if (DATA_DIR / "benchmarked_memristors.csv").exists() else None

    # ---------------------------------------------------------
    # Plot 1: Feature Matrix Screening Density & Top Candidate Overlay
    # ---------------------------------------------------------
    if df_ml is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        hb = ax.hexbin(
            df_ml['Total_Confinement_eV'], df_ml['Dielectric_Contrast_Ratio'], 
            gridsize=50, cmap='Blues', bins='log', alpha=0.65
        )
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label(f'Log10(Count) - {len(df_ml):,} Screened Candidates', fontsize=11, fontweight='bold')

        if df_bm is not None:
            sc = ax.scatter(
                df_bm['Total_Confinement_eV'], df_bm['Dielectric_Contrast_Ratio'], 
                c=df_bm['Composite_Benchmark_Index'], cmap='YlOrRd', edgecolor='black', linewidth=0.5, s=45, zorder=5
            )
            cb2 = fig.colorbar(sc, ax=ax, pad=0.02)
            cb2.set_label('Composite Benchmark Index', fontsize=11, fontweight='bold')

        ax.set_xlabel("Total Band Confinement Energy (eV)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Dielectric Contrast Ratio", fontsize=12, fontweight='bold')
        ax.set_title("01. Screening Feature Landscape Density", fontsize=13, fontweight='bold')
        plt.savefig(OUTPUT_DIR / "01_screening_landscape_density.png", dpi=300)
        plt.close()

  # ---------------------------------------------------------
    # Plot 2: Conduction vs Valence Band Offset Alignment (dEc vs dEv)
    # ---------------------------------------------------------
    if df_ml is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df_ml, x='dEc_eV', y='dEv_eV', hue='Thermodynamic_Driving_Force_eV',
            palette='magma', alpha=0.7, s=40, ax=ax
        )
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel(r"Conduction Band Offset $\Delta E_c$ (eV)", fontsize=12, fontweight='bold')
        ax.set_ylabel(r"Valence Band Offset $\Delta E_v$ (eV)", fontsize=12, fontweight='bold')
        ax.set_title("02. Heterojunction Band Offset Alignment Landscape", fontsize=13, fontweight='bold')
        plt.savefig(OUTPUT_DIR / "02_band_offset_alignment.png", dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # Plot 3: Model Quality Score Boxplots Across Architectures
    # ---------------------------------------------------------
    if df_p4b is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        p4b_cols = ['Consensus_Quality', 'XGB_Quality', 'ET_Quality', 'KNN_Quality', 'MLP_Quality']
        df_p4b_melt = df_p4b[p4b_cols].melt(var_name='Model', value_name='Quality_Score')
        df_p4b_melt['Model'] = df_p4b_melt['Model'].str.replace('_Quality', '')

        sns.boxplot(data=df_p4b_melt, x='Model', y='Quality_Score', palette='muted', ax=ax, width=0.5)
        ax.set_xlabel("Ensemble Architecture", fontsize=12, fontweight='bold')
        ax.set_ylabel("Predicted Quality Score", fontsize=12, fontweight='bold')
        ax.set_title(f"03. Model Ensemble Score Distributions (N={len(df_p4b)})", fontsize=13, fontweight='bold')
        plt.savefig(OUTPUT_DIR / "03_ml_ensemble_distributions.png", dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # Plot 4: Ensemble Verification Agreement Distribution
    # ---------------------------------------------------------
    if df_p4b is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df_p4b['Verification_Agreement_%'], kde=True, bins=20, color='teal', ax=ax)
        ax.axvline(df_p4b['Verification_Agreement_%'].mean(), color='red', linestyle='--', label=f'Mean: {df_p4b["Verification_Agreement_%"].mean():.1f}%')
        ax.set_xlabel("Verification Agreement (%)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Candidate Count", fontsize=12, fontweight='bold')
        ax.set_title("04. Multi-Model Verification Agreement Distribution", fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', frameon=True)
        plt.savefig(OUTPUT_DIR / "04_model_verification_agreement.png", dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # Plot 5: Dual-Tree vs Multi-Paradigm Score Parity
    # ---------------------------------------------------------
    if df_bm is not None:
        fig, ax = plt.subplots(figsize=(9, 7))
        x_val, y_val = df_bm['DualTree_Score'], df_bm['MultiParadigm_Score']
        r_val, _ = pearsonr(x_val, y_val)
        rho_val, _ = spearmanr(x_val, y_val)

        sns.scatterplot(
            data=df_bm, x='DualTree_Score', y='MultiParadigm_Score',
            hue='Master_Confidence_Index_%', palette='viridis', s=60, alpha=0.9, ax=ax
        )
        min_lim = min(x_val.min(), y_val.min()) * 0.98
        max_lim = max(x_val.max(), y_val.max()) * 1.02
        ax.plot([min_lim, max_lim], [min_lim, max_lim], 'r--', linewidth=1.8, label='1:1 Parity Line')

        ax.annotate(
            f"Pearson r = {r_val:.4f}\nSpearman ρ = {rho_val:.4f}",
            xy=(0.05, 0.85), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", lw=0.8)
        )
        ax.set_xlabel("Dual-Tree Score", fontsize=12, fontweight='bold')
        ax.set_ylabel("Multi-Paradigm Score", fontsize=12, fontweight='bold')
        ax.set_title("05. Dual-Tree vs Multi-Paradigm Parity Validation", fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', frameon=True)
        plt.savefig(OUTPUT_DIR / "05_model_parity_validation.png", dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # Plot 6: Synthesizability Index vs Lattice Mismatch
    # ---------------------------------------------------------
    if df_synth is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df_synth, x='Lattice_Mismatch_pct', y='Synthesizability_Index',
            hue='Synthesis_Regime', style='Synthesis_Regime', palette='Dark2', s=65, alpha=0.85, ax=ax
        )
        ax.axhline(0.70, color='crimson', linestyle='--', linewidth=1.5, label='Epitaxial Target Threshold (0.70)')
        ax.set_xlabel("Lattice Mismatch (%)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Synthesizability Index ($S_i$)", fontsize=12, fontweight='bold')
        ax.set_title(f"06. Synthesizability vs Lattice Mismatch (N={len(df_synth)})", fontsize=13, fontweight='bold')
        ax.legend(title="Synthesis Regime", loc='upper right', frameon=True)
        plt.savefig(OUTPUT_DIR / "06_synthesizability_mismatch.png", dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # Plot 7: Strain Decay Factor vs Lattice Mismatch
    # ---------------------------------------------------------
    if df_synth is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df_synth, x='Lattice_Mismatch_pct', y='Strain_Decay_Factor',
            hue='Synthesis_Regime', palette='Set1', s=60, ax=ax
        )
        ax.set_xlabel("Lattice Mismatch (%)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Strain Decay Factor", fontsize=12, fontweight='bold')
        ax.set_title("07. Interfacial Strain Decay Dynamics", fontsize=13, fontweight='bold')
        plt.savefig(OUTPUT_DIR / "07_lattice_mismatch_vs_strain.png", dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # Plot 8: Thermodynamic Hull Energy vs Synthesizability Index
    # ---------------------------------------------------------
    if df_sim is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df_sim, x='Max_Pair_Ehull_eV_atom', y='Synthesizability_Index',
            hue='Thermodynamically_Stable', palette={True: 'green', False: 'darkred'}, s=60, alpha=0.85, ax=ax
        )
        ax.axvline(0.10, color='orange', linestyle='--', label='Stability Bound (0.10 eV/atom)')
        ax.set_xlabel("Max Pair Energy Above Convex Hull (eV/atom)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Synthesizability Index ($S_i$)", fontsize=12, fontweight='bold')
        ax.set_title("08. Thermodynamic Hull Stability vs Synthesizability", fontsize=13, fontweight='bold')
        ax.legend(title="Stable Phase", loc='upper right', frameon=True)
        plt.savefig(OUTPUT_DIR / "08_thermodynamic_stability_ehull.png", dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # Plot 9: Simulated Retention Time vs ON/OFF Ratio
    # ---------------------------------------------------------
    if df_sim is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(
            df_sim['Retention_Time_Years'], df_sim['Simulated_ON_OFF_Ratio_2V'],
            c=df_sim['Design_Thickness_nm'], cmap='plasma', s=60, alpha=0.85, edgecolor='black', linewidth=0.3
        )
        ax.set_yscale('log')
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label('Design Thickness (nm)', fontsize=11, fontweight='bold')
        ax.axhline(1e4, color='red', linestyle='--', label='Target ON/OFF Ratio ($10^4$)')
        ax.axvline(10.0, color='blue', linestyle=':', label='Target Retention (10 Years)')
        ax.set_xlabel("Retention Time (Years)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Simulated ON/OFF Ratio @ 2V (Log Scale)", fontsize=12, fontweight='bold')
        ax.set_title("09. Device Performance: Retention vs ON/OFF Ratio", fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', frameon=True)
        plt.savefig(OUTPUT_DIR / "09_simulated_retention_vs_onoff.png", dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # Plot 10: Current Densities Landscape (J_ON vs J_OFF)
    # ---------------------------------------------------------
    if df_sim is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df_sim['J_OFF_2V_A_m2'], df_sim['J_ON_2V_A_m2'], c='purple', alpha=0.7, edgecolors='none', s=45)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Off-State Current Density $J_{OFF}$ (A/m²)", fontsize=12, fontweight='bold')
        ax.set_ylabel("On-State Current Density $J_{ON}$ (A/m²)", fontsize=12, fontweight='bold')
        ax.set_title("10. Device Switching Current Density Operating Window", fontsize=13, fontweight='bold')
        plt.savefig(OUTPUT_DIR / "10_current_density_landscape.png", dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # Plot 11: Literature Performance Gain Multipliers by Tier
    # ---------------------------------------------------------
    if df_bm is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        gain_cols = ['Confinement_Gain_vs_Lit', 'Synth_Gain_vs_Lit', 'Dielectric_Gain_vs_Lit']
        df_bm_melt = df_bm.melt(id_vars=['Benchmark_Tier'], value_vars=gain_cols, var_name='Domain', value_name='Gain_Factor')
        df_bm_melt['Domain'] = df_bm_melt['Domain'].str.replace('_Gain_vs_Lit', '')

        sns.boxplot(data=df_bm_melt, x='Domain', y='Gain_Factor', hue='Benchmark_Tier', palette='Set2', ax=ax, width=0.5)
        ax.axhline(1.0, color='black', linestyle=':', label='Baseline Literature Reference (1.0x)')
        ax.set_xlabel("Performance Domain", fontsize=12, fontweight='bold')
        ax.set_ylabel("Gain Multiplier vs Literature Reference", fontsize=12, fontweight='bold')
        ax.set_title("11. Performance Gain Multipliers across Benchmark Tiers", fontsize=13, fontweight='bold')
        ax.legend(title="Benchmark Tier", loc='upper left', frameon=True)
        plt.savefig(OUTPUT_DIR / "11_literature_gains_by_tier.png", dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # Plot 12: Master Consensus Score Breakdown across Tiers
    # ---------------------------------------------------------
    if df_bm is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=df_bm, x='Benchmark_Tier', y='Master_Consensus_Score', palette='Set3', inner='quartile', ax=ax)
        ax.set_xlabel("Benchmark Tier Classification", fontsize=12, fontweight='bold')
        ax.set_ylabel("Master Consensus Score", fontsize=12, fontweight='bold')
        ax.set_title("12. Master Consensus Score Distributions across Tiers", fontsize=13, fontweight='bold')
        plt.savefig(OUTPUT_DIR / "12_consensus_score_by_tier.png", dpi=300)
        plt.close()

    print(f"Phase 7B execution complete. Generated 12 plots in '{OUTPUT_DIR.relative_to(PROJECT_ROOT)}'.")

if __name__ == "__main__":
    run_phase7b_visualization_pipeline()