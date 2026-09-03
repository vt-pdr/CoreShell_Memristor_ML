import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

MODEL_MAP = {
    "rf": RandomForestRegressor,
    "gb": GradientBoostingRegressor,
    "ridge": Ridge,
    "linear": LinearRegression,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Fully Dynamic Phase 7A Model Diagnostics Pipeline")
    
    # Path & File Configuration
    parser.add_argument("--data-path", type=str, default="data/processed/benchmarked_memristors.csv", help="Path to input CSV dataset")
    parser.add_argument("--output-dir", type=str, default="output_plots/phase7a_plots", help="Directory to save output plots")
    parser.add_argument("--file-prefix", type=str, default="", help="Prefix for output plot filenames")
    
    # Column Specifications
    parser.add_argument("--target-col", type=str, default=None, help="Target column name (auto-detected if omitted)")
    parser.add_argument("--strain-col", type=str, default=None, help="Strain column name")
    parser.add_argument("--strain-keywords", type=str, default="strain,mismatch,lattice", help="Comma-separated keywords for strain matching")
    parser.add_argument("--strain-limit", type=float, default=None, help="Explicit strain limit threshold")
    parser.add_argument("--strain-percentile", type=float, default=0.90, help="Percentile (0.0-1.0) for strain limit fallback")
    
    parser.add_argument("--confinement-col", type=str, default=None, help="Confinement column name")
    parser.add_argument("--confinement-keywords", type=str, default="confinement,d_eg,delta_eg", help="Comma-separated keywords for confinement matching")
    parser.add_argument("--exclude-keys", type=str, default="id,index,sample,unnamed,tier,gain,score,agreement,master,dualtree,multiparadigm", help="Comma-separated keys to exclude leakage/non-feature columns")
    
    # Model Selection & Hyperparameters
    parser.add_argument("--model-type", type=str, choices=["rf", "gb", "ridge", "linear"], default="rf", help="Model architecture")
    parser.add_argument("--n-estimators", type=int, default=100, help="Tree count for ensemble models")
    parser.add_argument("--max-depth", type=int, default=None, help="Max depth for tree models")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    
    # Cross Validation & Evaluation Config
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--cv-scoring", type=str, default="r2", help="Scoring metric for cross-validation")
    
    # Plotting & Styling Controls
    parser.add_argument("--fig-width", type=float, default=8.0, help="Plot figure width in inches")
    parser.add_argument("--fig-height", type=float, default=6.0, help="Plot figure height in inches")
    parser.add_argument("--top-n-features", type=int, default=10, help="Number of features in importance plot")
    parser.add_argument("--theme", type=str, default="whitegrid", help="Seaborn visual theme")
    parser.add_argument("--palette", type=str, default="viridis", help="Seaborn color palette")
    parser.add_argument("--primary-color", type=str, default="royalblue", help="Primary plot color")
    parser.add_argument("--accent-color", type=str, default="crimson", help="Accent plot color")
    parser.add_argument("--dpi", type=int, default=150, help="Image export DPI resolution")
    
    return parser.parse_args()

def resolve_dataset_path(input_path_str: str) -> Path:
    path = Path(input_path_str)
    if path.exists():
        return path
    
    alt_paths = [
        Path.cwd() / input_path_str,
        Path(__file__).resolve().parent.parent / input_path_str,
    ]
    for alt in alt_paths:
        if alt.exists():
            return alt

    # Fallback search for any CSV in workspace if specified path does not exist
    csv_files = list(Path.cwd().glob("*.csv")) + list(Path.cwd().glob("**/*.csv"))
    if csv_files:
        print(f"[!] Target dataset '{input_path_str}' not found. Auto-selecting local CSV: {csv_files[0]}")
        return csv_files[0]

    raise FileNotFoundError(f"Dataset not found at target location: {input_path_str}")

def find_column_by_keywords(df_columns, specified_col, keywords_str, role_name):
    if specified_col:
        if specified_col in df_columns:
            return specified_col
        raise KeyError(f"Specified {role_name} column '{specified_col}' not found in dataset.")
    
    keywords = [k.strip().lower() for k in keywords_str.split(",") if k.strip()]
    for col in df_columns:
        if any(kw in col.lower() for kw in keywords):
            return col
            
    raise ValueError(f"Could not auto-detect {role_name} column using keywords {keywords}. Specify explicitly via --{role_name}-col.")

def resolve_target_column(df, specified_target, numeric_cols):
    if specified_target:
        if specified_target in df.columns:
            return specified_target
        raise KeyError(f"Target column '{specified_target}' not found in dataset columns.")
    
    # Priority keyword search for standard target columns
    target_keywords = ["target", "ratio", "r_off", "on_off", "y", "label", "quality", "conductance"]
    for col in numeric_cols:
        if any(kw in col.lower() for kw in target_keywords):
            print(f"[INFO] Auto-detected target column: '{col}'")
            return col
    
    fallback_col = numeric_cols[-1]
    print(f"[INFO] No --target-col specified. Defaulting to last numeric column: '{fallback_col}'")
    return fallback_col

def instantiate_model(args):
    model_cls = MODEL_MAP[args.model_type]
    if args.model_type in ["rf", "gb"]:
        kwargs = {"n_estimators": args.n_estimators, "random_state": args.random_state}
        if args.max_depth is not None:
            kwargs["max_depth"] = args.max_depth
        return model_cls(**kwargs)
    elif args.model_type == "ridge":
        return model_cls(random_state=args.random_state)
    return model_cls()

def main():
    args = parse_args()
    sns.set_theme(style=args.theme)
    figsize = (args.fig_width, args.fig_height)
    
    csv_path = resolve_dataset_path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.file_prefix}_" if args.file_prefix else ""
    
    df = pd.read_csv(csv_path)

    # 1. Get all numeric columns first
    all_numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if not all_numeric_cols:
        raise ValueError("No valid numeric columns found in the dataset.")

    # 2. Resolve target column BEFORE filtering features
    target_col = resolve_target_column(df, args.target_col, all_numeric_cols)

    # 3. Dynamic Feature Exclusion (Filtering out target proxies, IDs, and meta-scores)
    exclude_keys = {k.strip().lower() for k in args.exclude_keys.split(",") if k.strip()}
    feature_cols = [
        c for c in all_numeric_cols 
        if c != target_col and not any(k in c.lower() for k in exclude_keys)
    ]

    if not feature_cols:
        raise ValueError("No valid feature columns remaining after applying exclusion filters.")

    print(f"[INFO] Target Column: '{target_col}'")
    print(f"[INFO] Physical Feature Columns ({len(feature_cols)}): {feature_cols}")

    # Resolve Domain Specific Columns
    strain_col = find_column_by_keywords(df.columns, args.strain_col, args.strain_keywords, "strain")
    conf_col = find_column_by_keywords(df.columns, args.confinement_col, args.confinement_keywords, "confinement")

    # Clean missing values strictly across all active columns
    required_cols = list(set(feature_cols + [target_col, strain_col, conf_col]))
    clean_df = df[required_cols].dropna()

    X = clean_df[feature_cols]
    y = clean_df[target_col]

    # Model Initialization & Training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    
    model = instantiate_model(args)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Plot 1: Model Parity Validation
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_test, y_pred, alpha=0.7, color=args.primary_color, edgecolor='k', label='Test Set')
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color=args.accent_color, linestyle='--', lw=2, label='Ideal 1:1 Parity')
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    ax.text(0.05, 0.95, f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.4f}", transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_title(f"Model Parity ({args.model_type.upper()})", fontweight='bold')
    ax.set_xlabel(f"Actual {target_col}", fontweight='bold')
    ax.set_ylabel(f"Predicted {target_col}", fontweight='bold')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}01_model_parity_validation.png", dpi=args.dpi)
    plt.close(fig)

    # Plot 2: Residual Error Distribution
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(residuals, kde=True, color=args.accent_color, edgecolor='black', ax=ax)
    ax.axvline(0, color='black', linestyle='--', lw=1.5, label='Zero Error Baseline')
    ax.set_title("Residual Error Distribution", fontweight='bold')
    ax.set_xlabel(r"Residual Error ($y_{\mathrm{test}} - y_{\mathrm{pred}}$)", fontweight='bold')
    ax.set_ylabel("Frequency", fontweight='bold')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}02_residual_error_distribution.png", dpi=args.dpi)
    plt.close(fig)

    # Plot 3: Quality vs Strain Pareto Front
    strain_limit = (
        args.strain_limit 
        if args.strain_limit is not None 
        else float(clean_df[strain_col].quantile(args.strain_percentile))
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=clean_df, x=strain_col, y=target_col, color=args.primary_color, alpha=0.7, label='Candidates', ax=ax)
    ax.axvline(x=strain_limit, color=args.accent_color, linestyle='--', lw=2, label=rf'Threshold Limit ($\leq {strain_limit:.2f}$)')
    ax.set_title(f"{target_col} vs {strain_col} Pareto Front", fontweight='bold')
    ax.set_xlabel(strain_col, fontweight='bold')
    ax.set_ylabel(target_col, fontweight='bold')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}03_quality_vs_strain_pareto.png", dpi=args.dpi)
    plt.close(fig)

    # Plot 4: Feature Importance / Coefficients
    fig, ax = plt.subplots(figsize=figsize)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        imp_type = "Importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        imp_type = "|Coefficient|"
    else:
        importances = np.zeros(len(feature_cols))
        imp_type = "Score"

    feat_imp = pd.DataFrame({'Feature': feature_cols, imp_type: importances}).sort_values(by=imp_type, ascending=False).head(args.top_n_features)
    sns.barplot(data=feat_imp, x=imp_type, y='Feature', palette=args.palette, ax=ax)
    ax.set_title(f"Top {args.top_n_features} Feature Attribution ({args.model_type.upper()})", fontweight='bold')
    ax.set_xlabel(imp_type, fontweight='bold')
    ax.set_ylabel("Feature Name", fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}04_feature_importance_comparison.png", dpi=args.dpi)
    plt.close(fig)

    # Plot 5: Confinement vs Target Quality
    fig, ax = plt.subplots(figsize=figsize)
    sns.regplot(data=clean_df, x=conf_col, y=target_col,
                scatter_kws={'alpha': 0.6, 'color': args.primary_color},
                line_kws={'color': args.accent_color, 'lw': 2}, ax=ax)
    ax.set_title(f"{conf_col} vs {target_col}", fontweight='bold')
    ax.set_xlabel(conf_col, fontweight='bold')
    ax.set_ylabel(target_col, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}05_confinement_vs_quality.png", dpi=args.dpi)
    plt.close(fig)

    # Plot 6: Cross-Validation Stability
    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=args.cv_scoring)
    
    fig, ax = plt.subplots(figsize=figsize)
    folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
    ax.bar(folds, cv_scores, color=args.primary_color, edgecolor='black', alpha=0.85)
    ax.axhline(cv_scores.mean(), color=args.accent_color, linestyle='--', lw=2, label=f'Mean {args.cv_scoring.upper()} = {cv_scores.mean():.3f}')
    ax.set_title(f"{args.cv_folds}-Fold Cross-Validation ({args.cv_scoring.upper()})", fontweight='bold')
    ax.set_xlabel("K-Fold Validation Split", fontweight='bold')
    ax.set_ylabel(f"{args.cv_scoring.upper()} Score", fontweight='bold')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}06_cross_validation_stability.png", dpi=args.dpi)
    plt.close(fig)

    print(f"\n[✓] Diagnostics completed successfully! All plots saved to '{output_dir.resolve()}'")

if __name__ == "__main__":
    main()