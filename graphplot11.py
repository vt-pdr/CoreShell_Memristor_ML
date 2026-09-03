import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------------------
# 1. FILE DISCOVERY
# ---------------------------------------------------------------------
project_root = Path.cwd()


def locate_file(filename):
  matches = list(project_root.rglob(filename))
  if not matches:
    print(
        f"❌ ERROR: Could not locate '{filename}' inside {project_root}",
        file=sys.stderr,
    )
    sys.exit(1)
  return matches[0]


prof_csv_path = locate_file('synthetic_rs_dataset_real_eps.csv')
ml_csv_path = locate_file('verified_predictions.csv')

# ---------------------------------------------------------------------
# 2. DIRECT 1:1 INDEX LOADING (NO PD.MERGE)
# ---------------------------------------------------------------------
prof_df = pd.read_csv(prof_csv_path)
ml_df = pd.read_csv(ml_csv_path)

# Verify length equality to prevent accidental index mismatches
if len(prof_df) != len(ml_df):
  min_len = min(len(prof_df), len(ml_df))
  print(
      f'⚠️ Row count difference detected! Truncating to shortest length:'
      f' {min_len:,}'
  )
  prof_df = prof_df.iloc[:min_len]
  ml_df = ml_df.iloc[:min_len]

# Extract arrays directly by index position to avoid Cartesian product bugs
actual_hysteresis = prof_df['hysteresis_window_V'].values
pred_hysteresis = ml_df['Pred_hysteresis_window_V'].values

actual_log_onoff = np.log10(prof_df['on_off_ratio'].values)
pred_log_onoff = ml_df['Pred_log_on_off_ratio'].values

actual_log_retention = np.log10(prof_df['retention_time_s'].values)
pred_log_retention = ml_df['Pred_log_retention_s'].values

# ---------------------------------------------------------------------
# 3. TERMINAL VERIFICATION LOGGING
# ---------------------------------------------------------------------
print('=' * 75)
print(' 🔍 DATA EXTRACTION & CLEAN ALIGNMENT LOG')
print('=' * 75)
print(f'   Ground Truth File : {prof_csv_path.resolve()}')
print(f'   ML Predictions File: {ml_csv_path.resolve()}')
print(f'   Total Rows Matched : {len(prof_df):,} rows (Direct 1:1 Alignment)')
print('-' * 75)

targets = [
    (
        'Hysteresis Window (V)',
        actual_hysteresis,
        pred_hysteresis,
        'panel_a_hysteresis',
        '#2b7b78',
        ' V',
    ),
    (
        'log ON/OFF Ratio',
        actual_log_onoff,
        pred_log_onoff,
        'panel_b_log_onoff',
        '#801336',
        ' log(ratio)',
    ),
    (
        'log Retention Time (s)',
        actual_log_retention,
        pred_log_retention,
        'panel_c_log_retention',
        '#d4a373',
        ' log(s)',
    ),
]

print(' 📊 ACCURATE METRICS FROM DIRECT 1:1 PAIRING:')
for title, y_true, y_pred, fname, color, unit in targets:
  mae = mean_absolute_error(y_true, y_pred)
  r2 = r2_score(y_true, y_pred)
  print(f'   • {title:<25} -> MAE: {mae:.4f}{unit:<12} | R²: {r2:.4f}')

print('=' * 75)

# ---------------------------------------------------------------------
# 4. PLOTTING & FILE SAVE
# ---------------------------------------------------------------------
output_dir = project_root / 'output_plots'
output_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

for idx, (title, y_true, y_pred, fname, color, unit) in enumerate(targets):
  ax = axes[idx]
  mae = mean_absolute_error(y_true, y_pred)
  r2 = r2_score(y_true, y_pred)

  ax.scatter(y_true, y_pred, color=color, alpha=0.25, edgecolors='none', s=15)

  min_val = min(y_true.min(), y_pred.min())
  max_val = max(y_true.max(), y_pred.max())
  ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='Ideal (y=x)')

  ax.set_xlim(min_val, max_val)
  ax.set_ylim(min_val, max_val)
  ax.set_xlabel(f'Actual {title} [Prof Data]', fontweight='bold')
  ax.set_ylabel(f'Predicted {title} [ML Model]', fontweight='bold')
  ax.set_title(f'({chr(97+idx)}) {title}', fontsize=11, fontweight='bold', pad=8)

  text_str = f'MAE = {mae:.4f}{unit}\n$R^2$ = {r2:.4f}'
  ax.text(
      0.05,
      0.92,
      text_str,
      transform=ax.transAxes,
      fontsize=9,
      verticalalignment='top',
      bbox=dict(
          boxstyle='round,pad=0.4',
          facecolor='white',
          alpha=0.9,
          edgecolor='gray',
      ),
  )

plt.suptitle(
    'Verified Evaluation: Professor Data vs ML Predictions (N ='
    f' {len(prof_df):,})',
    fontsize=13,
    fontweight='bold',
    y=1.02,
)
plt.tight_layout()

combined_png_path = output_dir / 'combined_evaluation_plot.png'
combined_pdf_path = output_dir / 'combined_evaluation_plot.pdf'

plt.savefig(combined_png_path, dpi=300, bbox_inches='tight')
plt.savefig(combined_pdf_path, bbox_inches='tight')
plt.close(fig)

# Save individual panel figures
for idx, (title, y_true, y_pred, fname, color, unit) in enumerate(targets):
  fig_single, ax_single = plt.subplots(figsize=(5.5, 4.5), dpi=300)
  mae = mean_absolute_error(y_true, y_pred)
  r2 = r2_score(y_true, y_pred)

  ax_single.scatter(
      y_true, y_pred, color=color, alpha=0.25, edgecolors='none', s=15
  )
  min_val = min(y_true.min(), y_pred.min())
  max_val = max(y_true.max(), y_pred.max())
  ax_single.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)

  ax_single.set_xlim(min_val, max_val)
  ax_single.set_ylim(min_val, max_val)
  ax_single.set_xlabel(f'Actual {title} [Prof Data]', fontweight='bold')
  ax_single.set_ylabel(f'Predicted {title} [ML Model]', fontweight='bold')
  ax_single.set_title(
      f'({chr(97+idx)}) {title}', fontsize=11, fontweight='bold', pad=8
  )

  text_str = f'MAE = {mae:.4f}{unit}\n$R^2$ = {r2:.4f}'
  ax_single.text(
      0.05,
      0.92,
      text_str,
      transform=ax_single.transAxes,
      fontsize=9,
      verticalalignment='top',
      bbox=dict(
          boxstyle='round,pad=0.4',
          facecolor='white',
          alpha=0.9,
          edgecolor='gray',
      ),
  )

  plt.tight_layout()
  plt.savefig(output_dir / f'{fname}.png', dpi=300, bbox_inches='tight')
  plt.close(fig_single)

print(f' ✅ Plots updated and saved to: {output_dir.resolve()}')