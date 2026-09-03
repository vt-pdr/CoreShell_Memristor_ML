import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------------------
# 1. FILE DISCOVERY & PATH VERIFICATION
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

print('=' * 75)
print(' 🔍 DATA EXTRACTION & VERIFICATION LOG')
print('=' * 75)
print(f'   Project Root Directory : {project_root}')
print(f'   [1/2] Ground Truth File: {prof_csv_path.resolve()}')
print(f'   [2/2] ML Predictions File: {ml_csv_path.resolve()}')
print('-' * 75)

# ---------------------------------------------------------------------
# 2. LOAD DATASETS & VERIFY SHAPES
# ---------------------------------------------------------------------
prof_df = pd.read_csv(prof_csv_path)
ml_df = pd.read_csv(ml_csv_path)

print(
    f'   ✓ Loaded Raw Professor Dataset  : {prof_df.shape[0]:,} rows ×'
    f' {prof_df.shape[1]} columns'
)
print(
    f'   ✓ Loaded ML Predictions Dataset : {ml_df.shape[0]:,} rows ×'
    f' {ml_df.shape[1]} columns'
)

# Verify 1:1 Index Alignment
material_match = (
    prof_df['core_material'] == ml_df['core_material']
).all() and (prof_df['shell_material'] == ml_df['shell_material']).all()
print(
    f"   ✓ Core/Shell Row Alignment Match : {'EXACT 1:1 MATCH' if material_match else 'MISMATCH DETECTED'}"
)

print('\n 📄 SAMPLE EXTRACTED DATA (First 3 Rows from Disk):')
sample_table = pd.DataFrame({
    'Core': prof_df['core_material'].head(3),
    'Shell': prof_df['shell_material'].head(3),
    'Actual Hyst (V)': prof_df['hysteresis_window_V'].head(3),
    'Pred Hyst (V)': ml_df['Pred_hysteresis_window_V'].head(3),
})
print(sample_table.to_string(index=False))
print('-' * 75)

# ---------------------------------------------------------------------
# 3. DIRECT ARRAY EXTRACTION & LIVE METRIC COMPUTATION
# ---------------------------------------------------------------------
actual_hysteresis = prof_df['hysteresis_window_V'].values
pred_hysteresis = ml_df['Pred_hysteresis_window_V'].values

actual_log_onoff = np.log10(prof_df['on_off_ratio'].values)
pred_log_onoff = ml_df['Pred_log_on_off_ratio'].values

actual_log_retention = np.log10(prof_df['retention_time_s'].values)
pred_log_retention = ml_df['Pred_log_retention_s'].values

print(' 📊 LIVE COMPUTED METRICS FROM EXTRACTED CSV ARRAYS:')

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

metrics_summary = []
for title, y_true, y_pred, fname, color, unit in targets:
  mae = mean_absolute_error(y_true, y_pred)
  r2 = r2_score(y_true, y_pred)
  metrics_summary.append({
      'Metric': title,
      'MAE': f'{mae:.4f}{unit}',
      'R² Score': f'{r2:.4f}',
  })
  print(f'   • {title:<25} -> MAE: {mae:.4f}{unit:<12} | R²: {r2:.4f}')

print('-' * 75)

# ---------------------------------------------------------------------
# 4. PLOTTING & FILE OUTPUT
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

# Save Individual Panel Figures
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

print(' 📁 SAVED PLOTS TO DISK:')
print(f'   • Combined PNG : {combined_png_path.resolve()}')
print(f'   • Combined PDF : {combined_pdf_path.resolve()}')
print(f'   • Output Dir   : {output_dir.resolve()}')
print('=' * 75)