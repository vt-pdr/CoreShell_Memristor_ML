from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Automatically locate the files within your project folder
project_root = Path.cwd()


def locate_file(filename):
  matches = list(project_root.rglob(filename))
  if not matches:
    raise FileNotFoundError(
        f"Could not locate '{filename}' inside {project_root}"
    )
  return matches[0]


# 2. Load the datasets using located absolute paths
prof_df = pd.read_csv(locate_file('synthetic_rs_dataset_real_eps.csv'))
ml_df = pd.read_csv(locate_file('verified_predictions.csv'))

# 3. Merge on core and shell material pairs
merged_df = pd.merge(
    prof_df, ml_df, on=['core_material', 'shell_material'], how='inner'
)

# 4. Transform raw values to log scale matching ML targets
merged_df['Prof_log_on_off'] = np.log10(merged_df['on_off_ratio'])
merged_df['Prof_log_retention'] = np.log10(merged_df['retention_time_s'])

# 5. Plot actual vs predicted values
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

targets = [
    (
        axes[0],
        merged_df['hysteresis_window_V'],
        merged_df['Pred_hysteresis_window_V'],
        'Hysteresis (V)',
        'teal',
        ' V',
    ),
    (
        axes[1],
        merged_df['Prof_log_on_off'],
        merged_df['Pred_log_on_off_ratio'],
        'log ON/OFF',
        'crimson',
        ' log(ratio)',
    ),
    (
        axes[2],
        merged_df['Prof_log_retention'],
        merged_df['Pred_log_retention_s'],
        'log Retention',
        'goldenrod',
        ' log(s)',
    ),
]

for ax, y_true, y_pred, title, color, unit in targets:
  mae = mean_absolute_error(y_true, y_pred)
  r2 = r2_score(y_true, y_pred)

  ax.scatter(y_true, y_pred, alpha=0.25, color=color, s=15, edgecolors='none')

  min_val = min(y_true.min(), y_pred.min())
  max_val = max(y_true.max(), y_pred.max())
  ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)

  ax.set_title(title, fontweight='bold')
  ax.set_xlabel('Actual [Prof Dataset]', fontweight='bold')
  ax.set_ylabel('Predicted [ML Model]', fontweight='bold')
  ax.text(
      0.05,
      0.88,
      f'MAE: {mae:.4f}{unit}\nR²: {r2:.4f}',
      transform=ax.transAxes,
      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
  )

plt.tight_layout()
plt.savefig('verified_cross_csv_plot.png')
plt.show()