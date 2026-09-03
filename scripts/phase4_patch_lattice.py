import os
import re
import pandas as pd

# Effective ionic/covalent radius lookup table (Å)
ELEMENT_RADII = {
    'O': 1.40, 'F': 1.33, 'S': 1.84, 'Se': 1.98, 'Cl': 1.81, 'Br': 1.96, 'I': 2.20,
    'Ac': 1.12, 'Al': 0.53, 'Ba': 1.35, 'Be': 0.45, 'Bi': 1.03, 'Ca': 1.00, 'Cu': 0.73,
    'Dy': 0.91, 'Gd': 0.93, 'Ho': 0.90, 'In': 0.80, 'Mn': 0.67, 'P': 0.38, 'Si': 0.40,
    'Ti': 0.605, 'Tl': 0.88, 'Tm': 0.87, 'Y': 0.90, 'Zn': 0.74, 'Zr': 0.72, 'B': 0.27
}

def estimate_lattice_a(formula):
    tokens = re.findall(r'([A-Z][a-z]?)([\d\.]*)', str(formula))
    if not tokens:
        return 4.000
    total_r, count = 0.0, 0
    for el, num in tokens:
        n = float(num) if num else 1.0
        r = ELEMENT_RADII.get(el, 0.85)
        total_r += r * n
        count += n
    avg_r = total_r / max(count, 1.0)
    return round(2.0 * avg_r * 1.85, 4)

csv_path = os.path.join("data", "processed", "final_master_verified_memristors.csv")
df = pd.read_csv(csv_path)

if 'Core_a_A' not in df.columns:
    df['Core_a_A'] = df['Core_Formula'].apply(estimate_lattice_a)
if 'Shell_a_A' not in df.columns:
    df['Shell_a_A'] = df['Shell_Formula'].apply(estimate_lattice_a)

df.to_csv(csv_path, index=False)
print(f"Patched '{csv_path}' with Core_a_A and Shell_a_A columns.")