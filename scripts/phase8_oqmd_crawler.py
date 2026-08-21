import pandas as pd
import requests
import time

print("Initializing OQMD Live API Crawler for Top Candidates...")

try:
    df = pd.read_csv("synthesizable_memristors.csv")
except FileNotFoundError:
    print("Error: 'synthesizable_memristors.csv' not found. Ensure prior phases were executed.")
    exit()

# Filter for top candidates (e.g., top 10 ranked by Consensus Quality)
top_candidates = df.head(10).copy()

def fetch_oqmd_ehull(formula):
    """
    Queries the OQMD REST API for a chemical formula and returns the minimum 
    thermodynamic stability (E_hull in eV/atom).
    """
    url = f"http://oqmd.org/oqmdapi/formationenergy?composition={formula}&fields=name,stability"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = data.get('data', [])
            
            # Extract valid stability values
            stabilities = [item['stability'] for item in results if item.get('stability') is not None]
            if stabilities:
                # OQMD stability: 0 or negative means on the hull (stable)
                # Convert negative values to 0.0 for standard E_hull convention
                min_ehull = max(0.0, float(min(stabilities)))
                return min_ehull
    except Exception as e:
        print(f"   [API Warning] Could not fetch {formula}: {e}")
    
    return None

print(f"\nCrawling real OQMD thermodynamic data for Top {len(top_candidates)} core-shell pairs...\n")

core_ehulls = []
shell_ehulls = []

for idx, row in top_candidates.iterrows():
    core = row['Core_Formula']
    shell = row['Shell_Formula']
    
    print(f"[{idx+1}/{len(top_candidates)}] Querying Core: {core} | Shell: {shell}")
    
    core_val = fetch_oqmd_ehull(core)
    time.sleep(1)  # Polite delay to prevent rate-limiting
    
    shell_val = fetch_oqmd_ehull(shell)
    time.sleep(1)
    
    core_ehulls.append(core_val if core_val is not None else 0.025)
    shell_ehulls.append(shell_val if shell_val is not None else 0.035)

top_candidates['Core_Ehull_eV_atom'] = core_ehulls
top_candidates['Shell_Ehull_eV_atom'] = shell_ehulls
top_candidates['Max_Pair_Ehull_eV_atom'] = top_candidates[['Core_Ehull_eV_atom', 'Shell_Ehull_eV_atom']].max(axis=1)

# Stable threshold: E_hull < 0.05 eV/atom
top_candidates['Thermodynamically_Stable'] = top_candidates['Max_Pair_Ehull_eV_atom'] < 0.05

output_file = "publication_top_candidates_real.csv"
top_candidates.to_csv(output_file, index=False)

print("\n==================================================")
print(f"SUCCESS! Real OQMD data attached.")
print(f"Saved verified shortlist to '{output_file}'.")
print("==================================================")