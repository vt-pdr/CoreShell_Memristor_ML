import os
import sys
import pandas as pd

print("Initializing Phase 2 Band Alignment Engine...")
sys.stdout.flush()

# =====================================================================
# 1. PATH RESOLUTION & DYNAMIC CONFIGURATION
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "core_shell_parameters_full.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "type_1_memristor_pairs.csv")

# Minimum physical potential barrier height to prevent room-temperature thermal leak (eV)
MIN_BARRIER_OFFSET = 0.05  

# Effective mass confinement distribution ratio (m_e* = 0.2, m_h* = 0.8)
# Confinement shifts CBM upward by 80% of dEg, VBM downward by 20% of dEg
CBM_CONFINEMENT_RATIO = 0.8  
VBM_CONFINEMENT_RATIO = 0.2  

# =====================================================================
# 2. DATA INGESTION & VALIDATION
# =====================================================================
if not os.path.exists(INPUT_FILE):
    print(f"CRITICAL ERROR: Input file not found at '{INPUT_FILE}'.")
    print("Please ensure Phase 1 generated 'data/raw/core_shell_parameters_full.csv'.")
    sys.exit(1)

print(f"Loading Phase 1 materials dataset from: {INPUT_FILE}")
df_materials = pd.read_csv(INPUT_FILE)
print(f"Total candidate materials loaded: {len(df_materials)}")

# =====================================================================
# 3. TYPE-I BAND ALIGNMENT ENGINE (Nanoscale Corrected)
# =====================================================================
pairs = []
materials_list = df_materials.to_dict('records')

print(f"\nEvaluating Type-I Core-Shell combinations across {len(materials_list)**2} permutations...")

for core in materials_list:
    # 1. Calculate quantum confinement shift for the core
    dEg_core = core['Eg_nano_3nm_eV'] - core['Eg_bulk_eV']
    
    # Shift core band edges dynamically based on quantum confinement
    core_cbm_nano = core['CBM_bulk_eV'] + (dEg_core * CBM_CONFINEMENT_RATIO)
    core_vbm_nano = core['VBM_bulk_eV'] - (dEg_core * VBM_CONFINEMENT_RATIO)

    for shell in materials_list:
        # Prevent self-pairing (same material for core and shell)
        if core['MP_ID'] == shell['MP_ID']:
            continue
            
        # Shell is treated as bulk outer barrier matrix
        shell_cbm = shell['CBM_bulk_eV']
        shell_vbm = shell['VBM_bulk_eV']
        
        # Calculate true nanoscale conduction and valence band offsets
        dEc = shell_cbm - core_cbm_nano
        dEv = core_vbm_nano - shell_vbm
        
        # Strict Type-I Heterojunction condition:
        # Both electron and hole must be trapped inside the core (dEc > 0 and dEv > 0)
        if dEc >= MIN_BARRIER_OFFSET and dEv >= MIN_BARRIER_OFFSET:
            pairs.append({
                "Core_Formula": core['Formula'],
                "Core_MP_ID": core['MP_ID'],
                "Core_Eg_bulk_eV": core['Eg_bulk_eV'],
                "Core_Eg_nano_3nm_eV": core['Eg_nano_3nm_eV'],
                "Core_CBM_nano_eV": round(core_cbm_nano, 4),
                "Core_VBM_nano_eV": round(core_vbm_nano, 4),
                "Core_Dielectric_Static": core['Dielectric_Static'],
                "Core_Dielectric_Source": core.get('Dielectric_Source', 'Unknown'),
                
                "Shell_Formula": shell['Formula'],
                "Shell_MP_ID": shell['MP_ID'],
                "Shell_Eg_bulk_eV": shell['Eg_bulk_eV'],
                "Shell_CBM_bulk_eV": shell['CBM_bulk_eV'],
                "Shell_VBM_bulk_eV": shell['VBM_bulk_eV'],
                "Shell_Dielectric_Static": shell['Dielectric_Static'],
                "Shell_Dielectric_Source": shell.get('Dielectric_Source', 'Unknown'),
                
                "dEc_eV": round(dEc, 4),
                "dEv_eV": round(dEv, 4),
                "Total_Confinement_eV": round(dEc + dEv, 4)
            })

df_pairs = pd.DataFrame(pairs)

# =====================================================================
# 4. EXPORT & SUMMARY
# =====================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
df_pairs.to_csv(OUTPUT_FILE, index=False)

print("=" * 65)
print(f"SUCCESS! Type-I Heterojunction pairing complete.")
print(f"Valid Type-I Pairs Identified: {len(df_pairs)} / {len(materials_list)*(len(materials_list)-1)}")
print(f"Output saved to: '{OUTPUT_FILE}'")
print("=" * 65)