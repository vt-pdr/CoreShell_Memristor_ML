import sys
import pandas as pd
from mp_api.client import MPRester
from scipy.constants import hbar, m_e, e, epsilon_0, pi

print("Initializing Phase 1 Extractor... (Loading Materials Project physics libraries...)")
sys.stdout.flush()

# =====================================================================
# 1. SETUP & CONSTANTS
# =====================================================================
RAW_API_KEY = "5KPRzCoMDKcIqSlSVt73Fq32ZgRMjJMc"
MP_API_KEY = RAW_API_KEY.strip()
MAX_MATERIALS = 1000 

# Average valence plasma energy for standard solid-state semiconductors (~15.0 eV)
HBAR_OMEGA_P = 15.0 

# =====================================================================
# 2. PHYSICAL ESTIMATION ENGINE (Penn's Model)
# =====================================================================
def estimate_dielectric_penn(bandgap_ev):
    """
    Estimates optical/static dielectric constant using Penn's One-Gap Model:
    eps_r ≈ 1 + (hbar * omega_p / Eg)^2
    Provides realistic physical variance for compounds lacking explicit DFPT tensors.
    """
    if pd.isna(bandgap_ev) or bandgap_ev <= 0:
        return 10.0  # Safeguard for degenerate zero-gap cases
    
    eps_penn = 1.0 + (HBAR_OMEGA_P / bandgap_ev) ** 2
    # Cap to physically plausible upper limit for semiconductors (eps_r <= 150)
    return float(min(round(eps_penn, 4), 150.0))

# =====================================================================
# 3. NANOSCALE CORRECTION ENGINE (Brus Equation)
# =====================================================================
def apply_quantum_confinement(row, radius_nm=3.0, m_e_eff=0.2, m_h_eff=0.8):
    if pd.isna(row['Eg_bulk_eV']) or pd.isna(row['Dielectric_Static']):
        return row['Eg_bulk_eV']
        
    R = radius_nm * 1e-9
    eps_r = row['Dielectric_Static']
    m_e_kg, m_h_kg = m_e_eff * m_e, m_h_eff * m_e
    
    confinement_J = (hbar**2 * pi**2) / (2 * R**2) * ((1 / m_e_kg) + (1 / m_h_kg))
    coulomb_J = (1.786 * e**2) / (4 * pi * epsilon_0 * eps_r * R)
    
    Eg_nano = row['Eg_bulk_eV'] + (confinement_J / e) - (coulomb_J / e)
    return max(row['Eg_bulk_eV'], Eg_nano)

# =====================================================================
# 4. OFFICIAL MP-API HARVESTING PIPELINE
# =====================================================================
def fetch_all_stable_semiconductors(limit=MAX_MATERIALS):
    print("\nScanning Materials Project via MPRester for electronic & dielectric data...")
    data_records = []
    
    try:
        with MPRester(MP_API_KEY) as mpr:
            print("  -> Connected! Querying summary database...")
            
            results = mpr.summary.search(
                is_stable=True,
                band_gap=(0.5, 8.0),
                fields=[
                    "material_id", "formula_pretty", "band_gap", 
                    "cbm", "vbm", "formation_energy_per_atom", "has_props"
                ]
            )
            
            total_found = len(results)
            target_count = min(total_found, limit)
            top_docs = results[:target_count]
            print(f"  -> Retrieved {total_found} candidate materials! Processing top {target_count}...\n")
            
            # Cross-reference available dielectric tensors
            print("  -> Querying dielectric tensor endpoint...")
            mp_ids_with_diel = [
                str(doc.material_id) for doc in top_docs 
                if doc.has_props and "dielectric" in doc.has_props
            ]
            
            dielectric_map = {}
            if mp_ids_with_diel:
                try:
                    diel_docs = mpr.dielectric.search(material_ids=mp_ids_with_diel)
                    for d in diel_docs:
                        try:
                            e_tot = getattr(d, 'e_total', getattr(d, 'total', None))
                            if isinstance(e_tot, list) and len(e_tot) == 3 and isinstance(e_tot[0], list):
                                dielectric_map[str(d.material_id)] = (e_tot[0][0] + e_tot[1][1] + e_tot[2][2]) / 3.0
                            elif isinstance(e_tot, (int, float)):
                                dielectric_map[str(d.material_id)] = float(e_tot)
                        except Exception:
                            continue
                except Exception as e:
                    print(f"  -> Warning: Failed to batch-fetch dielectrics: {e}")
            
            count = 0
            for doc in top_docs:
                mp_id_str = str(doc.material_id)
                
                # Dynamic Physical Ingestion: MP API DFPT vs. Penn Model Calculation
                if mp_id_str in dielectric_map:
                    eps_static = dielectric_map[mp_id_str]
                    diel_source = "MP_DFPT_API"
                else:
                    eps_static = estimate_dielectric_penn(doc.band_gap)
                    diel_source = "Penn_Model_Calculated"
                
                data_records.append({
                    "Formula": doc.formula_pretty,
                    "MP_ID": mp_id_str,
                    "Eg_bulk_eV": doc.band_gap,
                    "CBM_bulk_eV": doc.cbm,
                    "VBM_bulk_eV": doc.vbm,
                    "Formation_Energy_eV": doc.formation_energy_per_atom,
                    "Dielectric_Static": eps_static,
                    "Dielectric_Source": diel_source
                })
                count += 1
                
                sys.stdout.write(f"\r  -> Assembling Material Matrix: {count}/{target_count} ")
                sys.stdout.flush()

            print() 

    except Exception as ex:
        print(f"\n  -> Critical API Error: {ex}")
            
    return pd.DataFrame(data_records)

# =====================================================================
# 5. EXECUTION PIPELINE
# =====================================================================
if __name__ == "__main__":
    df = fetch_all_stable_semiconductors(limit=MAX_MATERIALS)
    
    if not df.empty:
        print("\nApplying quantum confinement corrections (Brus Equation)...")
        df["Eg_nano_3nm_eV"] = df.apply(apply_quantum_confinement, axis=1)
        
        output_file = "core_shell_parameters_full.csv"
        df.to_csv(output_file, index=False)
        
        print("=" * 60)
        print(f"SUCCESS! Re-engineered dataset saved to '{output_file}'.")
        print(f"Total Materials Processed: {len(df)}")
        print(f"Dielectric Sources Breakdown:")
        print(df['Dielectric_Source'].value_counts().to_string())
        print("=" * 60)