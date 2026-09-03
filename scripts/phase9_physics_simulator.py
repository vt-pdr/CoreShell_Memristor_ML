import os
import sys
import argparse
import itertools
from typing import List, Optional, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================================================================================
# PHYSICAL CONSTANTS & DIELECTRIC LOOKUP TABLE (SI Units)
# =================================================================================
Q_ELEM: float = 1.602176634e-19         # Elementary charge (C)
M_ELEM: float = 9.1093837015e-31        # Free electron mass (kg)
HBAR: float = 1.054571817e-34           # Reduced Planck constant (J s)
H_PLANCK: float = 2.0 * np.pi * HBAR    # Planck constant (J s)
K_BOLTZ: float = 1.380649e-23           # Boltzmann constant (J/K)
EPS_0: float = 8.8541878128e-12         # Vacuum permittivity (F/m)

# Permittivity Empirical Bounds & Scaling Parameters
EPS_R_MIN: float = 2.5
EPS_R_MAX: float = 100.0
EPS_R_DEFAULT: float = 10.0
BANDGAP_SCALE_COEFF: float = 15.0

# Phonon Attempt Frequency Dynamics
PHONON_ATTEMPT_FREQ_DEFAULT: float = 1e13  # 10 THz lattice vibration frequency
TAU_0_DEFAULT: float = 1.0 / PHONON_ATTEMPT_FREQ_DEFAULT  # 1e-13 s attempt time

# Standard Oxide Relative Permittivity Lookup Table
OXIDE_EPS_R_MAP: Dict[str, float] = {
    'hfo2-tio2': 40.0,
    'hfo2/tio2': 40.0,
    'hfo2': 25.0,
    'tio2': 80.0,
    'ta2o5': 22.0,
    'al2o3': 9.0,
    'sio2': 3.9,
    'zro2': 25.0,
    'nio': 11.8,
    'zno': 8.5,
}


# =================================================================================
# TYPE-SAFE DATA CONVERTERS & COLUMN RESOLUTION
# =================================================================================
def safe_float(val: Any, default: float) -> float:
    """Converts input value to float, safely handling NaNs, None, infs, and parsing errors."""
    if val is None or pd.isna(val):
        return default
    try:
        parsed = float(val)
        if np.isnan(parsed) or np.isinf(parsed):
            return default
        return parsed
    except (ValueError, TypeError):
        return default


def resolve_column(
    df: pd.DataFrame, 
    candidate_names: List[str]
) -> Tuple[Optional[str], Optional[pd.Series]]:
    """Finds the first matching column name in a DataFrame (case-insensitive)."""
    df_cols_lower = {str(col).strip().lower(): col for col in df.columns}
    for cand in candidate_names:
        cand_clean = str(cand).strip().lower()
        if cand_clean in df_cols_lower:
            matched_name = df_cols_lower[cand_clean]
            return matched_name, df[matched_name]
            
    return None, None


def estimate_eps_r(
    material_name: str, 
    eg_shell: float,
    eps_map: Optional[Dict[str, float]] = None
) -> float:
    """
    Estimates relative permittivity using normalized oxide lookup table
    or empirical bandgap scaling: eps_r ~ BANDGAP_SCALE_COEFF / sqrt(Eg).
    """
    mapping = OXIDE_EPS_R_MAP if eps_map is None else eps_map
    mat_norm = str(material_name).lower().replace('/', '-').replace('_', '-').replace(' ', '-')
    
    # Sort keys by length descending to match composite oxides before single oxides
    sorted_keys = sorted(mapping.keys(), key=lambda k: len(str(k)), reverse=True)
    for ox_key in sorted_keys:
        key_norm = str(ox_key).lower().replace('/', '-').replace('_', '-').replace(' ', '-')
        if key_norm in mat_norm:
            return float(mapping[ox_key])
            
    if eg_shell > 0:
        scaled_eps = BANDGAP_SCALE_COEFF / np.sqrt(eg_shell)
        return float(np.clip(scaled_eps, EPS_R_MIN, EPS_R_MAX))
        
    return EPS_R_DEFAULT


def generate_synthetic_candidates() -> pd.DataFrame:
    """Generates fallback dataset if no local candidate CSV files are detected."""
    return pd.DataFrame([
        {
            'Material_Pair': 'Pt/HfO2/TiN',
            'Shell_Eg_bulk_eV': 5.7,
            'Total_Confinement_eV': 1.8,
            'Barrier_Height_eV': 2.2,
            'Effective_Mass_Ratio': 0.3,
            'Eps_r': 25.0,
            'Design_Thickness_nm': 5.0,
            'Design_Trap_Depth_eV': 1.2
        },
        {
            'Material_Pair': 'Au/TiO2/FTO',
            'Shell_Eg_bulk_eV': 3.2,
            'Total_Confinement_eV': 1.4,
            'Barrier_Height_eV': 1.6,
            'Effective_Mass_Ratio': 0.4,
            'Eps_r': 80.0,
            'Design_Thickness_nm': 4.5,
            'Design_Trap_Depth_eV': 1.0
        },
        {
            'Material_Pair': 'Pd/Ta2O5/Ta',
            'Shell_Eg_bulk_eV': 4.4,
            'Total_Confinement_eV': 1.6,
            'Barrier_Height_eV': 1.9,
            'Effective_Mass_Ratio': 0.35,
            'Eps_r': 22.0,
            'Design_Thickness_nm': 6.0,
            'Design_Trap_Depth_eV': 1.4
        }
    ])


def load_candidate_dataset(
    input_path: Optional[str] = None,
    custom_search_paths: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, str]:
    """Loads candidate data dynamically from user path, custom paths, or auto-discovery."""
    search_paths = []
    if input_path:
        search_paths.append(input_path)
        
    if custom_search_paths:
        search_paths.extend(custom_search_paths)
        
    search_paths.extend([
        os.path.join("data", "final", "publication_top_candidates_real.csv"),
        os.path.join("data", "processed", "publication_top_candidates_real.csv"),
        os.path.join("data", "processed", "oqmd_verified_memristors.csv"),
        "publication_top_candidates_real.csv",
        "oqmd_verified_memristors.csv"
    ])

    for path in search_paths:
        if path and os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    print(f"-> Loaded dataset from '{path}' ({len(df)} candidates)")
                    return df, path
            except Exception as err:
                print(f"-> Warning: Failed reading file '{path}': {err}")

    print("-> Notice: No input CSV file located. Initializing dynamic synthetic candidates.")
    return generate_synthetic_candidates(), "Synthetic_Generated"


# =================================================================================
# PHYSICAL TRANSPORT MODELS (SIMMONS & POOLE-FRENKEL)
# =================================================================================
def compute_simmons_vectorized(
    v_vec: np.ndarray,
    d_m: float,
    phi_b_eV: float,
    m_eff_kg: float,
    delta_v: Optional[float] = None,
    low_field_threshold_ratio: float = 0.1
) -> np.ndarray:
    """Computes Simmons quantum tunneling current density (A/m^2)."""
    v_vec = np.atleast_1d(v_vec)
    v_abs = np.abs(v_vec)
    v_sign = np.sign(v_vec)
    j_out = np.zeros_like(v_vec, dtype=np.float64)

    nz_mask = v_abs > 1e-12
    if not np.any(nz_mask):
        return j_out

    v_nz = v_abs[nz_mask]
    d_m_safe = max(d_m, 1e-10)
    e_field = v_nz / d_m_safe
    phi_b_J = max(phi_b_eV, 0.01) * Q_ELEM

    kappa_0 = np.sqrt(2.0 * m_eff_kg * phi_b_J) / HBAR
    j0_base = (Q_ELEM**2) / (4.0 * (np.pi**2) * HBAR * (d_m_safe**2))
    
    phi_eff1 = np.maximum(phi_b_eV - v_nz / 2.0, 1e-4)
    phi_eff2 = phi_b_eV + v_nz / 2.0

    kappa1 = np.sqrt(2.0 * m_eff_kg * phi_eff1 * Q_ELEM) / HBAR
    kappa2 = np.sqrt(2.0 * m_eff_kg * phi_eff2 * Q_ELEM) / HBAR

    # Strict physical upper bound clipping <= 0.0 for probability exponent
    exp_fwd = np.clip(-2.0 * kappa1 * d_m_safe, -700.0, 0.0)
    exp_rev = np.clip(-2.0 * kappa2 * d_m_safe, -700.0, 0.0)

    j_fwd = j0_base * phi_eff1 * np.exp(exp_fwd)
    j_rev = j0_base * phi_eff2 * np.exp(exp_rev)
    j_sim_raw = j_fwd - j_rev

    # Low-Field Linear Approximation (V << Phi_b)
    exp_low = np.clip(-2.0 * kappa_0 * d_m_safe, -700.0, 0.0)
    j_low_field = (Q_ELEM * np.sqrt(2.0 * m_eff_kg * phi_b_J) / (d_m_safe * HBAR**2)) * \
                  (v_nz * Q_ELEM / (2.0 * np.pi**2)) * np.exp(exp_low)
    
    use_low_field = (v_nz < low_field_threshold_ratio * phi_b_eV) | (j_sim_raw <= 0)
    j_sim = np.where(use_low_field, j_low_field, j_sim_raw)

    # Fowler-Nordheim High-Field Emission Regime (V > Phi_b)
    pref_fn = (Q_ELEM**3 * (e_field**2)) / (8.0 * np.pi * H_PLANCK * phi_b_J)
    exp_fn = - (4.0 * np.sqrt(2.0 * m_eff_kg) * (phi_b_J**1.5)) / (3.0 * Q_ELEM * HBAR * e_field)
    j_fn = pref_fn * np.exp(np.clip(exp_fn, -700.0, 0.0))

    # Dynamic scaling of sigmoidal transition parameter delta_v
    delta_v_val = delta_v if delta_v is not None else max(0.02, 0.05 * phi_b_eV)
    w_fn = 1.0 / (1.0 + np.exp(-np.clip((v_nz - phi_b_eV) / delta_v_val, -50.0, 50.0)))
    j_sub = (1.0 - w_fn) * j_sim + w_fn * j_fn

    j_out[nz_mask] = v_sign[nz_mask] * np.maximum(j_sub, 0.0)
    return j_out


def compute_poole_frenkel_vectorized(
    v_vec: np.ndarray,
    d_m: float,
    phi_t_eV: float,
    eps_r: float,
    sigma_0: float,
    kBT_eV: float
) -> np.ndarray:
    """Computes field-assisted Poole-Frenkel trap emission current density (A/m^2)."""
    v_vec = np.atleast_1d(v_vec)
    v_abs = np.abs(v_vec)
    v_sign = np.sign(v_vec)
    d_m_safe = max(d_m, 1e-10)
    eps_r_safe = max(eps_r, 1.0)
    phi_t_safe = max(phi_t_eV, 0.01)
    
    e_field = v_abs / d_m_safe
    beta_pf = np.sqrt(Q_ELEM**3 / (np.pi * eps_r_safe * EPS_0))
    pf_lowering_eV = (beta_pf * np.sqrt(e_field)) / Q_ELEM
    eff_trap_eV = np.maximum(phi_t_safe - pf_lowering_eV, 0.0)

    # Thermal emission probability exponent strictly <= 0.0
    exp_factor = np.clip(-eff_trap_eV / max(kBT_eV, 1e-6), -700.0, 0.0)
    j_pf = sigma_0 * e_field * np.exp(exp_factor)
    return v_sign * j_pf


def compute_total_current_density(
    v_eff: np.ndarray,
    d_m: float,
    phi_b_eV: float,
    m_eff_kg: float,
    phi_t_eV: float,
    eps_r: float,
    sigma_0: float,
    kBT_eV: float,
    j_compliance_max: float,
    include_pf: bool = False
) -> np.ndarray:
    """Calculates composite current density (Simmons + optional Poole-Frenkel)."""
    j_sim = compute_simmons_vectorized(v_eff, d_m, phi_b_eV, m_eff_kg)
    if include_pf:
        j_pf = compute_poole_frenkel_vectorized(v_eff, d_m, phi_t_eV, eps_r, sigma_0, kBT_eV)
        j_tot = j_sim + j_pf
    else:
        j_tot = j_sim
    return np.clip(j_tot, -j_compliance_max, j_compliance_max)


# =================================================================================
# NEWTON-RAPHSON SOLVER FOR SERIES RESISTANCE (Rs)
# =================================================================================
def solve_effective_voltage_with_rs(
    v_applied: np.ndarray,
    d_m: float,
    phi_b_eV: float,
    m_eff_kg: float,
    phi_t_eV: float,
    eps_r: float,
    sigma_0: float,
    kBT_eV: float,
    r_series: float,
    cell_area_m2: float,
    j_compliance_max: float,
    include_pf: bool = False,
    max_iter: int = 50,
    tol: float = 1e-6,
    line_search_iters: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Solves for effective oxide voltage drops V_eff using adaptive Newton-Raphson iteration."""
    v_applied = np.atleast_1d(v_applied)
    if r_series <= 0.0:
        j_clipped = compute_total_current_density(
            v_applied, d_m, phi_b_eV, m_eff_kg, phi_t_eV, eps_r, sigma_0, kBT_eV, j_compliance_max, include_pf
        )
        return v_applied, j_clipped

    v_eff = np.copy(v_applied)

    for _ in range(max_iter):
        j_eff = compute_total_current_density(
            v_eff, d_m, phi_b_eV, m_eff_kg, phi_t_eV, eps_r, sigma_0, kBT_eV, j_compliance_max, include_pf
        )
        i_eff = j_eff * cell_area_m2
        f_val = v_eff + i_eff * r_series - v_applied
        err = np.abs(f_val)
        
        # Adaptive tolerance mask
        conv_mask = err < np.maximum(tol, 1e-5 * np.abs(v_applied))
        if np.all(conv_mask):
            break
            
        # Adaptive finite-difference voltage step
        dv_step = np.maximum(1e-5, 1e-4 * np.abs(v_eff))
        j_plus = compute_total_current_density(
            v_eff + dv_step, d_m, phi_b_eV, m_eff_kg, phi_t_eV, eps_r, sigma_0, kBT_eV, j_compliance_max, include_pf
        )
        j_minus = compute_total_current_density(
            v_eff - dv_step, d_m, phi_b_eV, m_eff_kg, phi_t_eV, eps_r, sigma_0, kBT_eV, j_compliance_max, include_pf
        )
        
        dj_dv = (j_plus - j_minus) / (2.0 * dv_step)
        df_dv = 1.0 + r_series * cell_area_m2 * np.maximum(dj_dv, 0.0)
        df_dv_safe = np.maximum(df_dv, 1.0)
        delta_v = f_val / df_dv_safe

        best_v = np.copy(v_eff)
        best_err = np.copy(err)
        alpha = np.ones_like(v_eff)

        for _ls in range(line_search_iters):
            v_cand = v_eff - alpha * delta_v
            j_cand = compute_total_current_density(
                v_cand, d_m, phi_b_eV, m_eff_kg, phi_t_eV, eps_r, sigma_0, kBT_eV, j_compliance_max, include_pf
            )
            f_cand = v_cand + (j_cand * cell_area_m2) * r_series - v_applied
            err_cand = np.abs(f_cand)
            
            improved_mask = err_cand < best_err
            best_v[improved_mask] = v_cand[improved_mask]
            best_err[improved_mask] = err_cand[improved_mask]
            
            if np.all(err_cand < tol):
                break
            alpha *= 0.5

        # Anti-stasis perturbation step if line search stalls
        stasis_mask = ~conv_mask & np.isclose(best_err, err)
        if np.any(stasis_mask):
            best_v[stasis_mask] -= 0.1 * delta_v[stasis_mask]

        v_eff = best_v

    j_final = compute_total_current_density(
        v_eff, d_m, phi_b_eV, m_eff_kg, phi_t_eV, eps_r, sigma_0, kBT_eV, j_compliance_max, include_pf
    )
    return v_eff, j_final


# =================================================================================
# MAIN SIMULATION ENGINE & DISK STREAMING
# =================================================================================
def run_phase9_engine(
    input_path: Optional[str] = None,
    output_dir: str = os.path.join("data", "final"),
    thicknesses_nm: Optional[List[float]] = None,
    trap_depths_eV: Optional[List[float]] = None,
    v_read: float = 1.0,
    voltage_range: float = 3.0,
    voltage_points: int = 121,
    cell_area_m2: float = 1e-14,
    r_series: float = 50.0,
    temperature_K: float = 300.0,
    default_m_eff_ratio: float = 0.3,
    sigma_0: float = 1e-3,
    j_compliance_max: float = 1e9,
    f_attempt_Hz: float = PHONON_ATTEMPT_FREQ_DEFAULT,
    tau_0: Optional[float] = None,
    min_on_off_ratio: float = 10.0,
    volatile_retention_max_days: float = 1.0,
    nonvolatile_retention_min_years: float = 10.0,
    force_grid_sweep: bool = False
) -> Tuple[str, str]:
    """Hardware-Level Physics Engine for Simmons & Poole-Frenkel Transport Modeling."""
    print("=================================================================")
    print("  PHASE 9: HARDWARE-LEVEL SIMMONS & POOLE-FRENKEL PHYSICS ENGINE ")
    print("=================================================================\n")

    kBT_eV = (K_BOLTZ * max(temperature_K, 1e-3)) / Q_ELEM
    tau_attempt = tau_0 if tau_0 is not None else (1.0 / max(f_attempt_Hz, 1e-18))

    df_candidates, resolved_source = load_candidate_dataset(input_path)

    col_mat, s_mat = resolve_column(df_candidates, ['Material_Pair', 'Material', 'Candidate', 'System', 'Name', 'Oxide', 'Formula', 'Composition'])
    col_eg, s_eg = resolve_column(df_candidates, ['Shell_Eg_bulk_eV', 'Eg_eV', 'Bandgap_eV', 'Eg', 'Bandgap', 'E_g'])
    col_conf, s_conf = resolve_column(df_candidates, ['Total_Confinement_eV', 'Confinement_eV', 'U_conf_eV', 'Confinement', 'U_conf'])
    col_phi_b, s_phi_b = resolve_column(df_candidates, ['Barrier_Height_eV', 'Phi_b_eV', 'Barrier_eV', 'Barrier', 'Phi_b'])
    col_meff, s_meff = resolve_column(df_candidates, ['Effective_Mass_Ratio', 'm_eff_ratio', 'm_eff', 'M_eff', 'Effective_Mass'])
    col_eps, s_eps = resolve_column(df_candidates, ['Eps_r', 'Dielectric_Constant', 'eps_r', 'k_relative', 'K', 'Relative_Permittivity'])
    
    col_thick, s_thick = resolve_column(df_candidates, ['Design_Thickness_nm', 'Thickness_nm', 'd_nm', 'Thickness', 'd'])
    col_trap, s_trap = resolve_column(df_candidates, ['Design_Trap_Depth_eV', 'Trap_Depth_eV', 'phi_t_eV', 'Trap_Depth', 'Phi_t'])

    voltages_iv = np.linspace(-voltage_range, voltage_range, voltage_points)

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "simulated_device_metrics.csv")
    iv_path = os.path.join(output_dir, "simulated_iv_curves.csv")

    metrics_records = []
    iv_records = []
    total_simulations = 0

    # Label-safe positional iteration
    for i, (_, row) in enumerate(df_candidates.iterrows()):
        pair_name = str(s_mat.iloc[i]) if s_mat is not None and pd.notnull(s_mat.iloc[i]) else f"Candidate_{i}"
        eg_shell = safe_float(s_eg.iloc[i] if s_eg is not None else None, 3.0)
        u_conf = safe_float(s_conf.iloc[i] if s_conf is not None else None, 1.2)

        if eg_shell <= 0:
            continue

        m_eff_ratio = max(safe_float(s_meff.iloc[i] if s_meff is not None else None, default_m_eff_ratio), 0.01)
        m_eff = m_eff_ratio * M_ELEM
        
        phi_b = safe_float(s_phi_b.iloc[i] if s_phi_b is not None else None, -1.0)
        if phi_b <= 0:
            phi_b = max(eg_shell / 2.0, 1.5)

        raw_eps = s_eps.iloc[i] if s_eps is not None else None
        if pd.notnull(raw_eps):
            eps_r = safe_float(raw_eps, estimate_eps_r(pair_name, eg_shell))
        else:
            eps_r = estimate_eps_r(pair_name, eg_shell)

        row_thick = safe_float(s_thick.iloc[i] if s_thick is not None else None, -1.0)
        row_trap = safe_float(s_trap.iloc[i] if s_trap is not None else None, -1.0)

        if not force_grid_sweep and row_thick > 0 and row_trap > 0 and row_trap < phi_b:
            local_thicks = [row_thick]
            local_traps = [row_trap]
        else:
            local_thicks = thicknesses_nm if thicknesses_nm else [4.0, 4.5, 5.0, 5.5, 6.0]
            fallback_traps = trap_depths_eV if trap_depths_eV else [0.8, 1.0, 1.2, 1.4, 1.6]
            local_traps = [safe_float(t, 1.0) for t in fallback_traps if safe_float(t, -1.0) > 0 and safe_float(t, -1.0) < phi_b]
            
            if not local_traps:
                local_traps = [round(phi_b * factor, 3) for factor in [0.3, 0.5, 0.7] if phi_b * factor > 0.05]
            if not local_traps:
                local_traps = [max(0.1, round(phi_b * 0.5, 3))]

        for d_nm, phi_t in itertools.product(local_thicks, local_traps):
            if pd.isna(phi_t) or phi_t <= 0 or phi_t >= phi_b:
                continue

            d_m = max(d_nm * 1e-9, 1e-10)

            # Thermal Retention Lifetime Calculation
            e_retention_act = min(phi_t, u_conf)
            exp_factor = np.clip(e_retention_act / kBT_eV, 0.0, 700.0)
            ret_sec = tau_attempt * np.exp(exp_factor)
            
            if np.isnan(ret_sec) or np.isinf(ret_sec):
                ret_years = 1e15
            else:
                ret_years = min(float(ret_sec / (365.25 * 86400.0)), 1e15)

            # Read Voltage Metrics
            v_read_arr = np.array([v_read], dtype=np.float64)
            _, j_off_read_vec = solve_effective_voltage_with_rs(
                v_read_arr, d_m, phi_b, m_eff, phi_t, eps_r, sigma_0, kBT_eV,
                r_series, cell_area_m2, j_compliance_max, include_pf=False
            )
            _, j_on_read_vec = solve_effective_voltage_with_rs(
                v_read_arr, d_m, phi_b, m_eff, phi_t, eps_r, sigma_0, kBT_eV,
                r_series, cell_area_m2, j_compliance_max, include_pf=True
            )

            j_off_read_clipped = float(j_off_read_vec[0])
            j_on_read_clipped = float(j_on_read_vec[0])

            i_off_read_A = j_off_read_clipped * cell_area_m2
            i_on_read_A = j_on_read_clipped * cell_area_m2

            # Magnitude-based ON/OFF ratio for positive and negative voltages
            abs_j_off = abs(j_off_read_clipped)
            abs_j_on = abs(j_on_read_clipped)
            on_off_ratio = (abs_j_on / abs_j_off) if abs_j_off > 1e-30 else 1.0

            # Dynamic classification criteria
            volatile_cutoff_years = volatile_retention_max_days / 365.25
            is_volatile_selector = bool(on_off_ratio >= min_on_off_ratio and ret_years < volatile_cutoff_years)
            is_nonvolatile_memory = bool(on_off_ratio >= min_on_off_ratio and ret_years >= nonvolatile_retention_min_years)
            is_quasi_nonvolatile = bool(on_off_ratio >= min_on_off_ratio and volatile_cutoff_years <= ret_years < nonvolatile_retention_min_years)

            rec_dict = {
                'Material_Pair': pair_name,
                'Shell_Eg_bulk_eV': eg_shell,
                'Total_Confinement_eV': u_conf,
                'Barrier_Height_eV': phi_b,
                'Effective_Mass_Ratio': m_eff_ratio,
                'Calculated_Eps_r': round(eps_r, 3),
                'Design_Thickness_nm': d_nm,
                'Design_Trap_Depth_eV': phi_t,
                'Retention_Time_Years': ret_years,
                'Volatile_Selector_Candidate': is_volatile_selector,
                'Quasi_NonVolatile_Candidate': is_quasi_nonvolatile,
                'NonVolatile_Memory_Candidate': is_nonvolatile_memory,
                'Read_Voltage_V': v_read,
                'J_OFF_Read_A_m2': j_off_read_clipped,
                'J_ON_Read_A_m2': j_on_read_clipped,
                'I_OFF_Read_A': i_off_read_A,
                'I_ON_Read_A': i_on_read_A,
                'Simulated_ON_OFF_Ratio': on_off_ratio
            }
            metrics_records.append(rec_dict)

            # Dynamic I-V Sweep Calculation
            _, j_off_vec_clipped = solve_effective_voltage_with_rs(
                voltages_iv, d_m, phi_b, m_eff, phi_t, eps_r, sigma_0, kBT_eV,
                r_series, cell_area_m2, j_compliance_max, include_pf=False
            )
            v_eff_vec, j_on_vec_clipped = solve_effective_voltage_with_rs(
                voltages_iv, d_m, phi_b, m_eff, phi_t, eps_r, sigma_0, kBT_eV,
                r_series, cell_area_m2, j_compliance_max, include_pf=True
            )

            df_iv_chunk = pd.DataFrame({
                'Material_Pair': pair_name,
                'Design_Thickness_nm': d_nm,
                'Design_Trap_Depth_eV': phi_t,
                'Voltage_Applied_V': np.round(voltages_iv, 4),
                'Voltage_Effective_V': np.round(v_eff_vec, 4),
                'J_OFF_A_m2': j_off_vec_clipped,
                'J_ON_A_m2': j_on_vec_clipped,
                'I_OFF_A': j_off_vec_clipped * cell_area_m2,
                'I_ON_A': j_on_vec_clipped * cell_area_m2
            })
            iv_records.append(df_iv_chunk)
            total_simulations += 1

    # Optimized batch write to disk
    if metrics_records:
        df_metrics_all = pd.DataFrame(metrics_records)
        df_metrics_all.to_csv(metrics_path, index=False)

    if iv_records:
        df_iv_all = pd.concat(iv_records, ignore_index=True)
        df_iv_all.to_csv(iv_path, index=False)

    print("-> SUCCESS: Simulation Complete!")
    print(f"   Total Device Configurations Calculated: {total_simulations}")
    print(f"   Candidate Metrics saved to: {metrics_path}")
    print(f"   Full I-V Curves saved to:   {iv_path}\n")

    return metrics_path, iv_path


# =================================================================================
# DYNAMIC PLOTTING UTILITY
# =================================================================================
def plot_single_device_iv(
    iv_csv_path: str,
    material_pair: Optional[str] = None,
    thickness_nm: Optional[float] = None,
    trap_depth_eV: Optional[float] = None,
    use_log_scale: bool = True,
    i_floor: float = 1e-18,
    save_path: Optional[str] = None
) -> None:
    """Plots the I-V curve from generated output CSV with log(0) safeguards."""
    if not os.path.exists(iv_csv_path):
        print(f"-> Warning: Cannot plot, CSV path '{iv_csv_path}' does not exist.")
        return

    try:
        df_iv = pd.read_csv(iv_csv_path)
    except Exception as err:
        print(f"-> Warning: Failed to read I-V CSV '{iv_csv_path}': {err}")
        return

    if df_iv.empty:
        print("-> Warning: I-V CSV is empty.")
        return

    if material_pair is None or material_pair not in df_iv['Material_Pair'].values:
        material_pair = str(df_iv['Material_Pair'].iloc[0])

    df_mat = df_iv[df_iv['Material_Pair'] == material_pair]
    if df_mat.empty:
        print(f"-> Warning: No records found for Material_Pair '{material_pair}'.")
        return
    
    if thickness_nm is None:
        thickness_nm = float(df_mat['Design_Thickness_nm'].iloc[0])
    if trap_depth_eV is None:
        trap_depth_eV = float(df_mat['Design_Trap_Depth_eV'].iloc[0])

    sub = df_mat[
        (np.isclose(df_mat['Design_Thickness_nm'], thickness_nm)) & 
        (np.isclose(df_mat['Design_Trap_Depth_eV'], trap_depth_eV))
    ]

    if sub.empty:
        print(f"-> Warning: No matching dynamic records for Material='{material_pair}', "
              f"d={thickness_nm}nm, trap={trap_depth_eV}eV.")
        return

    plt.figure(figsize=(8, 5))
    v_app = sub['Voltage_Applied_V'].values
    i_off = sub['I_OFF_A'].values
    i_on = sub['I_ON_A'].values

    if use_log_scale:
        plt.semilogy(v_app, np.maximum(np.abs(i_off), i_floor), label='OFF State (Simmons Quantum Tunneling)', color='blue', linewidth=2)
        plt.semilogy(v_app, np.maximum(np.abs(i_on), i_floor), label='ON State (Simmons + Poole-Frenkel)', color='red', linestyle='--', linewidth=2)
        plt.ylabel('|Current| (A) [Log Scale]')
    else:
        plt.plot(v_app, i_off, label='OFF State (Simmons Quantum Tunneling)', color='blue', linewidth=2)
        plt.plot(v_app, i_on, label='ON State (Simmons + Poole-Frenkel)', color='red', linestyle='--', linewidth=2)
        plt.ylabel('Current (A) [Linear Scale]')

    plt.xlabel('Applied Voltage (V)')
    plt.title(f'Dynamic I-V Characteristics: {material_pair} (d = {thickness_nm} nm, phi_t = {trap_depth_eV} eV)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"-> Plot saved to '{save_path}'")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 9 Hardware Transport Physics Engine")
    parser.add_argument("--input", type=str, default=None, help="Input candidate CSV file path")
    parser.add_argument("--output_dir", type=str, default=os.path.join("data", "final"), help="Output directory")
    parser.add_argument("--grid_sweep", action="store_true", help="Force sweep over thickness and trap depth grids")
    
    args = parser.parse_args()
    
    metrics_csv, iv_csv = run_phase9_engine(
        input_path=args.input,
        output_dir=args.output_dir,
        force_grid_sweep=args.grid_sweep
    )
    
    plot_single_device_iv(iv_csv, save_path=os.path.join(args.output_dir, "sample_iv_curve.png"))