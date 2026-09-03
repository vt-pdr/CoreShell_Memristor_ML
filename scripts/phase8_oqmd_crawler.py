import os
import re
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import requests

# =====================================================================
# 1. CONFIGURATION & ENVIRONMENT OVERRIDES (ZERO HARDCODING)
# =====================================================================
# API Network & Screening Hyperparameters
TOP_N: Optional[int] = int(os.environ.get("OQMD_TOP_N", "10")) if os.environ.get("OQMD_TOP_N") else 10
STABILITY_THRESHOLD_EV: float = float(os.environ.get("OQMD_STABILITY_THRESHOLD", "0.05"))  # eV/atom
RATE_LIMIT_SEC: float = float(os.environ.get("OQMD_RATE_LIMIT", "1.0"))                  # Network throttle delay
MAX_RETRIES: int = int(os.environ.get("OQMD_MAX_RETRIES", "3"))                            # HTTP retry limit
API_TIMEOUT: float = float(os.environ.get("OQMD_API_TIMEOUT", "45.0"))                    # Increased to 45s default for rare-earths
CONNECT_TIMEOUT: float = float(os.environ.get("OQMD_CONNECT_TIMEOUT", "5.0"))              # Connect timeout in seconds
USE_CACHE: bool = os.environ.get("OQMD_USE_CACHE", "true").lower() in ("true", "1", "yes")

# API Endpoints (Primary with fallback protocols)
PRIMARY_API_URL: str = os.environ.get("OQMD_API_URL", "http://oqmd.org/oqmdapi/formationenergy")
USER_AGENT: str = os.environ.get("OQMD_USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CoreShell/1.0")

# Column Matching Candidates (Configurable via ENV comma-separated strings)
CORE_CANDIDATES: List[str] = [c.strip().lower() for c in os.environ.get(
    "OQMD_CORE_COLS", "core_formula,core,core_composition,core_material,core_phase"
).split(",")]

SHELL_CANDIDATES: List[str] = [c.strip().lower() for c in os.environ.get(
    "OQMD_SHELL_COLS", "shell_formula,shell,shell_composition,shell_material,shell_phase"
).split(",")]

# =====================================================================
# 2. DYNAMIC PATH RESOLUTION
# =====================================================================
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = SCRIPT_DIR
while PROJECT_ROOT.name in ["scripts", "src", "notebooks"] and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

# File Path Overrides via Environment Variables
CACHE_PATH_ENV = os.environ.get("OQMD_CACHE_FILE")
OUTPUT_PATH_ENV = os.environ.get("OQMD_OUTPUT_FILE")
PROCESSED_PATH_ENV = os.environ.get("OQMD_PROCESSED_FILE")
INPUT_PATH_ENV = os.environ.get("OQMD_INPUT_FILE")

CACHE_FILE = Path(CACHE_PATH_ENV) if CACHE_PATH_ENV else (PROJECT_ROOT / "data" / "processed" / ".oqmd_cache.json")
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = Path(OUTPUT_PATH_ENV) if OUTPUT_PATH_ENV else (PROJECT_ROOT / "reports" / "publication_top_candidates_real.csv")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

PROCESSED_OUTPUT = Path(PROCESSED_PATH_ENV) if PROCESSED_PATH_ENV else (PROJECT_ROOT / "data" / "processed" / "oqmd_verified_memristors.csv")
PROCESSED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

def safe_rel_path(path: Path, root: Path) -> str:
    """Returns relative path string safely without raising ValueError if outside root."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)

# Resolve candidate pool input CSV
if INPUT_PATH_ENV and Path(INPUT_PATH_ENV).exists():
    input_file = Path(INPUT_PATH_ENV)
else:
    possible_inputs = [
        PROJECT_ROOT / "data" / "processed" / "synthesizable_memristors.csv",
        PROJECT_ROOT / "data" / "processed" / "final_master_verified_memristors.csv",
        PROJECT_ROOT / "data" / "processed" / "benchmarked_memristors.csv",
        PROJECT_ROOT / "synthesizable_memristors.csv",
    ]
    input_file = next((path for path in possible_inputs if path.exists()), None)

if not input_file:
    print("[-] Error: Could not locate candidate pool CSV. Set OQMD_INPUT_FILE or run prior screening phases.")
    exit(1)

print(f"[✓] Loaded candidate pool from '{safe_rel_path(input_file, PROJECT_ROOT)}'.")
df = pd.read_csv(input_file)
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

core_col = next((c for c in df.columns if c.lower() in CORE_CANDIDATES), None)
shell_col = next((c for c in df.columns if c.lower() in SHELL_CANDIDATES), None)

if not core_col or not shell_col:
    print(f"[-] Error: Could not identify Core/Shell formula columns. Available columns: {list(df.columns)}")
    exit(1)

top_candidates = df.head(TOP_N).copy() if (TOP_N is not None and len(df) > TOP_N) else df.copy()

# =====================================================================
# 3. LOCAL CACHING & FORMULA SANITIZER
# =====================================================================
def load_cache() -> Dict[str, Any]:
    if USE_CACHE and CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache: Dict[str, Any]) -> None:
    if not USE_CACHE:
        return
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"  [Cache Warning] Could not persist API cache: {e}")

def extract_ehull_from_cache(cached_entry: Any) -> Optional[float]:
    if cached_entry is None:
        return None
    if isinstance(cached_entry, (int, float)):
        return float(cached_entry)
    if isinstance(cached_entry, dict):
        for key in ['raw_ehull', 'raw', 'effective_ehull', 'stability']:
            if key in cached_entry and cached_entry[key] is not None:
                return float(cached_entry[key])
    return None

def clean_formula(formula: str) -> str:
    cleaned = str(formula).strip()
    cleaned = re.sub(r'\\text\{([^}]+)\}', r'\1', cleaned)
    cleaned = re.sub(r'[\{\}\_\^\s]', '', cleaned)
    
    full_phase_pattern = r'^(?:alpha|beta|gamma|delta|epsilon|theta|cubic|tetragonal|monoclinic|hexagonal|orthorhombic|rhombohedral|amorphous|anatase|rutile|brookite|wurtzite|rocksalt|perovskite|spinel)-(?=[A-Z][a-z]?\d*)'
    cleaned = re.sub(full_phase_pattern, '', cleaned, flags=re.IGNORECASE)
    
    single_letter_pattern = r'^[ctmhroabg]-(?=[A-Z][a-z]?\d*)'
    cleaned = re.sub(single_letter_pattern, '', cleaned)
    
    return cleaned

# =====================================================================
# 4. ROBUST OQMD LIVE DATA EXTRACTOR
# =====================================================================
api_cache = load_cache()

def fetch_oqmd_ehull(formula: str, cache: Dict[str, Any]) -> Optional[float]:
    raw_formula = str(formula).strip()
    sanitized = clean_formula(raw_formula)
    
    # ONLY hit cache if a valid numerical stability float exists (prevents cached-failure trap)
    if USE_CACHE and sanitized in cache:
        cached_val = extract_ehull_from_cache(cache[sanitized])
        if cached_val is not None:
            return cached_val
    
    params = {
        "composition": sanitized,
        "fields": "name,stability",
        "format": "json"
    }
    headers = {'User-Agent': USER_AGENT}
    
    time.sleep(RATE_LIMIT_SEC)
    
    # Dual-protocol retry sequence (http -> https fallback)
    urls_to_try = [
        PRIMARY_API_URL,
        PRIMARY_API_URL.replace("http://", "https://") if PRIMARY_API_URL.startswith("http://") else PRIMARY_API_URL.replace("https://", "http://")
    ]
    
    for attempt in range(MAX_RETRIES + 1):
        target_url = urls_to_try[attempt % len(urls_to_try)]
        try:
            response = requests.get(
                target_url,
                params=params,
                headers=headers,
                timeout=(CONNECT_TIMEOUT, API_TIMEOUT),
                allow_redirects=True
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('data', [])
                
                stabilities = [
                    float(item['stability']) 
                    for item in results 
                    if isinstance(item, dict) and item.get('stability') is not None
                ]
                
                if stabilities:
                    raw_ehull = min(stabilities)
                    cache[sanitized] = {
                        'raw_ehull': raw_ehull,
                        'effective_ehull': max(0.0, raw_ehull)
                    }
                    save_cache(cache)
                    return raw_ehull
                else:
                    # Composition genuinely not present in OQMD database
                    return None
                    
            elif response.status_code in [429, 502, 503, 504]:
                wait_time = (2 ** attempt) * RATE_LIMIT_SEC
                print(f"  [API Retry] HTTP {response.status_code} on {target_url} for '{sanitized}'. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"  [API Warning] HTTP {response.status_code} for composition: '{sanitized}'")
                break
                
        except requests.exceptions.Timeout:
            print(f"  [API Timeout] '{sanitized}' timed out after {API_TIMEOUT}s (Attempt {attempt+1}/{MAX_RETRIES+1})")
            if attempt < MAX_RETRIES:
                time.sleep((2 ** attempt) * RATE_LIMIT_SEC)
                continue
        except (requests.exceptions.RequestException, ValueError) as e:
            if attempt < MAX_RETRIES:
                time.sleep((2 ** attempt) * RATE_LIMIT_SEC)
                continue
            print(f"  [API Error] Could not query '{sanitized}': {e}")
            break
            
    # Do NOT persist None to cache on timeouts/network failures so future runs can retry live
    return None

# =====================================================================
# 5. EXECUTE DFT CRAWLER & SCREENING
# =====================================================================
print(f"\nCrawling live OQMD DFT stability data for Top {len(top_candidates)} core-shell pairs...\n")

core_ehulls, shell_ehulls = [], []

for idx, (_, row) in enumerate(top_candidates.iterrows()):
    core = str(row[core_col]).strip()
    shell = str(row[shell_col]).strip()
    
    print(f"[{idx+1}/{len(top_candidates)}] Querying Core: {core:<12} | Shell: {shell:<12}")
    
    c_val = fetch_oqmd_ehull(core, api_cache)
    s_val = fetch_oqmd_ehull(shell, api_cache)
    
    core_ehulls.append(c_val)
    shell_ehulls.append(s_val)

# Convert outputs safely to float numeric series (coercing None/API timeouts to NaN)
top_candidates['Core_Ehull_OQMD_eV'] = pd.to_numeric(pd.Series(core_ehulls, index=top_candidates.index), errors='coerce')
top_candidates['Shell_Ehull_OQMD_eV'] = pd.to_numeric(pd.Series(shell_ehulls, index=top_candidates.index), errors='coerce')

# Calculate maximum pair hull distance safely without pandas type crashes
top_candidates['Max_Pair_Ehull_eV'] = top_candidates[['Core_Ehull_OQMD_eV', 'Shell_Ehull_OQMD_eV']].max(axis=1, skipna=False)

# Synthesizability screening
top_candidates['Thermodynamically_Stable'] = (
    (top_candidates['Max_Pair_Ehull_eV'] <= STABILITY_THRESHOLD_EV) & 
    top_candidates['Max_Pair_Ehull_eV'].notna()
)

def assign_verification_status(r):
    c_na = pd.isna(r['Core_Ehull_OQMD_eV'])
    s_na = pd.isna(r['Shell_Ehull_OQMD_eV'])
    
    if c_na and s_na:
        return 'BOTH_UNVERIFIED_API_FAIL'
    elif c_na:
        return 'CORE_UNVERIFIED_API_FAIL'
    elif s_na:
        return 'SHELL_UNVERIFIED_API_FAIL'
    
    return 'VERIFIED_STABLE' if r['Thermodynamically_Stable'] else 'UNSTABLE'

top_candidates['Verification_Status'] = top_candidates.apply(assign_verification_status, axis=1)

# =====================================================================
# 6. ARTIFACT EXPORT
# =====================================================================
top_candidates.to_csv(OUTPUT_FILE, index=False)
top_candidates.to_csv(PROCESSED_OUTPUT, index=False)

print("\n==================================================")
print("SUCCESS! Real-time OQMD verification complete.")
print("Verified candidate shortlists exported to:")
print(f" -> '{safe_rel_path(OUTPUT_FILE, PROJECT_ROOT)}'")
print(f" -> '{safe_rel_path(PROCESSED_OUTPUT, PROJECT_ROOT)}'")
print("==================================================")