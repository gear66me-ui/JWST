# JWST_0159
import re, requests

SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0158_JADES_GS_Z11_0_FULL_SPECTRAL_REDSHIFT_CHAIN_WIDGET.py"
source = requests.get(SOURCE_URL, timeout=120).text
source = source.replace('VERSION = "JWST_0158"', 'VERSION = "JWST_0159"')

replacement = r'''
def download_exact_jades_spectrum():
    cached = DATA / "JADES_GS_Z11_0_EXACT_POSITION_PRISM_SPEC1D.fits"
    if cached.exists() and cached.stat().st_size > 50000:
        return cached

    ra, dec = 53.1647632, -27.7746223
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        obs = Observations.query_region(f"{ra} {dec}", radius="0.8 arcsec")
    if len(obs) == 0:
        raise RuntimeError("No JWST observations found at the published JADES-GS-z11-0 coordinates")

    keep = np.ones(len(obs), dtype=bool)
    if "obs_collection" in obs.colnames:
        keep &= np.array([str(x).upper() == "JWST" for x in obs["obs_collection"]])
    if "proposal_id" in obs.colnames:
        keep &= np.array([str(x).strip() in ("1210", "3215") for x in obs["proposal_id"]])
    selected_obs = obs[keep]
    if len(selected_obs) == 0:
        selected_obs = obs

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        products = Observations.get_product_list(selected_obs)

    names = np.array([str(x).lower() for x in products["productFilename"]])
    masks = [
        np.array([n.endswith(".fits") and ("spec1d" in n or "x1d" in n) and ("prism" in n or "clear" in n) for n in names]),
        np.array([n.endswith(".fits") and ("spec1d" in n or "x1d" in n) for n in names]),
    ]
    candidates = None
    for mask in masks:
        if np.any(mask):
            candidates = products[mask]
            break
    if candidates is None or len(candidates) == 0:
        raise RuntimeError("No public 1D NIRSpec spectrum found at the exact JADES-GS-z11-0 coordinates")

    score = []
    for row in candidates:
        n = str(row["productFilename"]).lower()
        s = 0
        s += 100 if "spec1d" in n else 0
        s += 80 if "x1d" in n else 0
        s += 60 if "prism" in n else 0
        s += 40 if "clear" in n else 0
        s += 20 if "3215" in n else 0
        s += 10 if "1210" in n else 0
        score.append(s)
    chosen = candidates[int(np.argmax(score))]

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        manifest = Observations.download_products(Table(rows=[chosen]), download_dir=str(DATA), cache=True)
    src = Path(str(manifest["Local Path"][0]))
    if not src.exists():
        raise RuntimeError("MAST did not return the selected JADES-GS-z11-0 spectrum")
    cached.write_bytes(src.read_bytes())
    return cached


def read_spectrum'''

source = re.sub(r'def download_exact_jades_spectrum\(\):.*?\n\ndef read_spectrum', replacement, source, flags=re.S)
source = source.replace('z_grid = np.linspace(10.95, 11.75, 801)', 'z_grid = np.linspace(10.95, 11.75, 401)')
source = source.replace('"maxiter": 350', '"maxiter": 120')
source = source.replace('JWST_0158_', 'JWST_0159_')
exec(compile(source, "JWST_0159_runtime.py", "exec"), globals())
