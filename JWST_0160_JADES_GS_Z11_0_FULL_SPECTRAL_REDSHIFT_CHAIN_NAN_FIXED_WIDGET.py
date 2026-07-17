# JWST_0160
import re, requests

SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0159_JADES_GS_Z11_0_FULL_SPECTRAL_REDSHIFT_CHAIN_FIXED_WIDGET.py"
source = requests.get(SOURCE_URL, timeout=120).text
source = source.replace('# JWST_0159', '# JWST_0160', 1)
source = source.replace('VERSION = "JWST_0159"', 'VERSION = "JWST_0160"')
source = source.replace('JWST_0159_', 'JWST_0160_')

safe_model = r'''
def empirical_sed(lam_um, z, beta, width, amplitude):
    lam_um = np.asarray(lam_um, dtype=float)
    z = float(z); beta = float(beta); width = float(width); amplitude = float(amplitude)
    if not np.isfinite(z): z = Z_PAPER
    if not np.isfinite(beta): beta = -1.8
    if not np.isfinite(width) or width <= 0: width = 0.018
    if not np.isfinite(amplitude): amplitude = 1.0
    edge = LYA_REST * (1.0 + z)
    safe_lam = np.where(np.isfinite(lam_um), lam_um, edge)
    continuum = amplitude * (np.maximum(safe_lam, edge) / 1.7) ** beta
    transmission = 0.5 * (1.0 + np.tanh((safe_lam - edge) / max(width, 1e-4)))
    model = continuum * transmission
    diffs = np.diff(safe_lam)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return np.nan_to_num(model, nan=0.0, posinf=0.0, neginf=0.0)
    dl = float(np.nanmedian(diffs))
    medlam = float(np.nanmedian(safe_lam[np.isfinite(safe_lam)]))
    if not np.isfinite(dl) or dl <= 0 or not np.isfinite(medlam):
        return np.nan_to_num(model, nan=0.0, posinf=0.0, neginf=0.0)
    sigma_lam = max(medlam / (100.0 * 2.354820045), dl * 0.5)
    sigma_pix = sigma_lam / dl
    if not np.isfinite(sigma_pix) or sigma_pix <= 0:
        return np.nan_to_num(model, nan=0.0, posinf=0.0, neginf=0.0)
    return gaussian_filter1d(np.nan_to_num(model, nan=0.0, posinf=0.0, neginf=0.0), sigma_pix, mode="nearest")


def objective'''
source = re.sub(r'def empirical_sed\(lam_um, z, beta, width, amplitude\):.*?\n\ndef objective', safe_model, source, flags=re.S)

source = source.replace(
    'fitmask = (wave >= 1.05) & (wave <= 5.20)',
    'fitmask = (wave >= 1.05) & (wave <= 5.20) & np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err) & (err > 0)'
)
source = source.replace(
    'lam, y, s = wave[fitmask], flux[fitmask], err[fitmask]',
    'lam, y, s = wave[fitmask], flux[fitmask], err[fitmask]\n_u = np.r_[True, np.diff(lam) > 0]\nlam, y, s = lam[_u], y[_u], s[_u]'
)
source = source.replace(
    'z, beta, width = theta',
    'z, beta, width = np.asarray(theta, dtype=float)\n    if not np.all(np.isfinite([z, beta, width])): return 1e99',
    1
)
source = source.replace('z_grid = np.linspace(10.95, 11.75, 401)', 'z_grid = np.linspace(10.95, 11.75, 321)')
source = source.replace('"maxiter": 120', '"maxiter": 90')

exec(compile(source, "JWST_0160_runtime.py", "exec"), globals())
