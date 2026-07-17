# JWST_0120
# Fast bounded MAST query for MoM-z14 NIRSpec/PRISM spectrum with interactive line presets.

from pathlib import Path
from urllib.parse import quote
from datetime import datetime, timezone
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.mast import Observations
import ipywidgets as widgets
from IPython.display import display

try:
    from google.colab import output
    output.enable_custom_widget_manager()
except Exception:
    pass

VERSION = "JWST_0120"
TARGET_RA = 150.09354167
TARGET_DEC = 2.27325000
TARGET = SkyCoord(TARGET_RA*u.deg, TARGET_DEC*u.deg, frame="icrs")
PROGRAM = "5224"
Z = 14.44

ROOT = Path("/content/JWST_OUTPUT")
DATA_DIR = ROOT / "DATA" / VERSION / "FITS"
CSV_DIR = ROOT / "CSV"
PNG_DIR = ROOT / "PNG"
for p in (DATA_DIR, CSV_DIR, PNG_DIR):
    p.mkdir(parents=True, exist_ok=True)

print(f"CODE OUTPUT: {VERSION}", flush=True)
print(f"TARGET RA/DEC: {TARGET_RA:.8f}  {TARGET_DEC:.8f}", flush=True)
print(f"PROGRAM: {PROGRAM}   MODE: NIRSpec PRISM/CLEAR", flush=True)

LINES = {
    "Lyman-alpha break": [("Lyα 1215.67", 1215.67)],
    "Nitrogen — N V": [("N V 1238.82", 1238.82), ("N V 1242.80", 1242.80)],
    "Carbon — C II": [("C II 1334.53", 1334.53)],
    "Silicon/Oxygen — Si IV + O IV]": [("Si IV 1393.76", 1393.76), ("O IV] 1401.16", 1401.16), ("Si IV 1402.77", 1402.77)],
    "Nitrogen — N IV]": [("N IV] 1483.32", 1483.32), ("N IV] 1486.50", 1486.50)],
    "Carbon — C IV": [("C IV 1548.20", 1548.20), ("C IV 1550.77", 1550.77)],
    "Helium — He II": [("He II 1640.42", 1640.42)],
    "Oxygen — O III]": [("O III] 1660.81", 1660.81), ("O III] 1666.15", 1666.15)],
    "Nitrogen — N III]": [("N III] 1746.82", 1746.82), ("N III] 1748.65", 1748.65), ("N III] 1749.67", 1749.67), ("N III] 1752.16", 1752.16), ("N III] 1753.99", 1753.99)],
    "Silicon — Si III]": [("Si III] 1882.71", 1882.71), ("Si III] 1892.03", 1892.03)],
    "Carbon — C III]": [("C III] 1906.68", 1906.68), ("C III] 1908.73", 1908.73)],
    "Oxygen — [O II]": [("[O II] 3726.03", 3726.03), ("[O II] 3728.82", 3728.82)],
    "Neon — [Ne III]": [("[Ne III] 3868.76", 3868.76), ("[Ne III] 3967.47", 3967.47)],
}

print("QUERYING MAST — PROGRAM-ONLY FAST QUERY", flush=True)
try:
    Observations.TIMEOUT = 60
except Exception:
    pass

obs = Observations.query_criteria(obs_collection="JWST", proposal_id=PROGRAM)
print(f"PROGRAM OBSERVATIONS RETURNED: {len(obs)}", flush=True)
if len(obs) == 0:
    raise RuntimeError("No JWST observations returned for program 5224.")

# Keep only observations near the requested target and NIRSpec products.
ra_col = "s_ra" if "s_ra" in obs.colnames else None
dec_col = "s_dec" if "s_dec" in obs.colnames else None
near = np.ones(len(obs), dtype=bool)
if ra_col and dec_col:
    coords = SkyCoord(np.asarray(obs[ra_col], float)*u.deg, np.asarray(obs[dec_col], float)*u.deg)
    sep = coords.separation(TARGET).arcsec
    near &= np.isfinite(sep) & (sep <= 30.0)
if "instrument_name" in obs.colnames:
    near &= np.array(["NIRSPEC" in str(v).upper() for v in obs["instrument_name"]])
obs = obs[near]
print(f"NEAR-TARGET NIRSPEC OBSERVATIONS: {len(obs)}", flush=True)
if len(obs) == 0:
    raise RuntimeError("Program 5224 returned no NIRSpec observations within 30 arcsec of the target coordinates.")

print("FETCHING PRODUCT TABLE", flush=True)
products = Observations.get_product_list(obs)
pdf = products.to_pandas()
for col in pdf.columns:
    if pdf[col].dtype == object:
        pdf[col] = pdf[col].fillna("").astype(str)
print(f"PRODUCT ROWS: {len(pdf)}", flush=True)

name_col = "productFilename"
uri_col = "dataURI"
if name_col not in pdf or uri_col not in pdf:
    raise RuntimeError("MAST product table lacks productFilename/dataURI.")

names = pdf[name_col].str.lower()
desc = pdf.get("description", pd.Series("", index=pdf.index)).astype(str).str.lower()
x1d = names.str.endswith("_x1d.fits")
prism = names.str.contains("prism") | desc.str.contains("prism")
candidates = pdf[x1d & prism].copy()
if len(candidates) == 0:
    candidates = pdf[x1d].copy()
print(f"X1D CANDIDATES: {len(candidates)}", flush=True)
if len(candidates) == 0:
    raise RuntimeError("No x1d FITS products found.")


def download(uri, filename):
    path = DATA_DIR / filename
    if path.exists() and path.stat().st_size > 0:
        return path
    url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=" + quote(str(uri), safe="")
    print(f"DOWNLOADING: {filename}", flush=True)
    with requests.get(url, stream=True, timeout=(20, 120)) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1024*1024):
                if chunk:
                    f.write(chunk)
    return path


def coord_from_hdul(hdul):
    pairs = [("SRCRA", "SRCDEC"), ("SOURCE_RA", "SOURCE_DEC"), ("RA_TARG", "DEC_TARG"), ("TARG_RA", "TARG_DEC"), ("RA", "DEC")]
    for hdu in hdul:
        for rk, dk in pairs:
            if rk in hdu.header and dk in hdu.header:
                try:
                    return float(hdu.header[rk]), float(hdu.header[dk]), f"{hdu.name}:{rk}/{dk}"
                except Exception:
                    pass
    return None, None, "NONE"


def read_x1d(path):
    with fits.open(path, memmap=False) as hdul:
        ra, dec, csrc = coord_from_hdul(hdul)
        for hdu in hdul:
            data = hdu.data
            cols = list(getattr(data, "names", []) or [])
            cmap = {c.upper(): c for c in cols}
            if "WAVELENGTH" in cmap and "FLUX" in cmap:
                wave = np.asarray(data[cmap["WAVELENGTH"]], float).ravel()
                flux = np.asarray(data[cmap["FLUX"]], float).ravel()
                ekey = cmap.get("FLUX_ERROR") or cmap.get("ERROR") or cmap.get("ERR")
                err = np.asarray(data[ekey], float).ravel() if ekey else np.full_like(flux, np.nan)
                return ra, dec, csrc, hdu.name, wave, flux, err
    return None

ranked = []
for _, row in candidates.iterrows():
    filename = row[name_col]
    try:
        path = download(row[uri_col], filename)
        spec = read_x1d(path)
        if spec is None:
            print(f"SKIP NO SPECTRUM: {filename}", flush=True)
            continue
        ra, dec, csrc, hdu_name, wave, flux, err = spec
        offset = np.inf if ra is None or dec is None else SkyCoord(ra*u.deg, dec*u.deg).separation(TARGET).arcsec
        ranked.append((offset, path, ra, dec, csrc, hdu_name, wave, flux, err))
        print(f"CANDIDATE: {filename}   OFFSET {offset:.4f} arcsec   N={len(wave)}", flush=True)
    except Exception as exc:
        print(f"SKIP {filename}: {exc}", flush=True)

if not ranked:
    raise RuntimeError("No readable x1d spectrum found.")
ranked.sort(key=lambda x: x[0])
OFFSET, X1D_PATH, RA, DEC, COORD_SOURCE, SPEC_HDU, wave, flux, err = ranked[0]
if not np.isfinite(OFFSET) or OFFSET > 1.0:
    raise RuntimeError(f"Closest readable x1d is {OFFSET:.3f} arcsec from target; refusing to identify it as MoM-z14.")

finite = np.isfinite(wave) & np.isfinite(flux)
wave, flux, err = wave[finite], flux[finite], err[finite]
order = np.argsort(wave)
wave, flux, err = wave[order], flux[order], err[order]

print("VERIFIED MOM-Z14 PRODUCT", flush=True)
print(f"FILE: {X1D_PATH}", flush=True)
print(f"OFFSET: {OFFSET:.6f} arcsec", flush=True)
print(f"COORD SOURCE: {COORD_SOURCE}", flush=True)
print(f"SPECTRUM HDU: {SPEC_HDU}", flush=True)
print(f"NATIVE SAMPLES: {len(wave)}", flush=True)
print(f"WAVELENGTH RANGE: {wave.min():.6f} to {wave.max():.6f} um", flush=True)

native_csv = CSV_DIR / f"{VERSION}_MOM_Z14_PRISM_NATIVE_1D.csv"
pd.DataFrame({"wavelength_um": wave, "flux": flux, "flux_error": err}).to_csv(native_csv, index=False)

PRESETS = {"Full prism spectrum": (float(wave.min()), float(wave.max())), "All UV diagnostics": (max(float(wave.min()), 1.75), min(float(wave.max()), 3.15))}
for group, glines in LINES.items():
    obs_um = np.array([rest*(1+Z)/10000 for _, rest in glines])
    pad = 0.055 if len(glines) > 1 else 0.045
    PRESETS[group] = (max(float(wave.min()), float(obs_um.min()-pad)), min(float(wave.max()), float(obs_um.max()+pad)))
PRESETS = {k:v for k,v in PRESETS.items() if v[1] > v[0]}

dropdown = widgets.Dropdown(options=list(PRESETS), value="Nitrogen — N IV]" if "Nitrogen — N IV]" in PRESETS else list(PRESETS)[0], description="View")
zoom = widgets.FloatRangeSlider(value=PRESETS[dropdown.value], min=float(wave.min()), max=float(wave.max()), step=max((wave.max()-wave.min())/4000, 1e-5), description="Zoom μm", readout_format=".5f", continuous_update=False, layout=widgets.Layout(width="920px"))
cursor = widgets.FloatSlider(value=float(np.mean(PRESETS[dropdown.value])), min=float(wave.min()), max=float(wave.max()), step=max((wave.max()-wave.min())/8000, 1e-5), description="Cursor μm", readout_format=".6f", continuous_update=False, layout=widgets.Layout(width="920px"))
show_error = widgets.Checkbox(value=True, description="Show ±1σ")
show_all = widgets.Checkbox(value=False, description="Show all reference groups")
out = widgets.Output()


def preset_changed(change):
    if change.get("name") == "value":
        zoom.value = PRESETS[change["new"]]
        cursor.value = float(np.mean(PRESETS[change["new"]]))

dropdown.observe(preset_changed, names="value")


def draw(*_):
    out.clear_output(wait=True)
    x0, x1 = zoom.value
    m = (wave >= x0) & (wave <= x1)
    if not np.any(m):
        return
    nearest = int(np.argmin(np.abs(wave-cursor.value)))
    with out:
        fig, ax = plt.subplots(figsize=(14, 6.5))
        fig.patch.set_facecolor("#07111f")
        ax.set_facecolor("#07111f")
        ax.plot(wave[m], flux[m], lw=0.8)
        ax.scatter(wave[m], flux[m], s=7)
        em = m & np.isfinite(err)
        if show_error.value and np.any(em):
            ax.fill_between(wave[em], flux[em]-err[em], flux[em]+err[em], alpha=0.12)
        groups = LINES.items() if show_all.value else [(dropdown.value, LINES.get(dropdown.value, []))]
        ymin, ymax = np.nanmin(flux[m]), np.nanmax(flux[m])
        for _, glines in groups:
            for label, rest in glines:
                obs_um = rest*(1+Z)/10000
                if x0 <= obs_um <= x1:
                    ax.axvline(obs_um, lw=0.18, alpha=0.28)
                    ax.text(obs_um, ymax, label, rotation=90, fontsize=6.5, alpha=0.55, va="top", ha="right")
        ax.axvline(cursor.value, lw=0.35, ls=":", alpha=0.55)
        ax.scatter([wave[nearest]], [flux[nearest]], marker="x", s=45)
        ax.set_xlim(x0, x1)
        ax.set_xlabel("Observed wavelength [μm]")
        ax.set_ylabel("Flux [native pipeline units]")
        ax.set_title(f"MoM-z14 — JWST/NIRSpec PRISM — {len(wave)} native samples")
        ax.grid(alpha=0.12)
        plt.tight_layout()
        png = PNG_DIR / f"{VERSION}_CURRENT_VIEW.png"
        fig.savefig(png, dpi=180, bbox_inches="tight")
        plt.show()
        print(f"VIEW                 {dropdown.value}")
        print(f"CURSOR OBSERVED      {wave[nearest]:.6f} μm")
        print(f"CURSOR REST          {wave[nearest]*10000/(1+Z):.3f} Å")
        print(f"CURSOR FLUX          {flux[nearest]:.6e}")
        print(f"TARGET OFFSET        {OFFSET:.6f} arcsec")

for control in (zoom, cursor, show_error, show_all):
    control.observe(draw, names="value")
draw()
display(widgets.VBox([dropdown, zoom, cursor, widgets.HBox([show_error, show_all]), out]))

manifest = {
    "version": VERSION,
    "target_ra_deg": TARGET_RA,
    "target_dec_deg": TARGET_DEC,
    "redshift": Z,
    "program": PROGRAM,
    "selected_file": str(X1D_PATH),
    "offset_arcsec": float(OFFSET),
    "native_samples": int(len(wave)),
    "wavelength_um": [float(wave.min()), float(wave.max())],
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}
manifest_path = CSV_DIR / f"{VERSION}_MANIFEST.json"
manifest_path.write_text(json.dumps(manifest, indent=2))
print("OUTPUT SUMMARY")
print(f"Native CSV: {native_csv}")
print(f"Current plot: {PNG_DIR / f'{VERSION}_CURRENT_VIEW.png'}")
print(f"Manifest: {manifest_path}")
print(f"Timestamp UTC: {manifest['timestamp_utc']}")
print(f"# {VERSION}")
