# JWST_0119
# Fetch the real MoM-z14 NIRSpec/PRISM spectrum from MAST and inspect published UV-line regions.

from pathlib import Path
from urllib.parse import quote
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

VERSION = "JWST_0119"
print(f"CODE OUTPUT: {VERSION}")

# NASA/STScI coordinates for MoM-z14 in COSMOS.
TARGET = SkyCoord("10h00m22.45s", "+02d16m23.70s", frame="icrs")
TARGET_RA = TARGET.ra.deg
TARGET_DEC = TARGET.dec.deg
Z = 14.44
PROGRAM = "5224"

ROOT = Path("/content/JWST_OUTPUT")
DATA_DIR = ROOT / "DATA" / VERSION / "FITS"
CSV_DIR = ROOT / "CSV"
PNG_DIR = ROOT / "PNG"
for p in (DATA_DIR, CSV_DIR, PNG_DIR):
    p.mkdir(parents=True, exist_ok=True)

print(f"TARGET RA/DEC: {TARGET_RA:.8f}  {TARGET_DEC:.8f}")
print(f"PROGRAM: {PROGRAM}   MODE: NIRSpec PRISM/CLEAR")

# Rest-frame UV lines relevant to the published MoM-z14 prism spectrum and nearby diagnostics.
LINES = {
    "Lyman-alpha break": [("Lyα", 1215.67)],
    "Nitrogen — N IV]": [("N IV] 1483.32", 1483.32), ("N IV] 1486.50", 1486.50)],
    "Carbon — C IV": [("C IV 1548.20", 1548.20), ("C IV 1550.77", 1550.77)],
    "Helium — He II": [("He II 1640.42", 1640.42)],
    "Oxygen — O III]": [("O III] 1660.81", 1660.81), ("O III] 1666.15", 1666.15)],
    "Nitrogen — N III]": [
        ("N III] 1746.82", 1746.82), ("N III] 1748.65", 1748.65),
        ("N III] 1749.67", 1749.67), ("N III] 1752.16", 1752.16),
        ("N III] 1753.99", 1753.99),
    ],
    "Silicon — Si III]": [("Si III] 1882.71", 1882.71), ("Si III] 1892.03", 1892.03)],
    "Carbon — C III]": [("C III] 1906.68", 1906.68), ("C III] 1908.73", 1908.73)],
}

PRESETS = {"Full prism spectrum": (0.60, 5.30)}
for group, group_lines in LINES.items():
    obs = np.array([rest * (1.0 + Z) / 10000.0 for _, rest in group_lines])
    pad = 0.06 if group == "Lyman-alpha break" else 0.045
    PRESETS[group] = (max(0.60, float(obs.min() - pad)), min(5.30, float(obs.max() + pad)))
PRESETS["All published UV-line region"] = (2.20, 3.05)

# Query observations around the actual target, then search all program-5224 PRISM x1d products.
print("QUERYING MAST")
obs = Observations.query_region(TARGET, radius=6*u.arcsec)
if len(obs) == 0:
    raise RuntimeError("No MAST observations found around MoM-z14.")

mask = np.ones(len(obs), dtype=bool)
if "proposal_id" in obs.colnames:
    mask &= np.array([str(v).strip() == PROGRAM for v in obs["proposal_id"]])
if "obs_collection" in obs.colnames:
    mask &= np.array([str(v).upper() == "JWST" for v in obs["obs_collection"]])
obs = obs[mask]
if len(obs) == 0:
    raise RuntimeError("No JWST program-5224 observations found within 6 arcsec of MoM-z14.")

products = Observations.get_product_list(obs)
pdf = products.to_pandas()
for col in pdf.columns:
    if pdf[col].dtype == object:
        pdf[col] = pdf[col].fillna("").astype(str)

name_col = "productFilename"
uri_col = "dataURI"
if name_col not in pdf or uri_col not in pdf:
    raise RuntimeError("MAST product table lacks productFilename/dataURI columns.")

pnames = pdf[name_col].str.lower()
prism = pnames.str.contains("prism") | pdf.get("description", pd.Series("", index=pdf.index)).astype(str).str.lower().str.contains("prism")
x1d = pnames.str.endswith("_x1d.fits")
candidates = pdf[prism & x1d].copy()
if len(candidates) == 0:
    # Some MAST product names omit the word prism; inspect all program-5224 x1d files.
    candidates = pdf[x1d].copy()
if len(candidates) == 0:
    raise RuntimeError("No NIRSpec x1d FITS candidates found for program 5224.")

print(f"X1D CANDIDATES: {len(candidates)}")


def direct_download(uri, filename):
    path = DATA_DIR / filename
    if path.exists() and path.stat().st_size > 0:
        return path
    url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=" + quote(str(uri), safe="")
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with open(path, "wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return path


def header_coordinate(hdul):
    keys = [("SRCRA", "SRCDEC"), ("RA_TARG", "DEC_TARG"), ("TARG_RA", "TARG_DEC"), ("RA", "DEC")]
    for hdu in hdul:
        for rak, deck in keys:
            if rak in hdu.header and deck in hdu.header:
                try:
                    return float(hdu.header[rak]), float(hdu.header[deck]), f"{hdu.name}:{rak}/{deck}"
                except Exception:
                    pass
    return None, None, "NONE"


def extract_1d(path):
    with fits.open(path, memmap=False) as hdul:
        ra, dec, source = header_coordinate(hdul)
        best = None
        for hdu in hdul:
            data = hdu.data
            names = list(getattr(data, "names", []) or [])
            upper = {n.upper(): n for n in names}
            if "WAVELENGTH" in upper and "FLUX" in upper:
                wave = np.asarray(data[upper["WAVELENGTH"]], dtype=float).ravel()
                flux = np.asarray(data[upper["FLUX"]], dtype=float).ravel()
                err_name = upper.get("FLUX_ERROR") or upper.get("ERROR") or upper.get("ERR")
                err = np.asarray(data[err_name], dtype=float).ravel() if err_name else np.full_like(flux, np.nan)
                best = (wave, flux, err, hdu.name)
                break
        return ra, dec, source, best

ranked = []
for _, row in candidates.iterrows():
    filename = row[name_col]
    try:
        path = direct_download(row[uri_col], filename)
        ra, dec, coord_source, spectrum = extract_1d(path)
        if ra is None or dec is None:
            offset = np.inf
        else:
            offset = SkyCoord(ra*u.deg, dec*u.deg).separation(TARGET).arcsec
        ranked.append((offset, path, ra, dec, coord_source, spectrum))
        print(f"{filename}  offset={offset:.4f} arcsec")
    except Exception as exc:
        print(f"SKIP {filename}: {exc}")

ranked = [item for item in ranked if item[5] is not None]
if not ranked:
    raise RuntimeError("No readable x1d spectrum found.")
ranked.sort(key=lambda item: item[0])
OFFSET, X1D_PATH, RA, DEC, COORD_SOURCE, SPEC = ranked[0]
if not np.isfinite(OFFSET) or OFFSET > 1.0:
    raise RuntimeError(f"Closest readable x1d product is still {OFFSET:.3f} arcsec from MoM-z14.")

wave, flux, err, spec_hdu = SPEC
finite = np.isfinite(wave) & np.isfinite(flux)
wave, flux, err = wave[finite], flux[finite], err[finite]
order = np.argsort(wave)
wave, flux, err = wave[order], flux[order], err[order]

print("VERIFIED MOM-Z14 PRODUCT")
print(f"FILE: {X1D_PATH}")
print(f"OFFSET: {OFFSET:.6f} arcsec")
print(f"COORD SOURCE: {COORD_SOURCE}")
print(f"SPECTRUM HDU: {spec_hdu}")
print(f"NATIVE SAMPLES: {len(wave)}")
print(f"WAVELENGTH RANGE: {wave.min():.6f} to {wave.max():.6f} um")

# Download matching s2d product when present.
stem = X1D_PATH.name.replace("_x1d.fits", "")
related = pdf[pdf[name_col].str.startswith(stem) & pdf[name_col].str.endswith("_s2d.fits")]
S2D_PATH = None
if len(related):
    r = related.iloc[0]
    S2D_PATH = direct_download(r[uri_col], r[name_col])
    print(f"MATCHING S2D: {S2D_PATH}")

native_csv = CSV_DIR / f"{VERSION}_MOM_Z14_PRISM_NATIVE_1D.csv"
pd.DataFrame({"wavelength_um": wave, "flux": flux, "flux_error": err}).to_csv(native_csv, index=False)

preset = widgets.Dropdown(options=list(PRESETS.keys()), value="Nitrogen — N IV]", description="View")
zoom = widgets.FloatRangeSlider(value=PRESETS[preset.value], min=float(wave.min()), max=float(wave.max()),
                                step=max((wave.max()-wave.min())/3000.0, 1e-5), description="Zoom μm",
                                readout_format=".5f", continuous_update=False,
                                layout=widgets.Layout(width="900px"))
cursor = widgets.FloatSlider(value=np.mean(PRESETS[preset.value]), min=float(wave.min()), max=float(wave.max()),
                             step=max((wave.max()-wave.min())/6000.0, 1e-5), description="Cursor μm",
                             readout_format=".6f", continuous_update=False,
                             layout=widgets.Layout(width="900px"))
show_error = widgets.Checkbox(value=True, description="Show ±1σ")
show_all_lines = widgets.Checkbox(value=False, description="Show all line groups")
out = widgets.Output()


def set_preset(change):
    if change.get("name") == "value":
        zoom.value = PRESETS[change["new"]]
        cursor.value = float(np.mean(PRESETS[change["new"]]))

preset.observe(set_preset, names="value")


def draw(*_):
    out.clear_output(wait=True)
    x0, x1 = zoom.value
    m = (wave >= x0) & (wave <= x1)
    if not np.any(m):
        return
    nearest = int(np.argmin(np.abs(wave - cursor.value)))
    with out:
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(14, 6.5))
        ax.plot(wave[m], flux[m], lw=0.8)
        ax.scatter(wave[m], flux[m], s=8)
        good_err = m & np.isfinite(err)
        if show_error.value and np.any(good_err):
            ax.fill_between(wave[good_err], flux[good_err]-err[good_err], flux[good_err]+err[good_err], alpha=0.16)

        groups = LINES.items() if show_all_lines.value else [(preset.value, LINES.get(preset.value, []))]
        ymax = np.nanmax(flux[m])
        ymin = np.nanmin(flux[m])
        for group_name, group_lines in groups:
            for label, rest_a in group_lines:
                obs_um = rest_a * (1.0 + Z) / 10000.0
                if x0 <= obs_um <= x1:
                    ax.axvline(obs_um, color="orange", lw=0.35, alpha=0.48)
                    ax.text(obs_um, ymax, label, rotation=90, color="orange", alpha=0.72,
                            fontsize=7, va="top", ha="right")

        ax.axvline(cursor.value, lw=0.5, ls=":", alpha=0.7)
        ax.scatter([wave[nearest]], [flux[nearest]], marker="x", s=55)
        ax.set_xlim(x0, x1)
        ax.set_xlabel("Observed wavelength [μm]")
        ax.set_ylabel("Flux [native pipeline units]")
        ax.set_title(f"MoM-z14 — verified JWST/NIRSpec PRISM — {len(wave)} native samples")
        ax.grid(alpha=0.18)
        plt.tight_layout()
        initial_png = PNG_DIR / f"{VERSION}_CURRENT_VIEW.png"
        fig.savefig(initial_png, dpi=180, bbox_inches="tight")
        plt.show()
        rest_cursor = wave[nearest] * 10000.0 / (1.0 + Z)
        print(f"VIEW                 {preset.value}")
        print(f"CURSOR OBSERVED      {wave[nearest]:.6f} μm")
        print(f"CURSOR REST          {rest_cursor:.3f} Å")
        print(f"CURSOR FLUX          {flux[nearest]:.6e}")
        print(f"TARGET OFFSET        {OFFSET:.6f} arcsec")

for control in (zoom, cursor, show_error, show_all_lines):
    control.observe(draw, names="value")

display(widgets.VBox([
    widgets.HTML("<b>JWST_0119 — MoM-z14 PRISM interactive spectrum inspector</b>"),
    widgets.HBox([preset, show_error, show_all_lines]),
    zoom,
    cursor,
    out,
]))
draw()

manifest = {
    "version": VERSION,
    "program": PROGRAM,
    "target_ra_deg": TARGET_RA,
    "target_dec_deg": TARGET_DEC,
    "redshift_adopted": Z,
    "x1d_file": str(X1D_PATH),
    "s2d_file": str(S2D_PATH) if S2D_PATH else None,
    "offset_arcsec": float(OFFSET),
    "native_samples": int(len(wave)),
    "native_csv": str(native_csv),
}
manifest_path = DATA_DIR.parent / f"{VERSION}_MANIFEST.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("OUTPUT FILES")
print(f"X1D FITS: {X1D_PATH}")
print(f"S2D FITS: {S2D_PATH}")
print(f"CSV: {native_csv}")
print(f"PNG: {PNG_DIR / (VERSION + '_CURRENT_VIEW.png')}")
print(f"MANIFEST: {manifest_path}")
print(f"# {VERSION}")
