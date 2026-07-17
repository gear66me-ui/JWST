# JWST_0126
# Download and audit the exact native sampling of the MoM-z14 PRISM X1D and S2D products.

from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import quote
import requests
import numpy as np
import pandas as pd
from astropy.io import fits

VERSION = "JWST_0126"
SOURCE = "277193"
STEM = "jw05224-o004_s000277193_nirspec_clear-prism"
FILES = {
    "X1D": f"{STEM}_x1d.fits",
    "S2D": f"{STEM}_s2d.fits",
}

ROOT = Path("/content/JWST_OUTPUT")
DATA_DIR = ROOT / "DATA" / VERSION / "FITS"
CSV_DIR = ROOT / "CSV"
for directory in (DATA_DIR, CSV_DIR):
    directory.mkdir(parents=True, exist_ok=True)

print(f"CODE OUTPUT: {VERSION}")
print("MOM-Z14 NIRSPEC PRISM NATIVE PIXEL / SAMPLING AUDIT")
print("-" * 108)


def download_product(filename):
    path = DATA_DIR / filename
    if path.exists() and path.stat().st_size > 0:
        print(f"USING CACHED FILE: {path}")
        return path
    uri = f"mast:JWST/product/{filename}"
    url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=" + quote(uri, safe="")
    print(f"DOWNLOADING: {filename}")
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    print(f"DOWNLOADED: {path}  {path.stat().st_size / 1e6:.3f} MB")
    return path


paths = {kind: download_product(name) for kind, name in FILES.items()}

# ---------- X1D audit ----------
x1d_rows = []
wave = flux = err = dq = None
with fits.open(paths["X1D"], memmap=False) as hdul:
    print("\nX1D HDU INVENTORY")
    for index, hdu in enumerate(hdul):
        shape = None if hdu.data is None else np.shape(hdu.data)
        print(f"  {index:2d}  {hdu.name:<12s}  shape={shape}")
        data = hdu.data
        names = list(getattr(data, "names", []) or [])
        upper = {name.upper(): name for name in names}
        if wave is None and "WAVELENGTH" in upper and "FLUX" in upper:
            wave = np.asarray(data[upper["WAVELENGTH"]], dtype=float).ravel()
            flux = np.asarray(data[upper["FLUX"]], dtype=float).ravel()
            err_name = upper.get("FLUX_ERROR") or upper.get("ERROR") or upper.get("ERR")
            dq_name = upper.get("DQ")
            err = np.asarray(data[err_name], dtype=float).ravel() if err_name else np.full_like(flux, np.nan)
            dq = np.asarray(data[dq_name]).ravel() if dq_name else np.zeros_like(flux, dtype=np.int64)

if wave is None:
    raise RuntimeError("No WAVELENGTH/FLUX table found in the X1D product.")

finite_wave = np.isfinite(wave)
finite_flux = finite_wave & np.isfinite(flux)
valid_dq = finite_flux & (dq == 0)
order = np.argsort(wave[finite_wave])
wave_sorted = wave[finite_wave][order]
delta_um = np.diff(wave_sorted)

print("\nX1D EXACT COUNTS")
print(f"  TABLE ROWS / NATIVE SAMPLES       {len(wave):8d}")
print(f"  FINITE WAVELENGTH ROWS             {finite_wave.sum():8d}")
print(f"  FINITE WAVELENGTH + FLUX ROWS      {finite_flux.sum():8d}")
print(f"  DQ=0 FINITE ROWS                   {valid_dq.sum():8d}")
print(f"  WAVELENGTH RANGE [um]              {np.nanmin(wave):8.6f} to {np.nanmax(wave):8.6f}")
if len(delta_um):
    print(f"  MEDIAN NATIVE STEP [nm]            {np.nanmedian(delta_um) * 1000:8.4f}")
    print(f"  MINIMUM NATIVE STEP [nm]           {np.nanmin(delta_um) * 1000:8.4f}")
    print(f"  MAXIMUM NATIVE STEP [nm]           {np.nanmax(delta_um) * 1000:8.4f}")

# 500-nm windows from 0 to 5500 nm.
window_rows = []
wave_nm = wave * 1000.0
for lo in range(0, 5500, 500):
    hi = lo + 500
    in_window = finite_wave & (wave_nm >= lo) & (wave_nm < hi)
    in_finite = finite_flux & (wave_nm >= lo) & (wave_nm < hi)
    in_valid = valid_dq & (wave_nm >= lo) & (wave_nm < hi)
    local = np.sort(wave_nm[in_window])
    local_step = np.diff(local)
    row = {
        "window_nm": f"{lo}-{hi}",
        "native_rows": int(in_window.sum()),
        "finite_flux_rows": int(in_finite.sum()),
        "dq0_rows": int(in_valid.sum()),
        "median_step_nm": float(np.nanmedian(local_step)) if len(local_step) else np.nan,
        "min_step_nm": float(np.nanmin(local_step)) if len(local_step) else np.nan,
        "max_step_nm": float(np.nanmax(local_step)) if len(local_step) else np.nan,
    }
    window_rows.append(row)

window_df = pd.DataFrame(window_rows)
window_csv = CSV_DIR / f"{VERSION}_500NM_NATIVE_SAMPLE_COUNTS.csv"
window_df.to_csv(window_csv, index=False)

print("\n500-NM WINDOW COUNTS")
print(window_df.to_string(index=False, justify="right", float_format=lambda value: f"{value:.4f}"))

# ---------- S2D audit ----------
s2d_rows = []
print("\nS2D ARRAY INVENTORY")
with fits.open(paths["S2D"], memmap=False) as hdul:
    for index, hdu in enumerate(hdul):
        data = hdu.data
        shape = None if data is None else tuple(np.shape(data))
        total = 0 if data is None else int(np.size(data))
        print(f"  {index:2d}  {hdu.name:<12s}  shape={str(shape):<18s} elements={total}")
        s2d_rows.append({
            "hdu_index": index,
            "hdu_name": hdu.name,
            "shape": str(shape),
            "total_elements": total,
        })

s2d_csv = CSV_DIR / f"{VERSION}_S2D_HDU_ARRAY_SHAPES.csv"
pd.DataFrame(s2d_rows).to_csv(s2d_csv, index=False)

native_csv = CSV_DIR / f"{VERSION}_X1D_NATIVE_ROWS.csv"
pd.DataFrame({
    "wavelength_um": wave,
    "wavelength_nm": wave_nm,
    "flux": flux,
    "flux_error": err,
    "dq": dq,
    "finite_flux": finite_flux,
    "dq0_valid": valid_dq,
}).to_csv(native_csv, index=False)

print("\nINTERPRETATION")
print("  25 samples in a 500-nm window would imply about 20 nm per sample, not 2.5 um per sample.")
print("  The table above reports the actual count and wavelength spacing directly from the FITS array.")

print("\nOUTPUT SUMMARY")
print(f"X1D native rows: {native_csv}")
print(f"500-nm counts:   {window_csv}")
print(f"S2D shapes:      {s2d_csv}")
print(f"FITS directory:  {DATA_DIR}")
print(f"Timestamp UTC:   {datetime.now(timezone.utc).isoformat()}")
print(f"# {VERSION}")
