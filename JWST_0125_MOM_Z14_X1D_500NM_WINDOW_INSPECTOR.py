# JWST_0125
# Download the verified MoM-z14 NIRSpec PRISM x1d FITS product and inspect native samples.

from pathlib import Path
from urllib.parse import quote
from datetime import datetime, timezone
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import ipywidgets as widgets
from IPython.display import display

try:
    from google.colab import output
    output.enable_custom_widget_manager()
except Exception:
    pass

VERSION = "JWST_0125"
SOURCE_ID = "277193"
FILENAME = "jw05224-o004_s000277193_nirspec_clear-prism_x1d.fits"
URI = f"mast:JWST/product/{FILENAME}"

ROOT = Path("/content/JWST_OUTPUT")
DATA_DIR = ROOT / "DATA" / VERSION / "FITS"
CSV_DIR = ROOT / "CSV"
PNG_DIR = ROOT / "PNG"
for directory in (DATA_DIR, CSV_DIR, PNG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

print(f"CODE OUTPUT: {VERSION}")
print(f"SOURCE: MoM-z14 / program 5224 / source {SOURCE_ID}")
print(f"PRODUCT: {FILENAME}")

path = DATA_DIR / FILENAME
if not path.exists() or path.stat().st_size == 0:
    url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=" + quote(URI, safe="")
    print("DOWNLOADING VERIFIED X1D FITS")
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with open(path, "wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
else:
    print("USING EXISTING LOCAL FITS")


def extract_spectrum(fits_path):
    with fits.open(fits_path, memmap=False) as hdul:
        for hdu in hdul:
            data = hdu.data
            names = list(getattr(data, "names", []) or [])
            lookup = {name.upper(): name for name in names}
            if "WAVELENGTH" in lookup and "FLUX" in lookup:
                wave = np.asarray(data[lookup["WAVELENGTH"]], dtype=float).ravel()
                flux = np.asarray(data[lookup["FLUX"]], dtype=float).ravel()
                err_key = lookup.get("FLUX_ERROR") or lookup.get("ERROR") or lookup.get("ERR")
                dq_key = lookup.get("DQ")
                error = np.asarray(data[err_key], dtype=float).ravel() if err_key else np.full_like(flux, np.nan)
                dq = np.asarray(data[dq_key]).ravel() if dq_key else np.zeros_like(flux, dtype=int)
                return hdu.name, wave, flux, error, dq
    raise RuntimeError("No WAVELENGTH/FLUX table found in the X1D FITS file.")

hdu_name, wave_um, flux, error, dq = extract_spectrum(path)
wave_nm = wave_um * 1000.0
finite_wave = np.isfinite(wave_nm)
finite_flux = np.isfinite(flux)
valid = finite_wave & finite_flux
order = np.argsort(wave_nm[valid])
wave_nm = wave_nm[valid][order]
flux = flux[valid][order]
error = error[valid][order]
dq = dq[valid][order]

csv_path = CSV_DIR / f"{VERSION}_MOM_Z14_NATIVE_X1D.csv"
pd.DataFrame({
    "wavelength_nm": wave_nm,
    "flux_jy": flux,
    "flux_error_jy": error,
    "dq": dq,
}).to_csv(csv_path, index=False)

print(f"SPECTRUM HDU: {hdu_name}")
print(f"TOTAL NATIVE FINITE POINTS: {len(wave_nm)}")
print(f"WAVELENGTH RANGE: {wave_nm.min():.3f} to {wave_nm.max():.3f} nm")

windows = {"Full native spectrum": (float(wave_nm.min()), float(wave_nm.max()))}
for start in range(0, 5000, 500):
    windows[f"{start:04d}–{start + 500:04d} nm"] = (float(start), float(start + 500))

selector = widgets.Dropdown(
    options=list(windows.keys()),
    value="1500–2000 nm",
    description="Window",
    layout=widgets.Layout(width="420px"),
)
show_points = widgets.Checkbox(value=True, description="Show native points")
show_error = widgets.Checkbox(value=False, description="Show ±1σ")
quality_only = widgets.Checkbox(value=False, description="DQ = 0 only")
out = widgets.Output()


def draw(*_):
    out.clear_output(wait=True)
    xmin, xmax = windows[selector.value]
    mask = (wave_nm >= xmin) & (wave_nm <= xmax)
    if quality_only.value:
        mask &= dq == 0
    count = int(np.count_nonzero(mask))

    with out:
        if count == 0:
            print(f"No native samples in {selector.value}")
            return

        x = wave_nm[mask]
        y = flux[mask]
        e = error[mask]

        fig, ax = plt.subplots(figsize=(15, 6.5))
        ax.plot(x, y, linewidth=0.65, label="Native X1D flux")
        if show_points.value:
            ax.scatter(x, y, s=9, zorder=3, label="Native samples")
        good_error = np.isfinite(e)
        if show_error.value and np.any(good_error):
            ax.fill_between(x[good_error], y[good_error] - e[good_error], y[good_error] + e[good_error], alpha=0.14, label="±1σ")

        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("Observed wavelength [nm]")
        ax.set_ylabel("Flux [Jy]")
        ax.set_title(
            f"MoM-z14 — JWST/NIRSpec PRISM X1D — {selector.value}\n"
            f"{count} native plotted points | {len(wave_nm)} total finite native points"
        )
        ax.grid(alpha=0.18)
        ax.legend(loc="best")
        fig.tight_layout()

        safe = selector.value.replace("–", "-").replace(" ", "_")
        png_path = PNG_DIR / f"{VERSION}_{safe}.png"
        fig.savefig(png_path, dpi=190, bbox_inches="tight")
        plt.show()

        print(f"WINDOW                  {selector.value}")
        print(f"PLOTTED NATIVE POINTS   {count}")
        print(f"TOTAL NATIVE POINTS     {len(wave_nm)}")
        print(f"WINDOW DATA RANGE       {x.min():.3f} to {x.max():.3f} nm")
        print(f"PLOT PNG                {png_path}")

for control in (selector, show_points, show_error, quality_only):
    control.observe(draw, names="value")

display(widgets.HBox([selector, show_points, show_error, quality_only]))
display(out)
draw()

print("\nOUTPUT SUMMARY")
print(f"FITS: {path}")
print(f"CSV:  {csv_path}")
print(f"PNG DIRECTORY: {PNG_DIR}")
print(f"Timestamp UTC: {datetime.now(timezone.utc).isoformat()}")
print(f"# {VERSION}")
