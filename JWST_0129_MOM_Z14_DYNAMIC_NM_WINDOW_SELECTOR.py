# JWST_0129
# Interactive MoM-z14 PRISM inspector with selectable window width and wavelength-bin dropdown.

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

VERSION = "JWST_0129"
print(f"CODE OUTPUT: {VERSION}")

try:
    from google.colab import output
    output.enable_custom_widget_manager()
except Exception:
    pass

ROOT = Path("/content/JWST_OUTPUT")
FITS_DIR = ROOT / "DATA" / VERSION / "FITS"
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
for directory in (FITS_DIR, PNG_DIR, CSV_DIR):
    directory.mkdir(parents=True, exist_ok=True)

FILENAME = "jw05224-o004_s000277193_nirspec_clear-prism_x1d.fits"
URI = f"mast:JWST/product/{FILENAME}"
LOCAL = FITS_DIR / FILENAME

if not LOCAL.exists() or LOCAL.stat().st_size == 0:
    print(f"DOWNLOADING: {FILENAME}")
    url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=" + quote(URI, safe="")
    with requests.get(url, stream=True, timeout=240) as response:
        response.raise_for_status()
        with open(LOCAL, "wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    print(f"DOWNLOADED: {LOCAL}  {LOCAL.stat().st_size/1e6:.3f} MB")
else:
    print(f"USING LOCAL FILE: {LOCAL}")

with fits.open(LOCAL, memmap=False) as hdul:
    table = hdul["EXTRACT1D"].data
    names = {name.upper(): name for name in table.names}
    wave_um = np.asarray(table[names["WAVELENGTH"]], dtype=float).ravel()
    flux = np.asarray(table[names["FLUX"]], dtype=float).ravel()
    err_key = names.get("FLUX_ERROR") or names.get("ERROR") or names.get("ERR")
    err = np.asarray(table[err_key], dtype=float).ravel() if err_key else np.full_like(flux, np.nan)
    dq_key = names.get("DQ")
    dq = np.asarray(table[dq_key], dtype=int).ravel() if dq_key else np.zeros_like(flux, dtype=int)

wave_nm = wave_um * 1000.0
order = np.argsort(wave_nm)
wave_nm, flux, err, dq = wave_nm[order], flux[order], err[order], dq[order]
finite_wave = np.isfinite(wave_nm)
wave_nm, flux, err, dq = wave_nm[finite_wave], flux[finite_wave], err[finite_wave], dq[finite_wave]

native_csv = CSV_DIR / f"{VERSION}_MOM_Z14_NATIVE_X1D.csv"
pd.DataFrame({
    "wavelength_nm": wave_nm,
    "flux": flux,
    "flux_error": err,
    "dq": dq,
    "finite_flux": np.isfinite(flux),
}).to_csv(native_csv, index=False)

WIDTHS_NM = [25, 50, 100, 250, 500, 1000]
minimum_nm = float(np.floor(wave_nm.min()))
maximum_nm = float(np.ceil(wave_nm.max()))

width_dropdown = widgets.Dropdown(
    options=[(f"{value} nm", value) for value in WIDTHS_NM],
    value=25,
    description="Input 1",
    layout=widgets.Layout(width="260px"),
)

window_dropdown = widgets.Dropdown(
    options=[],
    description="Input 2",
    layout=widgets.Layout(width="430px"),
)

show_error = widgets.Checkbox(value=True, description="Show ±1σ")
show_dq0_only = widgets.Checkbox(value=False, description="DQ=0 only")
show_markers = widgets.Checkbox(value=True, description="Show native points")
out = widgets.Output()


def build_windows(width_nm):
    start0 = np.floor(minimum_nm / width_nm) * width_nm
    starts = np.arange(start0, maximum_nm, width_nm, dtype=float)
    options = []
    for start in starts:
        stop = start + width_nm
        label = f"{start:.0f}–{stop:.0f} nm"
        options.append((label, (float(start), float(stop))))
    return options


def choose_nearest_existing_window(options, old_value):
    if not options:
        return None
    if old_value is None:
        target = 2700.0
    else:
        target = 0.5 * (old_value[0] + old_value[1])
    return min((value for _, value in options), key=lambda pair: abs(0.5*(pair[0]+pair[1]) - target))


def refresh_windows(change=None):
    old_value = window_dropdown.value
    options = build_windows(width_dropdown.value)
    window_dropdown.options = options
    window_dropdown.value = choose_nearest_existing_window(options, old_value)


def draw(*_):
    out.clear_output(wait=True)
    if window_dropdown.value is None:
        return
    start_nm, stop_nm = window_dropdown.value
    mask = (wave_nm >= start_nm) & (wave_nm < stop_nm)
    valid = mask & np.isfinite(flux)
    if show_dq0_only.value:
        valid &= (dq == 0)

    rows_total = int(np.count_nonzero(mask))
    rows_flux = int(np.count_nonzero(mask & np.isfinite(flux)))
    rows_shown = int(np.count_nonzero(valid))

    with out:
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(14, 6.5))
        if rows_shown:
            ax.plot(wave_nm[valid], flux[valid], linewidth=0.75)
            if show_markers.value:
                ax.scatter(wave_nm[valid], flux[valid], s=18, zorder=3)
            err_mask = valid & np.isfinite(err)
            if show_error.value and np.any(err_mask):
                ax.fill_between(
                    wave_nm[err_mask],
                    flux[err_mask] - err[err_mask],
                    flux[err_mask] + err[err_mask],
                    alpha=0.14,
                )
        else:
            ax.text(0.5, 0.5, "No valid native samples in this window",
                    transform=ax.transAxes, ha="center", va="center", fontsize=14)

        ax.set_xlim(start_nm, stop_nm)
        ax.set_xlabel("Observed wavelength [nm]")
        ax.set_ylabel("Flux [native pipeline units]")
        ax.set_title(
            f"MoM-z14 NIRSpec/PRISM — {start_nm:.0f}–{stop_nm:.0f} nm | "
            f"native rows={rows_total}, finite flux={rows_flux}, displayed={rows_shown}"
        )
        ax.grid(alpha=0.16)
        ax.text(
            0.015, 0.97,
            f"Window width: {stop_nm-start_nm:.0f} nm\n"
            f"Native wavelength rows: {rows_total}\n"
            f"Finite-flux rows: {rows_flux}\n"
            f"Displayed rows: {rows_shown}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", alpha=0.22),
        )
        plt.tight_layout()
        png = PNG_DIR / f"{VERSION}_CURRENT_WINDOW.png"
        fig.savefig(png, dpi=220, bbox_inches="tight")
        plt.show()

        selected = pd.DataFrame({
            "wavelength_nm": wave_nm[mask],
            "flux": flux[mask],
            "flux_error": err[mask],
            "dq": dq[mask],
        })
        selected_csv = CSV_DIR / f"{VERSION}_CURRENT_WINDOW.csv"
        selected.to_csv(selected_csv, index=False)
        print(f"WINDOW WIDTH          {stop_nm-start_nm:.0f} nm")
        print(f"SELECTED WINDOW       {start_nm:.0f} to {stop_nm:.0f} nm")
        print(f"NATIVE ROWS           {rows_total}")
        print(f"FINITE-FLUX ROWS      {rows_flux}")
        print(f"DISPLAYED ROWS        {rows_shown}")
        print(f"WINDOW CSV            {selected_csv}")
        print(f"CURRENT PNG           {png}")


width_dropdown.observe(refresh_windows, names="value")
width_dropdown.observe(draw, names="value")
window_dropdown.observe(draw, names="value")
for control in (show_error, show_dq0_only, show_markers):
    control.observe(draw, names="value")

refresh_windows()
controls = widgets.VBox([
    widgets.HBox([width_dropdown, window_dropdown]),
    widgets.HBox([show_error, show_dq0_only, show_markers]),
])
display(controls, out)
draw()

print("\nOUTPUT SUMMARY")
print(f"Native CSV: {native_csv}")
print(f"FITS file:  {LOCAL}")
print(f"Timestamp UTC: {datetime.now(timezone.utc).isoformat()}")
print(f"# {VERSION}")
