# JWST_0128
# Interactive 15-nm native-sampling inspector for MoM-z14 PRISM products.

from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import quote
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

VERSION = "JWST_0128"
print(f"CODE OUTPUT: {VERSION}")
print("MOM-Z14 15-NM NATIVE-SAMPLING WINDOW WIZARD")
print("-" * 112)

ROOT = Path("/content/JWST_OUTPUT")
FITS_DIRS = [
    ROOT / "DATA" / "JWST_0127" / "FITS",
    ROOT / "DATA" / "JWST_0126" / "FITS",
    ROOT / "DATA" / VERSION / "FITS",
]
CSV_DIR = ROOT / "CSV"
PNG_DIR = ROOT / "PNG"
for p in (FITS_DIRS[-1], CSV_DIR, PNG_DIR):
    p.mkdir(parents=True, exist_ok=True)

FILES = {
    "x1d": "jw05224-o004_s000277193_nirspec_clear-prism_x1d.fits",
    "s2d": "jw05224-o004_s000277193_nirspec_clear-prism_s2d.fits",
    "crf": "jw05224-o004_s000277193_nirspec_clear-prism_crf.fits",
}


def find_or_download(filename):
    for d in FITS_DIRS:
        p = d / filename
        if p.exists() and p.stat().st_size > 0:
            print(f"FOUND: {p}")
            return p
    p = FITS_DIRS[-1] / filename
    uri = f"mast:JWST/product/{filename}"
    url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=" + quote(uri, safe="")
    print(f"DOWNLOADING: {filename}")
    with requests.get(url, stream=True, timeout=240) as r:
        r.raise_for_status()
        with open(p, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"DOWNLOADED: {p}  {p.stat().st_size/1e6:.3f} MB")
    return p


paths = {k: find_or_download(v) for k, v in FILES.items()}

with fits.open(paths["x1d"], memmap=False) as hdul:
    tab = hdul["EXTRACT1D"].data
    names = {n.upper(): n for n in tab.names}
    wave_um = np.asarray(tab[names["WAVELENGTH"]], float).ravel()
    flux = np.asarray(tab[names["FLUX"]], float).ravel()
    err_name = names.get("FLUX_ERROR") or names.get("ERROR") or names.get("ERR")
    err = np.asarray(tab[err_name], float).ravel() if err_name else np.full_like(flux, np.nan)
    dq_name = names.get("DQ")
    dq = np.asarray(tab[dq_name]).ravel() if dq_name else np.zeros_like(flux, dtype=int)

order = np.argsort(wave_um)
wave_um, flux, err, dq = wave_um[order], flux[order], err[order], dq[order]
wave_nm = wave_um * 1000.0
finite_wave = np.isfinite(wave_nm)
valid = finite_wave & np.isfinite(flux) & (dq == 0)


def load_image_cube(path, preferred="SCI"):
    with fits.open(path, memmap=False) as hdul:
        if preferred in hdul and hdul[preferred].data is not None:
            arr = np.asarray(hdul[preferred].data, dtype=float)
        else:
            arr = None
            for hdu in hdul:
                if hdu.data is not None and np.ndim(hdu.data) == 2:
                    arr = np.asarray(hdu.data, dtype=float)
                    break
        if arr is None:
            raise RuntimeError(f"No 2-D image found in {path.name}")
        return arr


s2d = load_image_cube(paths["s2d"])
crf = load_image_cube(paths["crf"])

# Build 15-nm windows aligned to 15-nm boundaries and covering the full native range.
start_nm = float(np.floor(np.nanmin(wave_nm) / 15.0) * 15.0)
stop_nm = float(np.ceil(np.nanmax(wave_nm) / 15.0) * 15.0)
edges = np.arange(start_nm, stop_nm + 15.0, 15.0)
windows = []
rows = []
for i in range(len(edges) - 1):
    lo, hi = float(edges[i]), float(edges[i + 1])
    m = (wave_nm >= lo) & (wave_nm < hi if i < len(edges)-2 else wave_nm <= hi)
    n_native = int(np.count_nonzero(m & finite_wave))
    n_valid = int(np.count_nonzero(m & valid))
    label = f"{lo:7.0f}–{hi:7.0f} nm   | native {n_native:2d} | valid {n_valid:2d}"
    windows.append((label, i))
    local = np.diff(wave_nm[m & finite_wave])
    rows.append({
        "window_index": i,
        "start_nm": lo,
        "end_nm": hi,
        "native_rows": n_native,
        "valid_rows": n_valid,
        "median_step_nm": float(np.nanmedian(local)) if local.size else np.nan,
        "min_step_nm": float(np.nanmin(local)) if local.size else np.nan,
        "max_step_nm": float(np.nanmax(local)) if local.size else np.nan,
    })

counts_csv = CSV_DIR / f"{VERSION}_15NM_WINDOW_COUNTS.csv"
pd.DataFrame(rows).to_csv(counts_csv, index=False)

print(f"X1D NATIVE ROWS: {len(wave_nm)}")
print(f"X1D VALID ROWS:  {int(np.count_nonzero(valid))}")
print(f"S2D SHAPE:       {s2d.shape}")
print(f"CRF SHAPE:       {crf.shape}")
print(f"15-NM WINDOWS:   {len(windows)}")

selector = widgets.Dropdown(options=windows, value=0, description="15 nm window:",
                            layout=widgets.Layout(width="720px"))
view = widgets.ToggleButtons(options=[("1-D spectrum", "x1d"), ("S2D strip", "s2d"), ("CRF strip", "crf")],
                             value="x1d", description="View:")
show_error = widgets.Checkbox(value=True, description="Show ±1σ")
show_markers = widgets.Checkbox(value=True, description="Show native points")
prev_button = widgets.Button(description="◀ Previous", button_style="")
next_button = widgets.Button(description="Next ▶", button_style="")
out = widgets.Output()


def current_bounds():
    i = int(selector.value)
    return i, float(edges[i]), float(edges[i+1])


def move(delta):
    i = max(0, min(len(windows)-1, int(selector.value) + delta))
    selector.value = i

prev_button.on_click(lambda _: move(-1))
next_button.on_click(lambda _: move(1))


def column_slice(arr, lo, hi):
    # Rectified products have one spectral column per X1D wavelength row.
    cols = np.where((wave_nm >= lo) & (wave_nm <= hi))[0]
    if cols.size == 0:
        center = int(np.argmin(np.abs(wave_nm - 0.5*(lo+hi))))
        cols = np.array([center])
    c0, c1 = max(0, int(cols.min())-1), min(arr.shape[1], int(cols.max())+2)
    return arr[:, c0:c1], c0, c1


def draw(*_):
    i, lo, hi = current_bounds()
    m = (wave_nm >= lo) & (wave_nm <= hi)
    n_native = int(np.count_nonzero(m & finite_wave))
    n_valid = int(np.count_nonzero(m & valid))
    out.clear_output(wait=True)
    with out:
        plt.style.use("dark_background")
        if view.value == "x1d":
            fig, ax = plt.subplots(figsize=(14, 6.5))
            ax.plot(wave_nm[m], flux[m], lw=0.65, alpha=0.90)
            if show_markers.value:
                ax.scatter(wave_nm[m], flux[m], s=22, zorder=3)
            ge = m & np.isfinite(err) & np.isfinite(flux)
            if show_error.value and np.any(ge):
                ax.fill_between(wave_nm[ge], flux[ge]-err[ge], flux[ge]+err[ge], alpha=0.13)
            ax.set_xlim(lo, hi)
            ax.set_xlabel("Observed wavelength [nm]")
            ax.set_ylabel("Flux [native pipeline units]")
            ax.set_title(f"MoM-z14 PRISM — 15 nm window {lo:.0f}–{hi:.0f} nm | native={n_native} valid={n_valid}")
            ax.grid(alpha=0.16)
            if n_native == 0:
                ax.text(0.5, 0.5, "NO NATIVE WAVELENGTH SAMPLE IN THIS 15-nm WINDOW",
                        transform=ax.transAxes, ha="center", va="center", fontsize=13)
        else:
            arr = s2d if view.value == "s2d" else crf
            cut, c0, c1 = column_slice(arr, lo, hi)
            fig, ax = plt.subplots(figsize=(14, 6.5))
            finite = cut[np.isfinite(cut)]
            if finite.size:
                vmin, vmax = np.nanpercentile(finite, [5, 95])
            else:
                vmin, vmax = None, None
            ax.imshow(cut, origin="lower", aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax,
                      extent=[wave_nm[c0], wave_nm[min(c1-1, len(wave_nm)-1)], 0, cut.shape[0]])
            ax.set_xlim(lo, hi)
            ax.set_xlabel("Observed wavelength [nm]")
            ax.set_ylabel("Spatial row")
            ax.set_title(f"MoM-z14 {view.value.upper()} — 15 nm window {lo:.0f}–{hi:.0f} nm | columns {c0}:{c1}")
        plt.tight_layout()
        png = PNG_DIR / f"{VERSION}_CURRENT_15NM_WINDOW.png"
        fig.savefig(png, dpi=220, bbox_inches="tight")
        plt.show()
        print(f"WINDOW INDEX          {i+1} / {len(windows)}")
        print(f"WINDOW                {lo:.3f} to {hi:.3f} nm")
        print(f"NATIVE X1D ROWS       {n_native}")
        print(f"VALID DQ=0 ROWS       {n_valid}")
        if n_native:
            w = wave_nm[m & finite_wave]
            print("NATIVE WAVELENGTHS    " + ", ".join(f"{x:.4f}" for x in w))
        print(f"CURRENT PNG           {png}")

for control in (selector, view, show_error, show_markers):
    control.observe(draw, names="value")

display(widgets.VBox([
    widgets.HBox([prev_button, next_button]),
    selector,
    view,
    widgets.HBox([show_error, show_markers]),
    out,
]))
draw()

print("\nOUTPUT SUMMARY")
print(f"15-nm count table: {counts_csv}")
print(f"Current plot PNG:  {PNG_DIR / f'{VERSION}_CURRENT_15NM_WINDOW.png'}")
print(f"Timestamp UTC:     {datetime.now(timezone.utc).isoformat()}")
print(f"# {VERSION}")
