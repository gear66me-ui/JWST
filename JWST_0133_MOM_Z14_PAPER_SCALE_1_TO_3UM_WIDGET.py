# JWST_0133
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone

for pkg in ["astropy", "ipywidgets", "requests", "pandas", "matplotlib", "numpy"]:
    try:
        __import__(pkg)
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from astropy.io import fits
from IPython.display import display, clear_output
import ipywidgets as widgets

VERSION = "JWST_0133"
ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
DATA_DIR = ROOT / "DATA" / VERSION
for d in (PNG_DIR, CSV_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

print(f"CODE OUTPUT: {VERSION}")
print("MOM-Z14 PAPER-SCALE SPECTRUM WIDGET — 1.0 TO 3.0 MICRONS")
print("-" * 112)

SPEC_NAME = "mom-cos04-v4_prism-clear_5224_277193.spec.fits"
SPEC_URL = f"https://s3.amazonaws.com/msaexp-nirspec/extractions/mom-cos04-v4/{SPEC_NAME}"
SPEC_PATH = DATA_DIR / SPEC_NAME

if not SPEC_PATH.exists() or SPEC_PATH.stat().st_size < 100000:
    print(f"DOWNLOADING: {SPEC_NAME}")
    with requests.get(SPEC_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(SPEC_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
else:
    print(f"USING CACHED FILE: {SPEC_PATH}")

with fits.open(SPEC_PATH, memmap=False) as hdul:
    tab = hdul["SPEC1D"].data
    names = [n.upper() for n in tab.names]
    name_map = {n.upper(): n for n in tab.names}

    def pick(*candidates):
        for c in candidates:
            if c.upper() in name_map:
                return np.asarray(tab[name_map[c.upper()]], dtype=float)
        return None

    wave = pick("wave", "wavelength", "lam")
    flux = pick("flux", "flux_corr", "fnu")
    err = pick("err", "error", "flux_err", "sigma")
    if err is None:
        ivar = pick("ivar", "inverse_variance")
        if ivar is not None:
            err = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.nan)
    if wave is None or flux is None:
        raise RuntimeError(f"Required SPEC1D columns not found. Columns={tab.names}")

    sci = np.asarray(hdul["SCI"].data, dtype=float)
    wht = np.asarray(hdul["WHT"].data, dtype=float)

# Flatten any singleton dimensions
wave = np.ravel(wave)
flux = np.ravel(flux)
err = np.ravel(err) if err is not None else np.full_like(flux, np.nan)

# DJA SPEC1D wavelength is microns; flux is normally microJy.
# Convert f_nu [microJy] to f_lambda [1e-20 erg s^-1 cm^-2 Angstrom^-1].
# f_lambda(per Angstrom) = f_nu(Jy)*1e-23 * c[Angstrom/s] / lambda[Angstrom]^2
# microJy = 1e-6 Jy; divide by 1e-20 for paper units.
c_ang_s = 2.99792458e18
lam_ang = wave * 1.0e4
conv = 1.0e-6 * 1.0e-23 * c_ang_s / (lam_ang ** 2) / 1.0e-20
flam20 = flux * conv
err20 = err * conv

valid = np.isfinite(wave) & np.isfinite(flam20)
wave = wave[valid]
flam20 = flam20[valid]
err20 = err20[valid]
order = np.argsort(wave)
wave, flam20, err20 = wave[order], flam20[order], err20[order]

# 2-D signal-to-noise-like panel
sn2d = sci * np.sqrt(np.clip(wht, 0, None))
finite_sn = sn2d[np.isfinite(sn2d)]
lim = np.nanpercentile(np.abs(finite_sn), 98.5) if finite_sn.size else 1.0
if not np.isfinite(lim) or lim <= 0:
    lim = 1.0

# Map 2-D columns to wavelength using full SPEC1D length when compatible
with fits.open(SPEC_PATH, memmap=False) as hdul:
    full_tab = hdul["SPEC1D"].data
    full_names = {n.upper(): n for n in full_tab.names}
    full_wave = np.ravel(np.asarray(full_tab[full_names.get("WAVE", full_names.get("WAVELENGTH"))], dtype=float))
if len(full_wave) != sn2d.shape[1]:
    full_wave = np.linspace(np.nanmin(wave), np.nanmax(wave), sn2d.shape[1])

# Export exact converted data
out_df = pd.DataFrame({
    "wavelength_um": wave,
    "flux_flam_1e-20_erg_s_cm2_A": flam20,
    "error_flam_1e-20_erg_s_cm2_A": err20,
})
csv_path = CSV_DIR / f"{VERSION}_MOM_Z14_DJA_PAPER_UNITS.csv"
out_df.to_csv(csv_path, index=False)

LINES = {
    "Lyα": 0.121567 * (1 + 14.44),
    "N IV]": 0.1487 * (1 + 14.44),
    "C IV": 0.15495 * (1 + 14.44),
    "He II": 0.1640 * (1 + 14.44),
    "O III]": 0.1663 * (1 + 14.44),
    "N III]": 0.1750 * (1 + 14.44),
    "C III]": 0.1908 * (1 + 14.44),
}

WINDOWS = {
    "Full paper region 1.0–3.0 μm": (1.0, 3.0),
    "1.0–1.5 μm": (1.0, 1.5),
    "1.5–2.0 μm": (1.5, 2.0),
    "Lyα break 1.75–2.05 μm": (1.75, 2.05),
    "N IV] 2.20–2.36 μm": (2.20, 2.36),
    "C IV 2.32–2.46 μm": (2.32, 2.46),
    "He II / O III] 2.46–2.64 μm": (2.46, 2.64),
    "N III] 2.62–2.78 μm": (2.62, 2.78),
    "C III] 2.86–3.00 μm": (2.86, 3.00),
    "2.0–2.5 μm": (2.0, 2.5),
    "2.5–3.0 μm": (2.5, 3.0),
}

window_dd = widgets.Dropdown(options=list(WINDOWS.keys()), value="Full paper region 1.0–3.0 μm", description="Window:", layout=widgets.Layout(width="520px"))
ymin_box = widgets.FloatText(value=-0.10, description="Y min:", layout=widgets.Layout(width="180px"))
ymax_box = widgets.FloatText(value=0.50, description="Y max:", layout=widgets.Layout(width="180px"))
save_btn = widgets.Button(description="Save current PNG", button_style="success")
out = widgets.Output()

last_fig = {"fig": None, "name": None}

def draw(*_):
    x0, x1 = WINDOWS[window_dd.value]
    m = (wave >= x0) & (wave <= x1)
    cols = (full_wave >= x0) & (full_wave <= x1)
    with out:
        clear_output(wait=True)
        fig = plt.figure(figsize=(14, 8.5), facecolor="white")
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 4.2], hspace=0.03)
        ax2 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharex=ax2)

        if np.any(cols):
            sub = sn2d[:, cols]
            extent = [x0, x1, -0.5, sub.shape[0]-0.5]
            ax2.imshow(sub, origin="lower", aspect="auto", extent=extent,
                       cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        ax2.set_ylabel("2-D S/N", fontsize=11)
        ax2.set_yticks([])
        ax2.tick_params(axis="x", labelbottom=False)
        for s in ax2.spines.values():
            s.set_linewidth(1.2)

        if np.any(m):
            ax1.errorbar(wave[m], flam20[m], yerr=err20[m], fmt="none",
                         ecolor="#9dd9ee", elinewidth=1.2, alpha=0.75, capsize=0, zorder=1)
            ax1.step(wave[m], flam20[m], where="mid", color="#152a8a", linewidth=1.8, zorder=2)
        for label, xpos in LINES.items():
            if x0 <= xpos <= x1:
                ax1.axvline(xpos, color="0.72", linestyle="--", linewidth=1.0, zorder=0)
                ax1.text(xpos, ymax_box.value * 0.97, label, rotation=90,
                         ha="right", va="top", fontsize=10, color="0.18")

        ax1.set_xlim(x0, x1)
        ax1.set_ylim(ymin_box.value, ymax_box.value)
        ax1.set_xlabel(r"Observed wavelength, $\lambda_{obs}$ [$\mu$m]", fontsize=14)
        ax1.set_ylabel(r"$f_\lambda$ [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=13)
        ax1.tick_params(labelsize=11, direction="out", length=5, width=1.1)
        ax1.grid(False)
        for s in ax1.spines.values():
            s.set_linewidth(1.2)

        npts = int(np.count_nonzero(m))
        fig.suptitle(f"MoM-z14 — DJA/msaexp PRISM/CLEAR | {x0:.2f}–{x1:.2f} μm | {npts} native samples",
                     fontsize=14, y=0.985)
        fig.tight_layout(rect=[0.02, 0.02, 0.995, 0.965])
        plt.show()
        last_fig["fig"] = fig
        last_fig["name"] = f"{VERSION}_MOM_Z14_{x0:.2f}_{x1:.2f}UM_PAPER_STYLE.png".replace(".", "p")

def save_current(_):
    if last_fig["fig"] is None:
        return
    path = PNG_DIR / last_fig["name"]
    last_fig["fig"].savefig(path, dpi=500, bbox_inches="tight")
    print(f"SAVED: {path}")

window_dd.observe(draw, names="value")
ymin_box.observe(draw, names="value")
ymax_box.observe(draw, names="value")
save_btn.on_click(save_current)

display(widgets.HBox([window_dd, ymin_box, ymax_box, save_btn]))
display(out)
draw()

# Save default full paper-region figure automatically
if last_fig["fig"] is not None:
    default_png = PNG_DIR / f"{VERSION}_MOM_Z14_1p0_TO_3p0UM_PAPER_STYLE.png"
    last_fig["fig"].savefig(default_png, dpi=500, bbox_inches="tight")
else:
    default_png = None

print("\nOUTPUT SUMMARY")
print(f"DJA spectrum:       {SPEC_PATH}")
print(f"Paper-units CSV:    {csv_path}")
print(f"Default paper PNG:  {default_png}")
print(f"Native samples:     {len(wave)}")
print(f"Timestamp UTC:      {datetime.now(timezone.utc).isoformat()}")
print(f"# {VERSION}")
