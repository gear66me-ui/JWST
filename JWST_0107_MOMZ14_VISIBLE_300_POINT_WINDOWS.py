#!/usr/bin/env python3
"""
JWST_0107_MOMZ14_VISIBLE_300_POINT_WINDOWS.py

Plot exactly 300 VISIBLE display markers in each of six MoM-z14 rest-frame
windows. The 300-point curve is a shape-preserving PCHIP resampling of the
exact Stage-3 PRISM/CLEAR X1D spectrum. Native measured bins remain separately
visible as large white outlined markers. No synthetic peaks, random noise,
Gaussian/Voigt line models, or AI-generated images.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def ensure_packages() -> None:
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "scipy": "scipy",
    }
    missing = [pip_name for module, pip_name in required.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_packages()

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

VERSION = "JWST_0107"
TARGET = "MoM-z14"
Z = 14.44
DISPLAY_POINTS = 300

ROOT = Path("/content/JWST_OUTPUT")
PNG = ROOT / "PNG"
CSV = ROOT / "CSV"
DATA = ROOT / "DATA" / VERSION
for directory in (PNG, CSV, DATA):
    directory.mkdir(parents=True, exist_ok=True)

SOURCE_NAME = "JWST_0098_04_jw05224-o004_s000277193_nirspec_clear-prism_x1d_HDU1_NATIVE_X1D.csv"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/JWST/main/"
    "artifacts/MOMZ14/JWST_0101/" + SOURCE_NAME
)
SOURCE_PATH = DATA / SOURCE_NAME

REGIONS = [
    {"key": "LYMAN_ALPHA_BREAK", "title": "H I Lyman-alpha break", "center_A": 1215.67,
     "color": "#ff9f1c", "lines": [("H I Ly-alpha", 1215.67)]},
    {"key": "N_IV", "title": "N IV] doublet", "center_A": 1486.50,
     "color": "#ff4fa3", "lines": [("N IV] 1483", 1483.32), ("N IV] 1487", 1486.50)]},
    {"key": "C_IV", "title": "C IV doublet", "center_A": 1550.00,
     "color": "#25c7f7", "lines": [("C IV 1548", 1548.20), ("C IV 1551", 1550.77)]},
    {"key": "HEII_OIII", "title": "He II + O III] complex", "center_A": 1650.00,
     "color": "#b978ff", "lines": [("He II 1640", 1640.42), ("O III] 1661", 1660.81),
                                      ("O III] 1666", 1666.15)]},
    {"key": "N_III", "title": "N III] multiplet", "center_A": 1750.00,
     "color": "#ef4444", "lines": [("N III] 1747", 1746.82), ("N III] 1749", 1748.65),
                                     ("N III] 1750", 1749.67), ("N III] 1752", 1752.16),
                                     ("N III] 1754", 1753.99)]},
    {"key": "C_III", "title": "C III] doublet", "center_A": 1900.00,
     "color": "#22c55e", "lines": [("C III] 1907", 1906.68), ("C III] 1909", 1908.73)]},
]


def set_style() -> None:
    plt.close("all")
    matplotlib.rcdefaults()
    matplotlib.rcParams.update({
        "text.usetex": False,
        "figure.facecolor": "#050812",
        "axes.facecolor": "#081321",
        "axes.edgecolor": "#94a3b8",
        "axes.labelcolor": "#eef6ff",
        "xtick.color": "#dce8f5",
        "ytick.color": "#dce8f5",
        "text.color": "#f8fbff",
        "font.size": 10,
        "savefig.facecolor": "#050812",
    })


def load_source() -> tuple[pd.DataFrame, Path]:
    candidates = [CSV / SOURCE_NAME, ROOT / "CSV" / SOURCE_NAME,
                  Path("/content") / SOURCE_NAME, SOURCE_PATH]
    source = next((p for p in candidates if p.exists() and p.stat().st_size > 5000), None)
    if source is None:
        urllib.request.urlretrieve(SOURCE_URL, SOURCE_PATH)
        source = SOURCE_PATH

    frame = pd.read_csv(source)
    required = ["observed_wavelength_um", "flux_native_display_units",
                "flux_error_native_display_units"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Missing required Stage-3 columns: {missing}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[np.isfinite(frame.observed_wavelength_um)
                  & np.isfinite(frame.flux_native_display_units)].copy()
    frame = frame.sort_values("observed_wavelength_um").drop_duplicates("observed_wavelength_um")
    error = frame.flux_error_native_display_units.to_numpy(float)
    good = np.isfinite(error) & (error > 0)
    replacement = float(np.nanmedian(error[good])) if good.any() else 1.0
    frame.loc[~good, "flux_error_native_display_units"] = replacement
    return frame.reset_index(drop=True), source


def obs_to_rest(observed_um: np.ndarray) -> np.ndarray:
    return np.asarray(observed_um, dtype=float) * 1.0e4 / (1.0 + Z)


def rest_to_obs(rest_A: np.ndarray) -> np.ndarray:
    return np.asarray(rest_A, dtype=float) * (1.0 + Z) * 1.0e-4


def prepare_region(frame: pd.DataFrame, region: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rest = obs_to_rest(frame.observed_wavelength_um.to_numpy(float))
    flux = frame.flux_native_display_units.to_numpy(float)
    error = frame.flux_error_native_display_units.to_numpy(float)
    low = region["center_A"] - 50.0
    high = region["center_A"] + 50.0
    mask = (rest >= low) & (rest <= high) & np.isfinite(flux) & np.isfinite(error)

    x_native = rest[mask]
    y_native = flux[mask]
    e_native = error[mask]
    order = np.argsort(x_native)
    x_native, y_native, e_native = x_native[order], y_native[order], e_native[order]
    x_native, unique_indices = np.unique(x_native, return_index=True)
    y_native, e_native = y_native[unique_indices], e_native[unique_indices]
    if x_native.size < 3:
        raise RuntimeError(f"{region['key']} has only {x_native.size} native samples.")

    # Exactly 300 display coordinates across the requested full +/-50 A window.
    x_display = np.linspace(low, high, DISPLAY_POINTS)
    flux_interp = PchipInterpolator(x_native, y_native, extrapolate=True)
    y_display = flux_interp(x_display)
    e_display = np.interp(x_display, x_native, e_native,
                          left=float(e_native[0]), right=float(e_native[-1]))

    native = pd.DataFrame({
        "region": region["key"],
        "sample_type": "native_measured",
        "rest_wavelength_A": x_native,
        "observed_wavelength_um": rest_to_obs(x_native),
        "flux_nJy": y_native,
        "flux_error_nJy": e_native,
        "independent_measurement": True,
    })
    display = pd.DataFrame({
        "region": region["key"],
        "sample_type": "display_interpolated",
        "rest_wavelength_A": x_display,
        "observed_wavelength_um": rest_to_obs(x_display),
        "flux_nJy": y_display,
        "flux_error_nJy": e_display,
        "independent_measurement": False,
    })
    if len(display) != DISPLAY_POINTS:
        raise RuntimeError(f"{region['key']} generated {len(display)} points, expected 300.")
    return native, display


def draw_panel(ax, region: dict, native: pd.DataFrame, display: pd.DataFrame,
               compact: bool = False) -> None:
    color = region["color"]
    xd = display.rest_wavelength_A.to_numpy(float)
    yd = display.flux_nJy.to_numpy(float)
    ed = display.flux_error_nJy.to_numpy(float)
    xn = native.rest_wavelength_A.to_numpy(float)
    yn = native.flux_nJy.to_numpy(float)
    en = native.flux_error_nJy.to_numpy(float)

    # Every one of the 300 display samples is visibly marked.
    ax.fill_between(xd, yd - ed, yd + ed, color=color, alpha=0.08, linewidth=0)
    ax.plot(xd, yd, color=color, linewidth=0.65, alpha=0.95)
    ax.scatter(xd, yd, s=10 if not compact else 6, marker="o", color=color,
               edgecolors="none", alpha=0.88,
               label=f"300 visible PCHIP display samples")

    # Native samples remain separately identifiable and are never hidden.
    ax.errorbar(xn, yn, yerr=en, fmt="o", markersize=7.0 if not compact else 5.0,
                markerfacecolor="#050812", markeredgecolor="#ffffff",
                markeredgewidth=1.2, ecolor="#dce8f5", capsize=2.3,
                linewidth=0.85, zorder=6,
                label=f"Native Stage-3 measured bins: {len(native)}")

    for label, wavelength in region["lines"]:
        ax.axvline(wavelength, color="#ffffff", linewidth=0.8, linestyle="--", alpha=0.82)
        if not compact:
            ax.annotate(label, xy=(wavelength, 0.97), xycoords=("data", "axes fraction"),
                        xytext=(3, 0), textcoords="offset points", rotation=90,
                        ha="left", va="top", fontsize=8.0, color="#ffffff")

    low, high = region["center_A"] - 50.0, region["center_A"] + 50.0
    ax.set_xlim(low, high)
    ax.axhline(0.0, color="#94a3b8", linewidth=0.5, alpha=0.55)
    ax.grid(True, linewidth=0.4, alpha=0.30, color="#41546c")
    ax.set_xlabel("Rest-frame vacuum wavelength [Å]")
    ax.set_ylabel("Flux density [nJy]")
    ax.set_title(
        f"{region['title']} | 300 visible markers | {len(native)} native bins",
        fontsize=10.5 if compact else 14.0,
        loc="left" if compact else "center",
    )


def individual_plot(region: dict, native: pd.DataFrame, display: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(18, 8.6), constrained_layout=True)
    draw_panel(ax, region, native, display, compact=False)
    top = ax.secondary_xaxis("top", functions=(rest_to_obs, obs_to_rest))
    top.set_xlabel("Observed wavelength [µm] at z = 14.44")
    ax.legend(loc="best", fontsize=8.7, framealpha=0.90)
    ax.text(
        0.012, 0.022,
        "Colored circles: all 300 plotted display samples. White outlined circles: native measured PRISM bins. "
        "The display samples add visual density, not new instrumental resolution.",
        transform=ax.transAxes, fontsize=8.4, va="bottom",
    )
    path = PNG / f"{VERSION}_{region['key']}_300_VISIBLE_MARKERS.png"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return path


def atlas(prepared: list[tuple[dict, pd.DataFrame, pd.DataFrame]]) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(21, 21), constrained_layout=True)
    for ax, (region, native, display) in zip(axes.ravel(), prepared):
        draw_panel(ax, region, native, display, compact=True)
    fig.suptitle(
        "MoM-z14 — exactly 300 visibly marked display samples in every spectral window\n"
        "Colored markers: 300-point PCHIP display grid | white outlined markers: native Stage-3 bins",
        fontsize=17.5,
    )
    path = PNG / f"{VERSION}_SIX_REGION_300_VISIBLE_MARKER_ATLAS.png"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return path


def main() -> None:
    set_style()
    frame, source = load_source()
    prepared = []
    native_tables = []
    display_tables = []
    summaries = []
    plot_paths = []

    for region in REGIONS:
        native, display = prepare_region(frame, region)
        prepared.append((region, native, display))
        native_tables.append(native)
        display_tables.append(display)
        plot_paths.append(individual_plot(region, native, display))
        summaries.append({
            "region": region["key"],
            "native_measured_bins": len(native),
            "visible_display_markers": len(display),
            "window_low_A": region["center_A"] - 50.0,
            "window_high_A": region["center_A"] + 50.0,
            "redshift": Z,
            "method": "PCHIP display resampling",
        })

    atlas_path = atlas(prepared)
    native_path = CSV / f"{VERSION}_NATIVE_MEASURED_BINS.csv"
    display_path = CSV / f"{VERSION}_300_VISIBLE_DISPLAY_SAMPLES.csv"
    summary_path = CSV / f"{VERSION}_POINT_COUNT_AUDIT.csv"
    pd.concat(native_tables, ignore_index=True).to_csv(native_path, index=False)
    pd.concat(display_tables, ignore_index=True).to_csv(display_path, index=False)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"TARGET                    {TARGET}")
    print(f"SOURCE                    {source}")
    print(f"REDSHIFT                  {Z:.5f}")
    print(f"VISIBLE DISPLAY MARKERS   {DISPLAY_POINTS} per region, exactly")
    print("NATIVE MEASURED BINS      shown separately as white outlined markers")
    print("METHOD                    PCHIP display resampling; no synthetic peaks")
    print(f"ATLAS PNG                 {atlas_path}")
    for path in plot_paths:
        print(f"REGION PNG                {path}")
    print(f"NATIVE CSV                {native_path}")
    print(f"300-POINT CSV             {display_path}")
    print(f"AUDIT CSV                 {summary_path}")
    print(f"Timestamp                 {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
