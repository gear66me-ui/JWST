#!/usr/bin/env python3
"""
JWST_0106_MOMZ14_EXACT_300_POINT_WINDOWS.py

Create exactly 300 plotted display samples in each of six MoM-z14 rest-frame
spectral windows. The measured source remains the exact Stage-3 PRISM/CLEAR
X1D spectrum. Native detector samples are shown explicitly beneath a
shape-preserving PCHIP display curve. No synthetic peaks, Gaussian/Voigt line
models, random noise, or AI-generated images are used.
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

VERSION = "JWST_0106"
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
    {
        "key": "LYMAN_ALPHA_BREAK",
        "title": "H I Lyman-alpha break",
        "center_A": 1215.67,
        "color": "#ff9f1c",
        "lines": [("H I Ly-alpha", 1215.67)],
    },
    {
        "key": "N_IV",
        "title": "N IV] doublet",
        "center_A": 1486.50,
        "color": "#ff4fa3",
        "lines": [("N IV] 1483", 1483.32), ("N IV] 1487", 1486.50)],
    },
    {
        "key": "C_IV",
        "title": "C IV doublet",
        "center_A": 1550.00,
        "color": "#25c7f7",
        "lines": [("C IV 1548", 1548.20), ("C IV 1551", 1550.77)],
    },
    {
        "key": "HEII_OIII",
        "title": "He II + O III] complex",
        "center_A": 1650.00,
        "color": "#b978ff",
        "lines": [("He II 1640", 1640.42), ("O III] 1661", 1660.81), ("O III] 1666", 1666.15)],
    },
    {
        "key": "N_III",
        "title": "N III] multiplet",
        "center_A": 1750.00,
        "color": "#ef4444",
        "lines": [
            ("N III] 1747", 1746.82),
            ("N III] 1749", 1748.65),
            ("N III] 1750", 1749.67),
            ("N III] 1752", 1752.16),
            ("N III] 1754", 1753.99),
        ],
    },
    {
        "key": "C_III",
        "title": "C III] doublet",
        "center_A": 1900.00,
        "color": "#22c55e",
        "lines": [("C III] 1907", 1906.68), ("C III] 1909", 1908.73)],
    },
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
    local_candidates = [
        CSV / SOURCE_NAME,
        ROOT / "CSV" / SOURCE_NAME,
        Path("/content") / SOURCE_NAME,
        SOURCE_PATH,
    ]
    source = next((p for p in local_candidates if p.exists() and p.stat().st_size > 5000), None)
    if source is None:
        urllib.request.urlretrieve(SOURCE_URL, SOURCE_PATH)
        source = SOURCE_PATH

    frame = pd.read_csv(source)
    required = {
        "observed_wavelength_um",
        "flux_native_display_units",
        "flux_error_native_display_units",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Missing required Stage-3 columns: {sorted(missing)}")

    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        np.isfinite(frame["observed_wavelength_um"])
        & np.isfinite(frame["flux_native_display_units"])
    ].copy()
    frame = frame.sort_values("observed_wavelength_um").drop_duplicates("observed_wavelength_um")

    err = frame["flux_error_native_display_units"].to_numpy(float)
    good = np.isfinite(err) & (err > 0)
    replacement = float(np.nanmedian(err[good])) if good.any() else 1.0
    frame.loc[~good, "flux_error_native_display_units"] = replacement
    return frame.reset_index(drop=True), source


def observed_to_rest_A(observed_um: np.ndarray) -> np.ndarray:
    return np.asarray(observed_um, dtype=float) * 1.0e4 / (1.0 + Z)


def rest_to_observed_um(rest_A: np.ndarray) -> np.ndarray:
    return np.asarray(rest_A, dtype=float) * (1.0 + Z) * 1.0e-4


def prepare_region(frame: pd.DataFrame, region: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rest = observed_to_rest_A(frame["observed_wavelength_um"].to_numpy(float))
    flux = frame["flux_native_display_units"].to_numpy(float)
    error = frame["flux_error_native_display_units"].to_numpy(float)

    low = region["center_A"] - 50.0
    high = region["center_A"] + 50.0
    mask = (rest >= low) & (rest <= high) & np.isfinite(flux) & np.isfinite(error)
    x_native = rest[mask]
    y_native = flux[mask]
    e_native = error[mask]

    order = np.argsort(x_native)
    x_native = x_native[order]
    y_native = y_native[order]
    e_native = e_native[order]

    unique_x, unique_index = np.unique(x_native, return_index=True)
    x_native = unique_x
    y_native = y_native[unique_index]
    e_native = e_native[unique_index]
    if len(x_native) < 3:
        raise RuntimeError(f"{region['key']} has only {len(x_native)} native samples; at least 3 are required.")

    x_display = np.linspace(float(x_native.min()), float(x_native.max()), DISPLAY_POINTS)
    flux_curve = PchipInterpolator(x_native, y_native, extrapolate=False)
    y_display = flux_curve(x_display)
    e_display = np.interp(x_display, x_native, e_native)

    native = pd.DataFrame({
        "region": region["key"],
        "sample_type": "native_measured",
        "rest_wavelength_A": x_native,
        "observed_wavelength_um": rest_to_observed_um(x_native),
        "flux_nJy": y_native,
        "flux_error_nJy": e_native,
        "independent_measurement": True,
        "display_interpolated": False,
    })
    display = pd.DataFrame({
        "region": region["key"],
        "sample_type": "display_interpolated",
        "rest_wavelength_A": x_display,
        "observed_wavelength_um": rest_to_observed_um(x_display),
        "flux_nJy": y_display,
        "flux_error_nJy": e_display,
        "independent_measurement": False,
        "display_interpolated": True,
    })
    if len(display) != DISPLAY_POINTS:
        raise RuntimeError(f"{region['key']} produced {len(display)} display samples instead of {DISPLAY_POINTS}.")
    return native, display


def decorate_axis(ax, region: dict, native: pd.DataFrame, display: pd.DataFrame, compact: bool) -> None:
    color = region["color"]
    x = display["rest_wavelength_A"].to_numpy(float)
    y = display["flux_nJy"].to_numpy(float)
    e = display["flux_error_nJy"].to_numpy(float)
    xn = native["rest_wavelength_A"].to_numpy(float)
    yn = native["flux_nJy"].to_numpy(float)
    en = native["flux_error_nJy"].to_numpy(float)

    ax.fill_between(x, y - e, y + e, color=color, alpha=0.10, linewidth=0,
                    label="Interpolated 1-sigma display envelope")
    ax.plot(x, y, color=color, linewidth=1.45,
            label=f"PCHIP display curve ({DISPLAY_POINTS} plotted samples)")
    ax.errorbar(xn, yn, yerr=en, fmt="o", markersize=4.6 if not compact else 3.2,
                color="#ffffff", ecolor="#94a3b8", capsize=2.0,
                linewidth=0.8, label=f"Native Stage-3 measurements (n={len(native)})")

    for label, wavelength in region["lines"]:
        ax.axvline(wavelength, color=color, linewidth=1.1, linestyle="--", alpha=0.95)
        if not compact:
            ax.annotate(label, xy=(wavelength, 0.97), xycoords=("data", "axes fraction"),
                        xytext=(3, 0), textcoords="offset points", rotation=90,
                        ha="left", va="top", fontsize=8.0, color=color)

    low = region["center_A"] - 50.0
    high = region["center_A"] + 50.0
    ax.set_xlim(low, high)
    ax.axhline(0.0, color="#94a3b8", linewidth=0.5, alpha=0.55)
    ax.grid(True, linewidth=0.42, alpha=0.34, color="#41546c")
    ax.set_title(
        f"{region['title']} | 300 display samples from {len(native)} native measurements",
        fontsize=10.7 if compact else 14.0,
        loc="left" if compact else "center",
    )
    ax.set_xlabel("Rest-frame vacuum wavelength [Å]")
    ax.set_ylabel("Flux density [nJy]")


def make_individual_plot(region: dict, native: pd.DataFrame, display: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(17.5, 8.5), constrained_layout=True)
    decorate_axis(ax, region, native, display, compact=False)
    top = ax.secondary_xaxis("top", functions=(rest_to_observed_um, observed_to_rest_A))
    top.set_xlabel("Observed wavelength [µm] at z = 14.44")
    ax.legend(loc="best", fontsize=8.6, framealpha=0.88)
    ax.text(
        0.012, 0.022,
        "Exactly 300 plotted vertices. Native white points are the measured PRISM bins; "
        "the colored curve is shape-preserving display interpolation only.",
        transform=ax.transAxes, fontsize=8.4, va="bottom",
    )
    output = PNG / f"{VERSION}_{region['key']}_EXACT_300_POINTS.png"
    fig.savefig(output, dpi=420, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return output


def make_atlas(prepared: list[tuple[dict, pd.DataFrame, pd.DataFrame]]) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(20, 21), constrained_layout=True)
    for ax, (region, native, display) in zip(axes.ravel(), prepared):
        decorate_axis(ax, region, native, display, compact=True)
    fig.suptitle(
        "MoM-z14 — six spectral windows with exactly 300 plotted samples each\n"
        "Colored curves: PCHIP display interpolation | white points: native Stage-3 PRISM measurements",
        fontsize=17.5,
    )
    output = PNG / f"{VERSION}_SIX_REGION_EXACT_300_POINT_ATLAS.png"
    fig.savefig(output, dpi=360, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return output


def main() -> None:
    set_style()
    frame, source = load_source()
    prepared = []
    native_frames = []
    display_frames = []
    summary_rows = []
    individual_paths = []

    for region in REGIONS:
        native, display = prepare_region(frame, region)
        prepared.append((region, native, display))
        native_frames.append(native)
        display_frames.append(display)
        individual_paths.append(make_individual_plot(region, native, display))
        summary_rows.append({
            "region": region["key"],
            "center_A": region["center_A"],
            "window_low_A": region["center_A"] - 50.0,
            "window_high_A": region["center_A"] + 50.0,
            "native_measured_points": len(native),
            "display_interpolated_points": len(display),
            "interpolation_method": "PCHIP shape-preserving",
            "redshift_used": Z,
        })

    atlas_path = make_atlas(prepared)
    native_path = CSV / f"{VERSION}_NATIVE_MEASURED_SAMPLES.csv"
    display_path = CSV / f"{VERSION}_EXACT_300_POINT_DISPLAY_SAMPLES.csv"
    summary_path = CSV / f"{VERSION}_REGION_POINT_COUNTS.csv"
    pd.concat(native_frames, ignore_index=True).to_csv(native_path, index=False)
    pd.concat(display_frames, ignore_index=True).to_csv(display_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"TARGET               {TARGET}")
    print(f"SOURCE               {source}")
    print(f"REDSHIFT             {Z:.5f}")
    print(f"DISPLAY POINTS       {DISPLAY_POINTS} per region, exactly")
    print("METHOD               PCHIP shape-preserving display interpolation")
    print("NATIVE DATA          preserved and plotted as white measured points")
    print(f"ATLAS PNG            {atlas_path}")
    for path in individual_paths:
        print(f"REGION PNG           {path}")
    print(f"NATIVE CSV           {native_path}")
    print(f"300-POINT CSV        {display_path}")
    print(f"SUMMARY CSV          {summary_path}")
    print(f"Timestamp            {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
