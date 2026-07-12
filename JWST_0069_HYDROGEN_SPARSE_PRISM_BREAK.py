# JWST_0069
# Hydrogen Ly-alpha: published laboratory VUV scan versus sparse real JWST/PRISM samples.
# Uses the strongest positive adjacent-sample jump near the expected Ly-alpha break.
# No smoothing model, no Gaussian profile, no AI images. Matplotlib only.

from pathlib import Path
from datetime import datetime, timezone
import importlib
import importlib.util
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VERSION = "JWST_0069"
REPO_RAW = "https://raw.githubusercontent.com/gear66me-ui/JWST/main"
LAB_FILE = "JWST_0067_HYDROGEN_LYA_LAB_SCAN_VS_JWST.py"
SELF_FETCH_FILE = "JWST_0068_HYDROGEN_SELF_FETCH.py"
LAB_PATH = Path("/content") / LAB_FILE if Path("/content").exists() else Path.cwd() / LAB_FILE
SELF_FETCH_PATH = Path("/content") / SELF_FETCH_FILE if Path("/content").exists() else Path.cwd() / SELF_FETCH_FILE

ROOT = Path("/content") if Path("/content").exists() else Path.cwd()
OUT = ROOT / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"
DATA = OUT / "DATA" / VERSION

WINDOW_LOW_NM = 1760.0
WINDOW_HIGH_NM = 1995.0
SEARCH_HALF_WIDTH_NM = 90.0
YELLOW = "#ffd84d"
RED = "#ff5a66"
ORANGE = "#ff9d2e"
POINT = "#d9edf7"


def need(import_name, pip_name=None):
    try:
        importlib.import_module(import_name)
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pip_name or import_name]
        )


def ensure_file(path, filename, minimum_size):
    if path.exists() and path.stat().st_size >= minimum_size:
        return path
    subprocess.run(
        [
            "curl", "-fsSL", "--connect-timeout", "15", "--max-time", "120",
            "-o", str(path), f"{REPO_RAW}/{filename}",
        ],
        check=True,
        timeout=130,
    )
    if not path.exists() or path.stat().st_size < minimum_size:
        raise RuntimeError(f"Could not download helper: {filename}")
    return path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def full_limits(values):
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    pad = 0.06 * (high - low) if high > low else max(abs(high) * 0.08, 1.0e-8)
    return low - pad, high + pad


def sparse_break(observed_nm, flux, expected_nm):
    mask = (
        np.isfinite(observed_nm)
        & np.isfinite(flux)
        & (observed_nm >= WINDOW_LOW_NM)
        & (observed_nm <= WINDOW_HIGH_NM)
    )
    x = observed_nm[mask]
    y = flux[mask]
    if len(x) < 6:
        raise RuntimeError(f"Only {len(x)} valid JWST samples lie in the plotting window")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    left_x = x[:-1]
    right_x = x[1:]
    dx = right_x - left_x
    dy = y[1:] - y[:-1]
    mid = 0.5 * (left_x + right_x)

    valid = (
        np.isfinite(dx)
        & np.isfinite(dy)
        & (dx > 0)
        & (mid >= expected_nm - SEARCH_HALF_WIDTH_NM)
        & (mid <= expected_nm + SEARCH_HALF_WIDTH_NM)
    )
    if not valid.any():
        raise RuntimeError("No adjacent JWST sample pair brackets the Ly-alpha search window")

    scale = float(np.nanmedian(np.abs(dy[valid] - np.nanmedian(dy[valid])))) * 1.4826
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(dy[valid]))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    positive = np.where(valid & (dy > 0))[0]
    candidates = positive if len(positive) else np.where(valid)[0]
    scores = dy[candidates] / scale - 0.12 * np.abs(mid[candidates] - expected_nm) / SEARCH_HALF_WIDTH_NM
    chosen = int(candidates[int(np.nanargmax(scores))])

    break_nm = float(mid[chosen])
    uncertainty_nm = 0.5 * float(dx[chosen])
    return {
        "x": x,
        "y": y,
        "left_nm": float(left_x[chosen]),
        "right_nm": float(right_x[chosen]),
        "left_flux": float(y[chosen]),
        "right_flux": float(y[chosen + 1]),
        "jump": float(dy[chosen]),
        "break_nm": break_nm,
        "uncertainty_nm": uncertainty_nm,
        "sample_count": int(len(x)),
    }


def style_axis(axis, lab):
    axis.set_facecolor(lab.AX_BG)
    axis.grid(True, color=lab.GRID, linewidth=0.50, alpha=0.48)
    axis.tick_params(colors=lab.TEXT, labelsize=8.5)
    axis.xaxis.label.set_color(lab.TEXT)
    axis.yaxis.label.set_color(lab.TEXT)
    axis.title.set_color(lab.TEXT)
    for spine in axis.spines.values():
        spine.set_color("#4fa8c8")
        spine.set_linewidth(0.85)


def make_plot(lab, lab_zoom, lab_crop_path, lab_url, source_path, source_status, observed_nm, flux):
    result = sparse_break(observed_nm, flux, lab.EXPECTED_OBS_NM)
    x = result["x"]
    y = result["y"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(16.0, 6.8), facecolor=lab.BG)
    style_axis(left, lab)
    style_axis(right, lab)

    left.imshow(
        np.asarray(lab_zoom),
        extent=[lab.LAB_ZOOM_LOW_NM, lab.LAB_ZOOM_HIGH_NM, 0.0, 1.0],
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )
    left.axvline(
        lab.LYA_REST_NM,
        color=YELLOW,
        linestyle=(0, (3, 5)),
        linewidth=0.60,
        alpha=0.68,
    )
    left.set_xlim(lab.LAB_ZOOM_LOW_NM, lab.LAB_ZOOM_HIGH_NM)
    left.set_ylim(0.0, 1.0)
    left.set_yticks([])
    left.set_title("PUBLISHED LAB HYDROGEN VUV SCAN — Lyα", fontsize=11.4, pad=13)
    left.set_xlabel("Laboratory wavelength, nm")
    left.set_ylabel("Published spectrometer signal (figure scale)")

    top_left = left.secondary_xaxis("top", functions=(lab.frequency_thz, lab.wavelength_nm))
    top_left.set_xlabel("Laboratory frequency, THz", color=lab.TEXT, labelpad=7)
    top_left.tick_params(colors=lab.TEXT, labelsize=8)

    right.plot(x, y, color=ORANGE, linewidth=0.70, alpha=0.76)
    right.scatter(
        x,
        y,
        s=30,
        color=POINT,
        edgecolor=lab.BG,
        linewidth=0.35,
        zorder=4,
    )
    right.axvline(
        result["break_nm"],
        color=RED,
        linestyle=(0, (3, 5)),
        linewidth=0.60,
        alpha=0.68,
        zorder=6,
    )
    right.scatter(
        [result["left_nm"], result["right_nm"]],
        [result["left_flux"], result["right_flux"]],
        s=42,
        facecolor=RED,
        edgecolor=lab.BG,
        linewidth=0.45,
        zorder=7,
    )
    right.set_xlim(WINDOW_LOW_NM, WINDOW_HIGH_NM)
    right.set_ylim(*full_limits(y))
    right.set_title("MoM-z14 JWST/PRISM — SAMPLE-BRACKETED Lyα BREAK", fontsize=11.4, pad=13)
    right.set_xlabel("Observed wavelength, nm")
    right.set_ylabel("JWST flux samples")

    top_right = right.secondary_xaxis("top", functions=(lab.frequency_thz, lab.wavelength_nm))
    top_right.set_xlabel("Observed frequency, THz", color=lab.TEXT, labelpad=7)
    top_right.tick_params(colors=lab.TEXT, labelsize=8)

    left.text(
        0.018,
        0.955,
        (
            "published hydrogen-discharge scan\n"
            f"rest marker = {lab.LYA_REST_NM:.6f} nm\n"
            f"rest frequency = {lab.frequency_thz(lab.LYA_REST_NM):.3f} THz"
        ),
        transform=left.transAxes,
        ha="left",
        va="top",
        color=lab.TEXT,
        fontsize=7.7,
        bbox=dict(
            boxstyle="round,pad=.30",
            facecolor="#07111f",
            edgecolor=YELLOW,
            linewidth=0.60,
            alpha=0.94,
        ),
    )
    right.text(
        0.018,
        0.955,
        (
            f"real PRISM samples = {result['sample_count']}\n"
            f"strongest positive jump = {result['left_nm']:.3f}–{result['right_nm']:.3f} nm\n"
            f"observed break = {result['break_nm']:.3f} ± {result['uncertainty_nm']:.3f} nm\n"
            f"z from break = {result['break_nm'] / lab.LYA_REST_NM - 1.0:.5f}"
        ),
        transform=right.transAxes,
        ha="left",
        va="top",
        color=lab.TEXT,
        fontsize=7.7,
        bbox=dict(
            boxstyle="round,pad=.30",
            facecolor="#07111f",
            edgecolor=RED,
            linewidth=0.60,
            alpha=0.94,
        ),
    )

    fig.suptitle(
        f"{VERSION} — HYDROGEN Lyα: PUBLISHED LAB SCAN versus REAL JWST/PRISM",
        color=lab.TEXT,
        fontsize=15.0,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.925,
        (
            "Left: measured laboratory VUV scan. Right: every available JWST sample in the window. "
            "The red marker is the midpoint of the strongest observed positive sample-to-sample jump; no smoothing fit is imposed."
        ),
        ha="center",
        color=lab.MUTED,
        fontsize=8.5,
    )
    fig.text(
        0.5,
        0.018,
        f"Lab source: {lab.LAB_ARTICLE}, Figure 2(a). JWST source: {source_path.name} ({source_status}).",
        ha="center",
        color=lab.MUTED,
        fontsize=7.8,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.12, wspace=0.16)

    plot_path = PNG / f"{VERSION}_{lab.GALAXY}_HYDROGEN_LYA_SPARSE_PRISM.png"
    fig.savefig(plot_path, dpi=245, facecolor=lab.BG, edgecolor=lab.BG)
    plt.show()
    plt.close(fig)

    sample_path = CSV / f"{VERSION}_{lab.GALAXY}_LYA_PRISM_SAMPLES.csv"
    pd.DataFrame(
        {
            "observed_wavelength_nm": x,
            "observed_frequency_THz": lab.frequency_thz(x),
            "jwst_flux": y,
            "selected_left_sample": np.isclose(x, result["left_nm"]),
            "selected_right_sample": np.isclose(x, result["right_nm"]),
            "break_midpoint_nm": result["break_nm"],
            "break_half_spacing_nm": result["uncertainty_nm"],
        }
    ).to_csv(sample_path, index=False)

    audit_path = CSV / f"{VERSION}_{lab.GALAXY}_HYDROGEN_BREAK_AUDIT.csv"
    pd.DataFrame([{
        "lab_figure_url": lab_url,
        "lab_crop_png": str(lab_crop_path),
        "rest_lya_nm": lab.LYA_REST_NM,
        "published_redshift_z": lab.Z,
        "expected_observed_nm": lab.EXPECTED_OBS_NM,
        "observed_break_midpoint_nm": result["break_nm"],
        "observed_break_half_spacing_nm": result["uncertainty_nm"],
        "left_bracketing_sample_nm": result["left_nm"],
        "right_bracketing_sample_nm": result["right_nm"],
        "sample_to_sample_flux_jump": result["jump"],
        "redshift_from_break_midpoint": result["break_nm"] / lab.LYA_REST_NM - 1.0,
        "jwst_sample_count": result["sample_count"],
        "jwst_source": str(source_path),
        "jwst_source_status": source_status,
    }]).to_csv(audit_path, index=False)

    return plot_path, sample_path, audit_path, result


def main():
    for import_name, pip_name in [
        ("numpy", None),
        ("pandas", None),
        ("matplotlib", None),
        ("requests", None),
        ("PIL", "Pillow"),
        ("astropy", None),
        ("astroquery", None),
    ]:
        need(import_name, pip_name)

    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    ensure_file(LAB_PATH, LAB_FILE, 15000)
    ensure_file(SELF_FETCH_PATH, SELF_FETCH_FILE, 5000)
    lab = load_module("jwst_0067_lab", LAB_PATH)
    fetcher = load_module("jwst_0068_fetcher", SELF_FETCH_PATH)

    lab.VERSION = VERSION
    lab.OUT = OUT
    lab.PNG = PNG
    lab.CSV = CSV
    lab.DATA = DATA / "LAB"

    fetcher.VERSION = VERSION
    fetcher.OUT = OUT
    fetcher.PNG = PNG
    fetcher.CSV = CSV
    fetcher.DATA = DATA
    fetcher.configure_lab_module(lab)

    ensure_file(fetcher.MAST_PATH, fetcher.MAST_MODULE_NAME, 12000)
    mast = load_module("jwst_0060_mast", fetcher.MAST_PATH)
    fetcher.configure_mast_module(mast)

    print(f"CODE OUTPUT: {VERSION}")
    print("STEP 1/4 | Locate or retrieve the coordinate-verified MoM-z14 spectrum")
    source_path, source_status, mast_metadata = fetcher.acquire_jwst_csv(lab, mast)
    observed_nm, flux, wave_column, flux_column = lab.load_jwst(source_path)

    print("STEP 2/4 | Download the published hydrogen laboratory VUV scan")
    lab_image, lab_raw_path, lab_url = lab.download_lab_figure()

    print("STEP 3/4 | Crop the published laboratory Ly-alpha scan")
    lab_zoom, lab_crop_path, plot_box = lab.crop_lab_scan(lab_image)

    print("STEP 4/4 | Plot all sparse JWST samples and bracket the observed break")
    plot_path, sample_path, audit_path, result = make_plot(
        lab,
        lab_zoom,
        lab_crop_path,
        lab_url,
        source_path,
        source_status,
        observed_nm,
        flux,
    )

    print()
    print(f"JWST SOURCE          : {source_path}")
    print(f"JWST SOURCE STATUS   : {source_status}")
    print(f"WAVELENGTH COLUMN    : {wave_column}")
    print(f"FLUX COLUMN          : {flux_column}")
    print(f"JWST SAMPLE COUNT    : {result['sample_count']}")
    print(f"BREAK BRACKET        : {result['left_nm']:.6f} to {result['right_nm']:.6f} nm")
    print(f"BREAK MIDPOINT       : {result['break_nm']:.6f} nm")
    print(f"HALF-SPACING         : {result['uncertainty_nm']:.6f} nm")
    print(f"BREAK REDSHIFT       : {result['break_nm'] / lab.LYA_REST_NM - 1.0:.6f}")
    print(f"LAB RAW FIGURE       : {lab_raw_path}")
    print(f"LAB CROP PNG         : {lab_crop_path}")
    print(f"LAB PLOT BOX PX      : {plot_box}")
    print(f"PLOT PNG             : {plot_path}")
    print(f"JWST SAMPLE CSV      : {sample_path}")
    print(f"AUDIT CSV            : {audit_path}")
    if mast_metadata is not None:
        print(f"JWST PRODUCT         : {Path(mast_metadata['product']).name}")
        print(f"JWST SEPARATION      : {mast_metadata['sep']:.6f} arcsec")
        print(f"JWST COORDINATE KEYS : {mast_metadata['coord_source']}")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
