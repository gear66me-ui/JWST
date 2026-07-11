# JWST_0064
# Fourteen narrow transition plots with one peak-coincident marker per panel.
# LEFT marker: selected rest-frame peak, thin yellow dashed line.
# RIGHT marker: corresponding observed peak, thin red dashed line.
# No shifted-reference marker. Full local flux range. No AI images.

from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import re
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VERSION = "JWST_0064"
BASE_NAME = "JWST_0062_MOMZ14_FEATURE_ZOOM_PAIRS.py"
BASE_URL = f"https://raw.githubusercontent.com/gear66me-ui/JWST/main/{BASE_NAME}"
BASE_PATH = Path("/content") / BASE_NAME if Path("/content").exists() else Path.cwd() / BASE_NAME

OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"

STRETCH = 15.44
REST_HALF_WIDTH_NM = 1.0
PEAK_SEARCH_HALF_WIDTH_NM = 0.45
YELLOW = "#ffd84d"
RED = "#ff5a66"


def ensure_base():
    if BASE_PATH.exists() and BASE_PATH.stat().st_size > 5000:
        return BASE_PATH
    subprocess.run(
        [
            "curl", "-fsSL", "--connect-timeout", "15", "--max-time", "90",
            "-o", str(BASE_PATH), BASE_URL,
        ],
        check=True,
        timeout=100,
    )
    if not BASE_PATH.exists() or BASE_PATH.stat().st_size < 5000:
        raise RuntimeError("Could not download JWST_0062 helper script.")
    return BASE_PATH


def load_base(path):
    spec = importlib.util.spec_from_file_location("jwst_0062_base", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load JWST_0062 helper module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def full_limits(values):
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    padding = 0.045 * (high - low) if high > low else max(abs(high) * 0.06, 1.0e-8)
    return low - padding, high + padding


def choose_peak(rest_wave, observed_wave, flux, nominal_rest_nm):
    search = (
        np.isfinite(rest_wave)
        & np.isfinite(observed_wave)
        & np.isfinite(flux)
        & (rest_wave >= nominal_rest_nm - PEAK_SEARCH_HALF_WIDTH_NM)
        & (rest_wave <= nominal_rest_nm + PEAK_SEARCH_HALF_WIDTH_NM)
    )
    if int(search.sum()) < 1:
        search = np.isfinite(rest_wave) & np.isfinite(observed_wave) & np.isfinite(flux)
    indices = np.flatnonzero(search)
    chosen = int(indices[np.nanargmax(flux[search])])
    return float(rest_wave[chosen]), float(observed_wave[chosen]), float(flux[chosen])


def plot_transition(base, number, label, nominal_rest_nm, observed_nm, flux, source):
    rest_all = observed_nm / STRETCH
    low = nominal_rest_nm - REST_HALF_WIDTH_NM
    high = nominal_rest_nm + REST_HALF_WIDTH_NM
    mask = (
        np.isfinite(rest_all)
        & np.isfinite(observed_nm)
        & np.isfinite(flux)
        & (rest_all >= low)
        & (rest_all <= high)
    )
    if int(mask.sum()) < 5:
        raise RuntimeError(f"{label}: only {int(mask.sum())} samples in ±1 nm")

    rest_wave = rest_all[mask]
    obs_wave = observed_nm[mask]
    local_flux = flux[mask]
    order = np.argsort(rest_wave)
    rest_wave = rest_wave[order]
    obs_wave = obs_wave[order]
    local_flux = local_flux[order]

    peak_rest_nm, peak_obs_nm, peak_flux = choose_peak(
        rest_wave,
        obs_wave,
        local_flux,
        nominal_rest_nm,
    )

    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(15.5, 6.4),
        sharey=True,
        facecolor=base.BG,
    )
    base.style(left)
    base.style(right)

    left.plot(rest_wave, local_flux, color=base.CYAN, linewidth=0.78, alpha=0.88)
    left.scatter(
        rest_wave, local_flux, s=18, color=base.POINT,
        edgecolor=base.BG, linewidth=0.28, zorder=4,
    )
    right.plot(obs_wave, local_flux, color=base.ORANGE, linewidth=0.78, alpha=0.88)
    right.scatter(
        obs_wave, local_flux, s=18, color=base.POINT,
        edgecolor=base.BG, linewidth=0.28, zorder=4,
    )

    left.axvline(
        peak_rest_nm,
        color=YELLOW,
        linestyle=(0, (3, 4)),
        linewidth=0.65,
        alpha=0.82,
        zorder=5,
    )
    right.axvline(
        peak_obs_nm,
        color=RED,
        linestyle=(0, (3, 4)),
        linewidth=0.65,
        alpha=0.82,
        zorder=5,
    )

    left.scatter(
        [peak_rest_nm], [peak_flux], s=34, facecolor=YELLOW,
        edgecolor=base.BG, linewidth=0.45, zorder=7,
    )
    right.scatter(
        [peak_obs_nm], [peak_flux], s=34, facecolor=RED,
        edgecolor=base.BG, linewidth=0.45, zorder=7,
    )

    left.set_xlim(low, high)
    right.set_xlim(low * STRETCH, high * STRETCH)
    left.set_ylim(*full_limits(local_flux))

    left.set_title(f"REST-FRAME PEAK — {label}", fontsize=11.8, pad=13)
    right.set_title(f"OBSERVED PEAK — {label}", fontsize=11.8, pad=13)
    left.set_xlabel("Rest wavelength, nm")
    right.set_xlabel("Observed wavelength, nm")
    left.set_ylabel("JWST flux samples")

    top_left = left.secondary_xaxis("top", functions=(base.nu, base.lam))
    top_left.set_xlabel("Rest frequency, THz", color=base.TEXT, labelpad=7)
    top_left.tick_params(colors=base.TEXT, labelsize=8)
    top_right = right.secondary_xaxis("top", functions=(base.nu, base.lam))
    top_right.set_xlabel("Observed frequency, THz", color=base.TEXT, labelpad=7)
    top_right.tick_params(colors=base.TEXT, labelsize=8)

    left.text(
        0.018,
        0.045,
        (
            f"selected peak\n"
            f"λrest = {peak_rest_nm:.6f} nm\n"
            f"νrest = {base.nu(peak_rest_nm):.6f} THz"
        ),
        transform=left.transAxes,
        ha="left",
        va="bottom",
        color=base.TEXT,
        fontsize=7.8,
        bbox=dict(
            boxstyle="round,pad=.28",
            facecolor="#07111f",
            edgecolor=YELLOW,
            linewidth=0.65,
            alpha=0.94,
        ),
    )
    right.text(
        0.018,
        0.045,
        (
            f"same measured sample\n"
            f"λobserved = {peak_obs_nm:.6f} nm\n"
            f"νobserved = {base.nu(peak_obs_nm):.6f} THz"
        ),
        transform=right.transAxes,
        ha="left",
        va="bottom",
        color=base.TEXT,
        fontsize=7.8,
        bbox=dict(
            boxstyle="round,pad=.28",
            facecolor="#07111f",
            edgecolor=RED,
            linewidth=0.65,
            alpha=0.94,
        ),
    )

    fig.suptitle(
        f"{VERSION} — {base.GALAXY} — transition {number:02d}/14",
        color=base.TEXT,
        fontsize=14.8,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.925,
        "Thin yellow and red dashed markers pass through the same selected flux sample in the rest and observed frames.",
        ha="center",
        color=base.MUTED,
        fontsize=8.7,
    )
    fig.text(
        0.5,
        0.018,
        f"Source: {source.name}. Window: ±{REST_HALF_WIDTH_NM:.1f} nm rest frame. Full local flux height shown.",
        ha="center",
        color=base.MUTED,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.12, wspace=0.12)

    stem = f"{VERSION}_{number:02d}_{safe_name(label)}"
    png_path = PNG / f"{stem}_THIN_PEAK_MARKERS.png"
    csv_path = CSV / f"{stem}_SAMPLES.csv"
    fig.savefig(png_path, dpi=245, facecolor=base.BG, edgecolor=base.BG)
    plt.show()
    plt.close(fig)

    pd.DataFrame(
        {
            "transition_number": number,
            "transition_label": label,
            "nominal_rest_wavelength_nm": nominal_rest_nm,
            "rest_wavelength_nm": rest_wave,
            "rest_frequency_THz": base.nu(rest_wave),
            "observed_wavelength_nm": obs_wave,
            "observed_frequency_THz": base.nu(obs_wave),
            "jwst_flux": local_flux,
            "selected_peak_rest_nm": peak_rest_nm,
            "selected_peak_observed_nm": peak_obs_nm,
            "selected_peak_flux": peak_flux,
            "stretch_factor": STRETCH,
            "redshift_z": base.Z,
        }
    ).to_csv(csv_path, index=False)

    return {
        "number": number,
        "label": label,
        "nominal_rest_nm": nominal_rest_nm,
        "peak_rest_nm": peak_rest_nm,
        "peak_observed_nm": peak_obs_nm,
        "peak_rest_THz": float(base.nu(peak_rest_nm)),
        "peak_observed_THz": float(base.nu(peak_obs_nm)),
        "peak_flux": peak_flux,
        "samples": int(mask.sum()),
        "flux_min": float(np.nanmin(local_flux)),
        "flux_max": float(np.nanmax(local_flux)),
        "png": str(png_path),
        "csv": str(csv_path),
    }


def main():
    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)

    base = load_base(ensure_base())
    source = base.locate_spectrum()
    observed_nm, flux, wavelength_column, flux_column = base.load_data(source)

    print(f"CODE OUTPUT: {VERSION}")
    print("MODE       : ±1 nm windows; one peak-coincident marker per panel")
    print(f"SOURCE CSV : {source}")
    print()

    rows = []
    failures = []
    for number, label, nominal_rest_nm in base.LINES:
        try:
            print(f"PLOT {number:02d}/14 | {label}")
            rows.append(
                plot_transition(
                    base,
                    number,
                    label,
                    nominal_rest_nm,
                    observed_nm,
                    flux,
                    source,
                )
            )
        except Exception as exc:
            print(f"FAILED {number:02d}/14 | {label} | {type(exc).__name__}: {exc}")
            failures.append((number, label, type(exc).__name__, str(exc)))

    if not rows:
        raise RuntimeError("No narrow-band transition plots were generated")

    summary_path = CSV / f"{VERSION}_{base.GALAXY}_PEAK_INDEX.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    if failures:
        pd.DataFrame(
            failures,
            columns=["number", "label", "error_type", "message"],
        ).to_csv(CSV / f"{VERSION}_{base.GALAXY}_FAILURES.csv", index=False)

    print()
    for row in rows:
        print(
            f"{row['number']:02d} | {row['label']:<15} | "
            f"rest peak={row['peak_rest_nm']:10.6f} nm | "
            f"observed peak={row['peak_observed_nm']:11.6f} nm | "
            f"samples={row['samples']:4d}"
        )
    print()
    print(f"WAVELENGTH COLUMN : {wavelength_column}")
    print(f"FLUX COLUMN       : {flux_column}")
    print(f"REST HALF-WIDTH   : {REST_HALF_WIDTH_NM:.3f} nm")
    print(f"PEAK SEARCH WIDTH : {PEAK_SEARCH_HALF_WIDTH_NM:.3f} nm")
    print(f"GENERATED         : {len(rows)} PNG + {len(rows)} CSV")
    print(f"SUMMARY CSV       : {summary_path}")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
