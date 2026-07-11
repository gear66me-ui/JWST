# JWST_0063
# Narrow-band rest/observed transition plots with explicit reference lines.
# LEFT: dashed laboratory rest reference.
# RIGHT: dashed expected observed reference + solid measured local maximum.
# Full local flux range; no percentile clipping. No AI images. Matplotlib only.

from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import re
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VERSION = "JWST_0063"
BASE_NAME = "JWST_0062_MOMZ14_FEATURE_ZOOM_PAIRS.py"
BASE_URL = f"https://raw.githubusercontent.com/gear66me-ui/JWST/main/{BASE_NAME}"
BASE_PATH = Path("/content") / BASE_NAME if Path("/content").exists() else Path.cwd() / BASE_NAME
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"

STRETCH = 15.44
REST_HALF_WIDTH_NM = 2.5
PEAK_SEARCH_HALF_WIDTH_NM = 0.90
MEASURED = "#ff4d5a"


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


def full_flux_limits(values):
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    padding = 0.055 * (maximum - minimum) if maximum > minimum else max(abs(maximum) * 0.08, 1e-8)
    return minimum - padding, maximum + padding


def measured_peak(rest_wave, flux, center_nm):
    mask = (
        np.isfinite(rest_wave)
        & np.isfinite(flux)
        & (rest_wave >= center_nm - PEAK_SEARCH_HALF_WIDTH_NM)
        & (rest_wave <= center_nm + PEAK_SEARCH_HALF_WIDTH_NM)
    )
    if int(mask.sum()) == 0:
        return np.nan, np.nan
    local_wave = rest_wave[mask]
    local_flux = flux[mask]
    index = int(np.nanargmax(local_flux))
    return float(local_wave[index]), float(local_flux[index])


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def plot_transition(base, number, label, rest_reference_nm, observed_nm, flux, source):
    rest_all = observed_nm / STRETCH
    low = rest_reference_nm - REST_HALF_WIDTH_NM
    high = rest_reference_nm + REST_HALF_WIDTH_NM
    mask = (
        np.isfinite(rest_all)
        & np.isfinite(observed_nm)
        & np.isfinite(flux)
        & (rest_all >= low)
        & (rest_all <= high)
    )
    if int(mask.sum()) < 5:
        raise RuntimeError(f"{label}: only {int(mask.sum())} samples in the narrow band")

    rest_wave = rest_all[mask]
    obs_wave = observed_nm[mask]
    local_flux = flux[mask]
    order = np.argsort(rest_wave)
    rest_wave = rest_wave[order]
    obs_wave = obs_wave[order]
    local_flux = local_flux[order]

    measured_rest_nm, measured_flux = measured_peak(rest_wave, local_flux, rest_reference_nm)
    measured_obs_nm = measured_rest_nm * STRETCH
    expected_obs_nm = rest_reference_nm * STRETCH

    fig, (left, right) = plt.subplots(1, 2, figsize=(15.5, 6.5), sharey=True, facecolor=base.BG)
    base.style(left)
    base.style(right)

    left.plot(rest_wave, local_flux, color=base.CYAN, lw=0.95)
    left.scatter(rest_wave, local_flux, s=24, color=base.POINT, edgecolor=base.BG, lw=0.35, zorder=4)
    right.plot(obs_wave, local_flux, color=base.ORANGE, lw=0.95)
    right.scatter(obs_wave, local_flux, s=24, color=base.POINT, edgecolor=base.BG, lw=0.35, zorder=4)

    left.axvline(
        rest_reference_nm,
        color=base.CYAN,
        ls="--",
        lw=2.2,
        label="laboratory rest reference",
        zorder=5,
    )
    right.axvline(
        expected_obs_nm,
        color=base.ORANGE,
        ls="--",
        lw=2.2,
        label="reference shifted by 1+z",
        zorder=5,
    )
    if np.isfinite(measured_obs_nm):
        right.axvline(
            measured_obs_nm,
            color=MEASURED,
            ls="-",
            lw=1.8,
            label="measured local maximum",
            zorder=6,
        )

    left.set_xlim(low, high)
    right.set_xlim(low * STRETCH, high * STRETCH)
    left.set_ylim(*full_flux_limits(local_flux))

    left.set_title(f"REST REFERENCE — {label}", fontsize=12, pad=10)
    right.set_title(f"OBSERVED JWST BAND — {label}", fontsize=12, pad=10)
    left.set_xlabel("Rest wavelength, nm")
    right.set_xlabel("Observed wavelength, nm")
    left.set_ylabel("JWST flux samples")

    top_left = left.secondary_xaxis("top", functions=(base.nu, base.lam))
    top_left.set_xlabel("Rest frequency, THz", color=base.TEXT)
    top_left.tick_params(colors=base.TEXT, labelsize=8)
    top_right = right.secondary_xaxis("top", functions=(base.nu, base.lam))
    top_right.set_xlabel("Observed frequency, THz", color=base.TEXT)
    top_right.tick_params(colors=base.TEXT, labelsize=8)

    left.text(
        0.02, 0.05,
        f"LAB REFERENCE\nλrest = {rest_reference_nm:.3f} nm\nνrest = {base.nu(rest_reference_nm):.3f} THz",
        transform=left.transAxes,
        color=base.TEXT,
        fontsize=8.2,
        va="bottom",
        bbox=dict(boxstyle="round,pad=.3", fc="#07111f", ec=base.CYAN, alpha=.95),
    )
    peak_text = "not resolved"
    if np.isfinite(measured_obs_nm):
        peak_text = f"{measured_obs_nm:.3f} nm / {base.nu(measured_obs_nm):.3f} THz"
    right.text(
        0.02, 0.05,
        (
            f"SHIFTED REFERENCE\nλexpected = {expected_obs_nm:.3f} nm\n"
            f"νexpected = {base.nu(expected_obs_nm):.3f} THz\nlocal maximum = {peak_text}"
        ),
        transform=right.transAxes,
        color=base.TEXT,
        fontsize=8.2,
        va="bottom",
        bbox=dict(boxstyle="round,pad=.3", fc="#07111f", ec=base.ORANGE, alpha=.95),
    )

    for axis in (left, right):
        legend = axis.legend(loc="upper right", fontsize=7.0, facecolor="#07111f", edgecolor=base.GRID, framealpha=.96)
        for text in legend.get_texts():
            text.set_color(base.TEXT)

    fig.suptitle(
        f"{VERSION} — {base.GALAXY} — transition {number:02d}/14",
        color=base.TEXT,
        fontsize=15,
        fontweight="bold",
        y=.977,
    )
    fig.text(
        .5, .918,
        "Left dashed line = laboratory rest reference. Right dashed line = expected observed position. Solid red line = measured local maximum.",
        ha="center",
        color=base.MUTED,
        fontsize=8.9,
    )
    fig.text(
        .5, .018,
        f"Source: {source.name}. Full local minimum and maximum shown; no percentile clipping.",
        ha="center",
        color=base.MUTED,
        fontsize=8,
    )
    fig.subplots_adjust(left=.07, right=.985, top=.85, bottom=.12, wspace=.12)

    stem = f"{VERSION}_{number:02d}_{safe_name(label)}"
    png_path = PNG / f"{stem}_REFERENCE_LINES.png"
    csv_path = CSV / f"{stem}_SAMPLES.csv"
    fig.savefig(png_path, dpi=245, facecolor=base.BG)
    plt.show()
    plt.close(fig)

    pd.DataFrame(
        {
            "transition_number": number,
            "transition_label": label,
            "rest_wavelength_nm": rest_wave,
            "rest_frequency_THz": base.nu(rest_wave),
            "observed_wavelength_nm": obs_wave,
            "observed_frequency_THz": base.nu(obs_wave),
            "jwst_flux": local_flux,
            "laboratory_rest_reference_nm": rest_reference_nm,
            "expected_observed_reference_nm": expected_obs_nm,
            "measured_local_peak_observed_nm": measured_obs_nm,
            "measured_local_peak_flux": measured_flux,
            "stretch_factor": STRETCH,
            "redshift_z": base.Z,
        }
    ).to_csv(csv_path, index=False)

    return {
        "number": number,
        "label": label,
        "rest_reference_nm": rest_reference_nm,
        "expected_observed_nm": expected_obs_nm,
        "measured_observed_nm": measured_obs_nm,
        "rest_reference_THz": float(base.nu(rest_reference_nm)),
        "expected_observed_THz": float(base.nu(expected_obs_nm)),
        "measured_observed_THz": float(base.nu(measured_obs_nm)) if np.isfinite(measured_obs_nm) else np.nan,
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
    print("MODE       : 14 narrow-band plots with explicit reference lines")
    print(f"SOURCE CSV : {source}")
    print()

    results = []
    failures = []
    for number, label, rest_reference_nm in base.LINES:
        try:
            print(f"PLOT {number:02d}/14 | {label}")
            results.append(plot_transition(base, number, label, rest_reference_nm, observed_nm, flux, source))
        except Exception as exc:
            print(f"FAILED {number:02d}/14 | {label} | {type(exc).__name__}: {exc}")
            failures.append((number, label, type(exc).__name__, str(exc)))

    if not results:
        raise RuntimeError("No transition plots were generated")

    summary = CSV / f"{VERSION}_{base.GALAXY}_REFERENCE_LINE_INDEX.csv"
    pd.DataFrame(results).to_csv(summary, index=False)
    if failures:
        pd.DataFrame(failures, columns=["number", "label", "error_type", "message"]).to_csv(
            CSV / f"{VERSION}_{base.GALAXY}_FAILURES.csv", index=False
        )

    print()
    for row in results:
        print(
            f"{row['number']:02d} | {row['label']:<15} | "
            f"rest={row['rest_reference_nm']:8.3f} nm | "
            f"expected={row['expected_observed_nm']:9.3f} nm | "
            f"measured={row['measured_observed_nm']:9.3f} nm | samples={row['samples']:4d}"
        )
    print()
    print(f"WAVELENGTH COLUMN : {wavelength_column}")
    print(f"FLUX COLUMN       : {flux_column}")
    print(f"REST HALF-WIDTH   : {REST_HALF_WIDTH_NM:.3f} nm")
    print(f"PEAK SEARCH WIDTH : {PEAK_SEARCH_HALF_WIDTH_NM:.3f} nm")
    print(f"GENERATED         : {len(results)} PNG + {len(results)} CSV")
    print(f"SUMMARY CSV       : {summary}")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
