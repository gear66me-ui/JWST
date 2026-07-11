# JWST_0066
# N IV only: continuous NIST ASD-derived reference trace versus real JWST samples.
# The left trace is built from actual NIST wavelengths and tabulated strengths,
# convolved with a stated instrument response. It is not copied JWST data and
# is not claimed to be a raw laboratory detector trace.
# No AI images. Matplotlib only.

from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VERSION = "JWST_0066"
BASE_NAME = "JWST_0065_NIV_NIST_LAB_VS_JWST.py"
BASE_URL = f"https://raw.githubusercontent.com/gear66me-ui/JWST/main/{BASE_NAME}"
BASE_PATH = Path("/content") / BASE_NAME if Path("/content").exists() else Path.cwd() / BASE_NAME

OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"

REST_LOW_NM = 147.90
REST_HIGH_NM = 149.05
REFERENCE_RESOLVING_POWER = 2500.0
GRID_SAMPLES = 5000
LAB_COLOR = "#ffd84d"
JWST_COLOR = "#ff9d2e"
POINT_COLOR = "#d9edf7"
MARKER_COLOR = "#ff5a66"


def ensure_base():
    if BASE_PATH.exists() and BASE_PATH.stat().st_size > 8000:
        return BASE_PATH
    subprocess.run(
        ["curl", "-fsSL", "--connect-timeout", "15", "--max-time", "90",
         "-o", str(BASE_PATH), BASE_URL],
        check=True,
        timeout=100,
    )
    if not BASE_PATH.exists() or BASE_PATH.stat().st_size < 8000:
        raise RuntimeError("Could not download JWST_0065 helper script")
    return BASE_PATH


def load_base(path):
    spec = importlib.util.spec_from_file_location("jwst_0065_base", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load JWST_0065 helper module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def query_nist_trace_data(base):
    base.REST_LOW_NM = REST_LOW_NM
    base.REST_HIGH_NM = REST_HIGH_NM
    lines, selected, raw_path, selected_path = base.query_nist_niv()
    needed = {"wavelength_nm", "strength_normalized"}
    missing = needed.difference(lines.columns)
    if missing:
        raise RuntimeError(f"NIST helper output is missing columns: {sorted(missing)}")
    lines = lines[
        np.isfinite(lines["wavelength_nm"])
        & np.isfinite(lines["strength_normalized"])
        & (lines["wavelength_nm"] >= REST_LOW_NM)
        & (lines["wavelength_nm"] <= REST_HIGH_NM)
    ].copy()
    if lines.empty:
        raise RuntimeError("No usable NIST N IV line strengths in the requested window")
    return lines.sort_values("wavelength_nm"), selected, raw_path, selected_path


def make_trace(lines, base):
    wave = np.linspace(REST_LOW_NM, REST_HIGH_NM, GRID_SAMPLES)
    trace = np.zeros_like(wave)
    for row in lines.itertuples():
        center = float(row.wavelength_nm)
        amplitude = float(row.strength_normalized)
        fwhm = center / REFERENCE_RESOLVING_POWER
        sigma = fwhm / 2.354820045
        trace += amplitude * np.exp(-0.5 * ((wave - center) / sigma) ** 2)
    maximum = float(np.nanmax(trace))
    if not np.isfinite(maximum) or maximum <= 0:
        raise RuntimeError("NIST line data could not produce a finite continuous trace")
    trace /= maximum
    path = CSV / f"{VERSION}_NIST_NIV_CONTINUOUS_TRACE.csv"
    pd.DataFrame({
        "rest_wavelength_nm": wave,
        "rest_frequency_THz": base.frequency_thz(wave),
        "nist_asd_trace_normalized": trace,
        "rendering_resolving_power": REFERENCE_RESOLVING_POWER,
    }).to_csv(path, index=False)
    return wave, trace, path


def full_limits(values):
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    pad = 0.055 * (high - low) if high > low else max(abs(high) * 0.08, 1e-8)
    return low - pad, high + pad


def matched_centers(lines):
    centers = []
    for target in [148.332, 148.650]:
        index = (lines["wavelength_nm"] - target).abs().idxmin()
        value = float(lines.loc[index, "wavelength_nm"])
        if abs(value - target) <= 0.25:
            centers.append(value)
    centers = sorted(set(centers))
    if not centers:
        raise RuntimeError("NIST N IV doublet centers were not found")
    return centers


def build_plot(base, lines, trace_wave, trace, observed_nm, flux, source_path, source_status):
    obs_low = REST_LOW_NM * base.STRETCH
    obs_high = REST_HIGH_NM * base.STRETCH
    mask = (
        np.isfinite(observed_nm) & np.isfinite(flux)
        & (observed_nm >= obs_low) & (observed_nm <= obs_high)
    )
    if int(mask.sum()) < 5:
        raise RuntimeError(f"Only {int(mask.sum())} JWST samples lie in the N IV window")
    x_obs, y_obs = observed_nm[mask], flux[mask]
    order = np.argsort(x_obs)
    x_obs, y_obs = x_obs[order], y_obs[order]
    centers = matched_centers(lines)

    fig, (left, right) = plt.subplots(1, 2, figsize=(15.8, 6.8), facecolor=base.BG)
    base.style(left)
    base.style(right)

    left.plot(trace_wave, trace, color=LAB_COLOR, linewidth=1.55)
    left.fill_between(trace_wave, 0.0, trace, color=LAB_COLOR, alpha=0.10)
    right.plot(x_obs, y_obs, color=JWST_COLOR, linewidth=0.82, alpha=0.92)
    right.scatter(x_obs, y_obs, s=22, color=POINT_COLOR,
                  edgecolor=base.BG, linewidth=0.30, zorder=4)

    for center in centers:
        left.axvline(center, color=MARKER_COLOR, linestyle=(0, (3, 4)),
                     linewidth=0.72, alpha=0.82)
        right.axvline(center * base.STRETCH, color=MARKER_COLOR,
                      linestyle=(0, (3, 4)), linewidth=0.72, alpha=0.82)

    left.set_xlim(REST_LOW_NM, REST_HIGH_NM)
    left.set_ylim(0.0, 1.08)
    right.set_xlim(obs_low, obs_high)
    right.set_ylim(*full_limits(y_obs))

    left.set_title("N IV — NIST ASD-DERIVED CONTINUOUS TRACE", fontsize=11.2, pad=13)
    right.set_title("N IV — REAL JWST OBSERVED SPECTRUM", fontsize=11.2, pad=13)
    left.set_xlabel("Rest wavelength, nm")
    right.set_xlabel("Observed wavelength, nm")
    left.set_ylabel("Normalized NIST line-strength trace")
    right.set_ylabel("JWST flux samples")

    top_left = left.secondary_xaxis("top", functions=(base.frequency_thz, base.wavelength_nm))
    top_left.set_xlabel("Rest frequency, THz", color=base.TEXT, labelpad=7)
    top_left.tick_params(colors=base.TEXT, labelsize=8)
    top_right = right.secondary_xaxis("top", functions=(base.frequency_thz, base.wavelength_nm))
    top_right.set_xlabel("Observed frequency, THz", color=base.TEXT, labelpad=7)
    top_right.tick_params(colors=base.TEXT, labelsize=8)

    center_text = "\n".join(
        f"{center:.6f} nm / {base.frequency_thz(center):.3f} THz"
        for center in centers
    )
    left.text(0.018, 0.955,
              f"NIST ASD wavelengths + strengths\nR={REFERENCE_RESOLVING_POWER:.0f} display response\n{center_text}",
              transform=left.transAxes, ha="left", va="top", color=base.TEXT,
              fontsize=7.6,
              bbox=dict(boxstyle="round,pad=.30", facecolor="#07111f",
                        edgecolor=LAB_COLOR, linewidth=.70, alpha=.95))
    right.text(0.018, 0.955,
               f"real JWST samples={len(x_obs)}\npublished z={base.Z:.2f}\nred dashed=NIST centers × (1+z)",
               transform=right.transAxes, ha="left", va="top", color=base.TEXT,
               fontsize=7.6,
               bbox=dict(boxstyle="round,pad=.30", facecolor="#07111f",
                         edgecolor=JWST_COLOR, linewidth=.70, alpha=.95))

    fig.suptitle(f"{VERSION} — {base.GALAXY}: N IV NIST TRACE versus JWST",
                 color=base.TEXT, fontsize=15.0, fontweight="bold", y=.985)
    fig.text(.5, .925,
             "Different datasets: NIST ASD atomic line data on the left; measured JWST flux samples on the right.",
             ha="center", color=base.MUTED, fontsize=8.9)
    fig.text(.5, .018,
             f"JWST source: {source_path.name} ({source_status}). The left curve is an instrument-convolved rendering of NIST data, not a raw detector trace.",
             ha="center", color=base.MUTED, fontsize=7.9)
    fig.subplots_adjust(left=.075, right=.985, top=.84, bottom=.12, wspace=.16)

    png_path = PNG / f"{VERSION}_{base.GALAXY}_NIV_NIST_TRACE_VS_JWST.png"
    fig.savefig(png_path, dpi=245, facecolor=base.BG, edgecolor=base.BG)
    plt.show()
    plt.close(fig)

    jwst_path = CSV / f"{VERSION}_{base.GALAXY}_NIV_JWST_WINDOW.csv"
    pd.DataFrame({
        "observed_wavelength_nm": x_obs,
        "observed_frequency_THz": base.frequency_thz(x_obs),
        "jwst_flux": y_obs,
    }).to_csv(jwst_path, index=False)
    return png_path, jwst_path, len(x_obs), centers


def main():
    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)
    base = load_base(ensure_base())
    source_path, source_status = base.locate_jwst_csv()
    observed_nm, flux, wave_column, flux_column = base.load_jwst(source_path)
    lines, selected, raw_path, selected_path = query_nist_trace_data(base)
    trace_wave, trace, trace_path = make_trace(lines, base)
    plot_path, jwst_path, sample_count, centers = build_plot(
        base, lines, trace_wave, trace, observed_nm, flux, source_path, source_status)

    print(f"CODE OUTPUT: {VERSION}")
    print("LEFT DATA          : NIST ASD wavelengths + tabulated strengths")
    print("LEFT TRACE         : continuous instrument-convolved rendering")
    print(f"REFERENCE R        : {REFERENCE_RESOLVING_POWER:.3f}")
    print(f"RIGHT DATA         : {source_path}")
    print(f"JWST SOURCE STATUS : {source_status}")
    print(f"REST WINDOW        : {REST_LOW_NM:.6f} to {REST_HIGH_NM:.6f} nm")
    print(f"OBSERVED WINDOW    : {REST_LOW_NM * base.STRETCH:.6f} to {REST_HIGH_NM * base.STRETCH:.6f} nm")
    print(f"NIST CENTERS       : {', '.join(f'{value:.6f}' for value in centers)} nm")
    print(f"JWST SAMPLES       : {sample_count}")
    print(f"WAVELENGTH COLUMN  : {wave_column}")
    print(f"FLUX COLUMN        : {flux_column}")
    print(f"PLOT PNG           : {plot_path}")
    print(f"NIST RAW CSV       : {raw_path}")
    print(f"NIST SELECTED CSV  : {selected_path}")
    print(f"NIST TRACE CSV     : {trace_path}")
    print(f"JWST WINDOW CSV    : {jwst_path}")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
