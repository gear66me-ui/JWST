#!/usr/bin/env python3
"""MoM-z14 Stage-3 PRISM/CLEAR redshift audit from the exact combined X1D.

Uses only the measured Stage-3 spectrum uploaded by JWST_0101. It deliberately
rejects the individual exposure-level X1D files that caused JWST_0100 to plot a
negative nod/background trace. The code fits the Ly-alpha break and performs an
instrument-resolution matched-filter search for the five reported rest-UV line
complexes. Published z=14.44 is shown only as an external comparison and is not
used as a prior or forced solution.
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
from scipy.special import erf

VERSION = "JWST_0102"
TARGET = "MoM-z14"
SOURCE_ID = 277193
PUBLISHED_Z = 14.44
LYA_A = 1215.67
PRISM_R = 100.0
Z_GRID = np.linspace(13.75, 15.05, 1301)

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

GROUPS = [
    {
        "key": "LYMAN_ALPHA_BREAK", "title": "H I Lyman-alpha break",
        "center_A": 1215.67, "color": "#ff9f1c", "lines": [("H I Ly-alpha", 1215.67, 1.0)],
    },
    {
        "key": "N_IV", "title": "N IV] doublet", "center_A": 1486.50,
        "color": "#ff4fa3", "lines": [("N IV] 1483", 1483.32, 0.45), ("N IV] 1487", 1486.50, 1.0)],
    },
    {
        "key": "C_IV", "title": "C IV doublet", "center_A": 1550.00,
        "color": "#25c7f7", "lines": [("C IV 1548", 1548.20, 1.0), ("C IV 1551", 1550.77, 0.55)],
    },
    {
        "key": "HEII_OIII", "title": "He II + O III] complex", "center_A": 1650.00,
        "color": "#b978ff", "lines": [
            ("He II 1640", 1640.42, 0.85), ("O III] 1661", 1660.81, 0.45),
            ("O III] 1666", 1666.15, 1.0),
        ],
    },
    {
        "key": "N_III", "title": "N III] multiplet", "center_A": 1750.00,
        "color": "#ef4444", "lines": [
            ("N III] 1747", 1746.82, 0.35), ("N III] 1749", 1748.65, 0.65),
            ("N III] 1750", 1749.67, 1.0), ("N III] 1752", 1752.16, 0.65),
            ("N III] 1754", 1753.99, 0.35),
        ],
    },
    {
        "key": "C_III", "title": "C III] doublet", "center_A": 1900.00,
        "color": "#22c55e", "lines": [("C III] 1907", 1906.68, 0.70), ("C III] 1909", 1908.73, 1.0)],
    },
]
LINE_GROUPS = GROUPS[1:]


def style() -> None:
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


def download_source() -> Path:
    local_candidates = [
        CSV / SOURCE_NAME,
        ROOT / "CSV" / SOURCE_NAME,
        Path("/content") / SOURCE_NAME,
    ]
    for candidate in local_candidates:
        if candidate.exists() and candidate.stat().st_size > 5000:
            return candidate
    if not SOURCE_PATH.exists() or SOURCE_PATH.stat().st_size < 5000:
        urllib.request.urlretrieve(SOURCE_URL, SOURCE_PATH)
    return SOURCE_PATH


def load_stage3() -> tuple[pd.DataFrame, Path]:
    path = download_source()
    frame = pd.read_csv(path)
    required = {"observed_wavelength_um", "flux_native_display_units", "flux_error_native_display_units"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Stage-3 CSV missing columns: {sorted(missing)}")
    sid = pd.to_numeric(frame.get("source_id", np.nan), errors="coerce")
    if sid.notna().any() and int(sid.dropna().iloc[0]) != SOURCE_ID:
        raise RuntimeError("Source ID is not the exact MoM-z14 Stage-3 extraction.")
    grating = str(frame.get("grating", pd.Series([""])).iloc[0]).upper()
    filt = str(frame.get("filter", pd.Series([""])).iloc[0]).upper()
    if grating != "PRISM" or filt != "CLEAR":
        raise RuntimeError(f"Expected PRISM/CLEAR; received {grating}/{filt}.")
    frame = frame.copy()
    for column in ("observed_wavelength_um", "flux_native_display_units", "flux_error_native_display_units"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[np.isfinite(frame.observed_wavelength_um) & np.isfinite(frame.flux_native_display_units)].copy()
    frame = frame.sort_values("observed_wavelength_um").drop_duplicates("observed_wavelength_um")
    error = frame.flux_error_native_display_units.to_numpy(float)
    good = np.isfinite(error) & (error > 0)
    replacement = float(np.nanmedian(error[good])) if good.any() else 1.0
    frame.loc[~good, "flux_error_native_display_units"] = replacement
    return frame.reset_index(drop=True), path


def weighted_linear_fit(matrix: np.ndarray, y: np.ndarray, error: np.ndarray):
    weight = 1.0 / np.maximum(error, 1e-12)
    design = matrix * weight[:, None]
    target = y * weight
    coeff, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    model = matrix @ coeff
    chi2 = float(np.sum(((y - model) / error) ** 2))
    covariance = np.linalg.pinv(design.T @ design)
    return coeff, model, chi2, covariance


def break_fit_at_z(wave: np.ndarray, flux: np.ndarray, error: np.ndarray, z: float):
    mask = (wave >= 1.68) & (wave <= 2.18)
    x, y, e = wave[mask], flux[mask], error[mask]
    edge = LYA_A * (1.0 + z) * 1e-4
    best = None
    for scale in np.linspace(0.45, 1.80, 10):
        sigma = scale * edge / (PRISM_R * 2.355)
        step = 0.5 * (1.0 + erf((x - edge) / (np.sqrt(2.0) * sigma)))
        dx = x - edge
        matrix = np.column_stack([np.ones_like(x), dx, step, step * dx])
        coeff, model, trial, _ = weighted_linear_fit(matrix, y, e)
        if best is None or trial < best[0]:
            best = (trial, model, coeff, sigma, edge)
    return x, y, e, best[1], best


def break_grid(wave: np.ndarray, flux: np.ndarray, error: np.ndarray):
    chi2 = np.full_like(Z_GRID, np.nan, dtype=float)
    payloads = {}
    for index, z in enumerate(Z_GRID):
        x, y, e, model, details = break_fit_at_z(wave, flux, error, float(z))
        chi2[index] = details[0]
        payloads[index] = (x, y, e, model, details)
    loglike = -0.5 * (chi2 - np.nanmin(chi2))
    best_index = int(np.nanargmax(loglike))
    x, y, e, model, details = payloads[best_index]
    return loglike, Z_GRID[best_index], x, y, e, model, details


def group_fit(wave: np.ndarray, flux: np.ndarray, error: np.ndarray, group: dict, z: float):
    expected = np.array([rest * (1.0 + z) * 1e-4 for _, rest, _ in group["lines"]])
    weights = np.array([weight for _, _, weight in group["lines"]], dtype=float)
    center = float(np.average(expected, weights=weights))
    half_window = max(0.060, 50.0 * (1.0 + z) * 1e-4)
    mask = (wave >= center - half_window) & (wave <= center + half_window)
    x, y, e = wave[mask], flux[mask], error[mask]
    if len(x) < 5:
        return {"snr": np.nan, "delta_chi2": 0.0, "amplitude": np.nan, "amplitude_error": np.nan,
                "x": x, "y": y, "e": e, "continuum": np.full_like(x, np.nan),
                "model": np.full_like(x, np.nan), "template": np.full_like(x, np.nan), "expected": expected}
    template = np.zeros_like(x)
    for expected_line, relative in zip(expected, weights):
        sigma = expected_line / (PRISM_R * 2.355)
        template += relative * np.exp(-0.5 * ((x - expected_line) / sigma) ** 2)
    if np.nanmax(template) > 0:
        template /= np.nanmax(template)
    dx = x - center
    matrix = np.column_stack([np.ones_like(x), dx, template])
    coeff, model, chi2_line, covariance = weighted_linear_fit(matrix, y, e)
    null_matrix = np.column_stack([np.ones_like(x), dx])
    _, continuum, chi2_null, _ = weighted_linear_fit(null_matrix, y, e)
    amplitude = float(coeff[-1])
    amplitude_error = float(np.sqrt(max(covariance[-1, -1], 0.0)))
    snr = amplitude / amplitude_error if amplitude_error > 0 else np.nan
    delta = max(0.0, chi2_null - chi2_line) if amplitude > 0 else 0.0
    return {"snr": snr, "delta_chi2": delta, "amplitude": amplitude,
            "amplitude_error": amplitude_error, "x": x, "y": y, "e": e,
            "continuum": continuum, "model": model, "template": template, "expected": expected}


def line_grid(wave: np.ndarray, flux: np.ndarray, error: np.ndarray):
    columns = {group["key"]: np.zeros_like(Z_GRID) for group in LINE_GROUPS}
    total = np.zeros_like(Z_GRID)
    for index, z in enumerate(Z_GRID):
        for group in LINE_GROUPS:
            result = group_fit(wave, flux, error, group, float(z))
            score = 0.5 * result["delta_chi2"]
            columns[group["key"]][index] = score
            total[index] += score
    total -= np.nanmax(total)
    best = float(Z_GRID[int(np.nanargmax(total))])
    return total, best, columns


def posterior(loglike: np.ndarray):
    probability = np.exp(loglike - np.nanmax(loglike))
    area = np.trapz(probability, Z_GRID)
    probability = probability / area if area > 0 else np.full_like(probability, 1.0 / len(probability))
    cumulative = np.cumsum((probability[:-1] + probability[1:]) * 0.5 * np.diff(Z_GRID))
    cumulative = np.concatenate([[0.0], cumulative])
    cumulative /= cumulative[-1]
    quantiles = np.interp([0.16, 0.50, 0.84], cumulative, Z_GRID)
    map_z = float(Z_GRID[int(np.nanargmax(probability))])
    return probability, map_z, float(quantiles[0]), float(quantiles[1]), float(quantiles[2])


def plot_likelihood(break_prob, line_prob, combined_prob, results: dict) -> Path:
    fig, ax = plt.subplots(figsize=(15.5, 7.2), constrained_layout=True)
    ax.plot(Z_GRID, break_prob / break_prob.max(), lw=1.6, label="Ly-alpha break likelihood")
    ax.plot(Z_GRID, line_prob / line_prob.max(), lw=1.6, label="Five-complex matched-filter likelihood")
    ax.plot(Z_GRID, combined_prob / combined_prob.max(), lw=2.5, label="Combined likelihood")
    ax.axvline(PUBLISHED_Z, color="#facc15", lw=1.3, ls="--", label="Published z = 14.44 (comparison only)")
    ax.axvline(results["combined_map"], color="#ffffff", lw=1.2, ls=":", label=f"Measured MAP z = {results['combined_map']:.4f}")
    ax.axvspan(results["combined_low"], results["combined_high"], alpha=0.15, label="Combined 68% interval")
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Relative likelihood")
    ax.set_title("MoM-z14 redshift audit — exact combined Stage-3 PRISM/CLEAR X1D")
    ax.grid(True, lw=0.45, alpha=0.38)
    ax.legend(loc="upper right", fontsize=8.8)
    path = PNG / f"{VERSION}_REDSHIFT_LIKELIHOOD.png"
    fig.savefig(path, dpi=420, bbox_inches="tight")
    plt.show(); plt.close(fig)
    return path


def plot_regions(wave, flux, error, best_z, break_payload, source_name):
    outputs, measurement_rows = [], []
    for index, group in enumerate(GROUPS, 1):
        rest_at_best = wave * 1e4 / (1.0 + best_z)
        low, high = group["center_A"] - 50.0, group["center_A"] + 50.0
        mask = (rest_at_best >= low) & (rest_at_best <= high)
        x_rest, x_obs = rest_at_best[mask], wave[mask]
        y, e = flux[mask], error[mask]
        fig, ax = plt.subplots(figsize=(16.5, 8.1), constrained_layout=True)
        ax.errorbar(x_rest, y, yerr=e, fmt="o-", ms=3.8, lw=0.75, capsize=1.8,
                    color="#eef7ff", ecolor=group["color"], alpha=0.95,
                    label=f"Measured Stage-3 samples (n={len(y)})")
        if group["key"] == "LYMAN_ALPHA_BREAK":
            bx, _, _, model, payload = break_payload
            b_rest = bx * 1e4 / (1.0 + best_z)
            inside = (b_rest >= low) & (b_rest <= high)
            ax.plot(b_rest[inside], model[inside], color=group["color"], lw=2.0,
                    label="Best-fit instrument-convolved break")
            score_snr = np.nan
        else:
            fit = group_fit(wave, flux, error, group, best_z)
            fit_rest = fit["x"] * 1e4 / (1.0 + best_z)
            ax.plot(fit_rest, fit["continuum"], color="#facc15", lw=1.15, ls="--", label="Local fitted continuum")
            ax.plot(fit_rest, fit["model"], color=group["color"], lw=2.0,
                    label=f"Resolution-matched line fit; S/N={fit['snr']:.2f}")
            score_snr = fit["snr"]
        for label, rest, _ in group["lines"]:
            ax.axvline(rest, color=group["color"], lw=1.35, ls="--", alpha=0.95)
            published_rest_on_best_axis = rest * (1.0 + PUBLISHED_Z) / (1.0 + best_z)
            ax.axvline(published_rest_on_best_axis, color="#94a3b8", lw=0.75, ls=":", alpha=0.75)
            ax.text(rest, ax.get_ylim()[1] if ax.get_ylim()[1] else 1, label, rotation=90,
                    ha="right", va="top", fontsize=7.8, color=group["color"])
        ax.set_xlim(low, high)
        ax.axhline(0, lw=0.5, alpha=0.55)
        ax.grid(True, lw=0.42, alpha=0.38)
        ax.set_xlabel(f"Rest-frame vacuum wavelength [Å], using measured z={best_z:.4f}")
        ax.set_ylabel("Flux density [nJy]")
        ax.set_title(f"{TARGET} — {group['title']} | exact Stage-3 combined PRISM/CLEAR")
        top = ax.secondary_xaxis("top", functions=(
            lambda rest: rest * (1.0 + best_z) * 1e-4,
            lambda obs: obs * 1e4 / (1.0 + best_z),
        ))
        top.set_xlabel("Observed wavelength [µm]")
        ax.legend(loc="best", fontsize=8.4)
        ax.text(0.012, 0.025,
                "Colored dashed: fitted-z reference   |   gray dotted: published z=14.44 reference\n"
                "Observed points are unaltered; only the overlaid fit is instrument-convolved.",
                transform=ax.transAxes, fontsize=8.1, va="bottom")
        path = PNG / f"{VERSION}_{index:02d}_{group['key']}_STAGE3_REDSHIFT.png"
        fig.savefig(path, dpi=420, bbox_inches="tight")
        plt.show(); plt.close(fig)
        outputs.append(path)
        measurement_rows.append({
            "region": group["key"], "center_A": group["center_A"], "samples": len(y),
            "line_fit_snr": score_snr, "measured_global_z": best_z,
            "published_z_comparison": PUBLISHED_Z, "source": source_name,
        })
    return outputs, pd.DataFrame(measurement_rows)


def plot_overview(wave, flux, error, best_z, source_name) -> Path:
    fig, ax = plt.subplots(figsize=(18, 8.6), constrained_layout=True)
    ax.fill_between(wave, flux - error, flux + error, alpha=0.13, linewidth=0, label="1σ uncertainty")
    ax.plot(wave, flux, lw=0.72, color="#eef7ff", label="Exact Stage-3 combined X1D")
    for group in GROUPS:
        for label, rest, _ in group["lines"]:
            observed = rest * (1.0 + best_z) * 1e-4
            ax.axvline(observed, color=group["color"], lw=0.8, ls="--", alpha=0.80)
    ax.axvline(LYA_A * (1.0 + best_z) * 1e-4, color=GROUPS[0]["color"], lw=1.8)
    ax.set_xlim(1.55, 3.15)
    ax.set_xlabel("Observed wavelength [µm]")
    ax.set_ylabel("Flux density [nJy]")
    ax.set_title(f"{TARGET} — Stage-3 PRISM/CLEAR overview | measured combined z={best_z:.4f}")
    ax.grid(True, lw=0.42, alpha=0.38)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.text(0.01, 0.025, f"Source: {source_name}", transform=ax.transAxes, fontsize=8)
    path = PNG / f"{VERSION}_STAGE3_OVERVIEW.png"
    fig.savefig(path, dpi=420, bbox_inches="tight")
    plt.show(); plt.close(fig)
    return path


def main() -> None:
    style()
    frame, source_path = load_stage3()
    wave = frame.observed_wavelength_um.to_numpy(float)
    flux = frame.flux_native_display_units.to_numpy(float)
    error = frame.flux_error_native_display_units.to_numpy(float)

    break_loglike, break_best, bx, by, be, break_model, break_details = break_grid(wave, flux, error)
    line_loglike, line_best, line_columns = line_grid(wave, flux, error)
    combined_loglike = break_loglike + line_loglike

    break_prob, break_map, break_low, break_med, break_high = posterior(break_loglike)
    line_prob, line_map, line_low, line_med, line_high = posterior(line_loglike)
    combined_prob, combined_map, combined_low, combined_med, combined_high = posterior(combined_loglike)

    ciii_loglike = line_columns["C_III"] - np.nanmax(line_columns["C_III"])
    _, ciii_map, ciii_low, ciii_med, ciii_high = posterior(ciii_loglike)

    results = {
        "break_map": break_map, "break_low": break_low, "break_median": break_med, "break_high": break_high,
        "lines_map": line_map, "lines_low": line_low, "lines_median": line_med, "lines_high": line_high,
        "ciii_map": ciii_map, "ciii_low": ciii_low, "ciii_median": ciii_med, "ciii_high": ciii_high,
        "combined_map": combined_map, "combined_low": combined_low,
        "combined_median": combined_med, "combined_high": combined_high,
        "published_z": PUBLISHED_Z,
    }

    likelihood_path = plot_likelihood(break_prob, line_prob, combined_prob, results)
    overview_path = plot_overview(wave, flux, error, combined_map, source_path.name)
    bx, by, be, break_model, break_details = break_fit_at_z(wave, flux, error, combined_map)
    break_payload = (bx, by, be, break_model, break_details)
    region_paths, measurement_table = plot_regions(
        wave, flux, error, combined_map, break_payload, source_path.name
    )

    grid_table = pd.DataFrame({
        "z": Z_GRID,
        "break_relative_likelihood": break_prob / break_prob.max(),
        "lines_relative_likelihood": line_prob / line_prob.max(),
        "combined_relative_likelihood": combined_prob / combined_prob.max(),
        **{f"{key}_relative_score": np.exp(values - np.nanmax(values)) for key, values in line_columns.items()},
    })
    grid_path = CSV / f"{VERSION}_REDSHIFT_GRID.csv"
    result_path = CSV / f"{VERSION}_REDSHIFT_RESULTS.csv"
    measurement_path = CSV / f"{VERSION}_REGION_MEASUREMENTS.csv"
    grid_table.to_csv(grid_path, index=False)
    pd.DataFrame([results]).to_csv(result_path, index=False)
    measurement_table.to_csv(measurement_path, index=False)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"TARGET          {TARGET}")
    print(f"SOURCE          {source_path}")
    print("DATA PRODUCT    Stage-3 combined PRISM/CLEAR X1D; source 277193")
    print("REJECTED        exposure-level negative nod/background X1D products")
    print(f"BREAK z         {break_map:.5f}  [{break_low:.5f}, {break_high:.5f}]")
    print(f"LINES z         {line_map:.5f}  [{line_low:.5f}, {line_high:.5f}]")
    print(f"C III] z        {ciii_map:.5f}  [{ciii_low:.5f}, {ciii_high:.5f}]")
    print(f"COMBINED z      {combined_map:.5f}  [{combined_low:.5f}, {combined_high:.5f}]")
    print(f"PUBLISHED       {PUBLISHED_Z:.5f}  comparison only; not imposed")
    print(f"OVERVIEW PNG    {overview_path}")
    print(f"LIKELIHOOD PNG  {likelihood_path}")
    for path in region_paths:
        print(f"REGION PNG      {path}")
    print(f"RESULT CSV      {result_path}")
    print(f"GRID CSV        {grid_path}")
    print(f"MEASUREMENTS    {measurement_path}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
