#!/usr/bin/env python3
"""MoM-z14 native-sampling audit using every uploaded PRISM/CLEAR X1D.

The Stage-3 spectrum has only about ten independent wavelength channels inside a
+/-50 Angstrom rest-frame window. This script shows those channels honestly,
adds every available exposure-level measurement in a separate S/N panel, and
evaluates the fitted physical model on a dense grid only for visual clarity.
Dense model points are never represented as additional measured data.
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

VERSION = "JWST_0103"
TARGET = "MoM-z14"
PUBLISHED_Z = 14.44
DENSE_MODEL_POINTS = 2000

ROOT = Path("/content/JWST_OUTPUT")
PNG = ROOT / "PNG"
CSV = ROOT / "CSV"
DATA = ROOT / "DATA" / VERSION
for directory in (PNG, CSV, DATA):
    directory.mkdir(parents=True, exist_ok=True)

BASE = "https://raw.githubusercontent.com/gear66me-ui/JWST/main"
ARTIFACT_BASE = BASE + "/artifacts/MOMZ14/JWST_0101"
CANDIDATE_NAME = "JWST_0098_MOMZ14_MATCHED_X1D_CANDIDATES.csv"
CANDIDATE_URL = ARTIFACT_BASE + "/" + CANDIDATE_NAME
CANDIDATE_PATH = DATA / CANDIDATE_NAME
AUDIT_SCRIPT = DATA / "JWST_0102_MOMZ14_STAGE3_REDSHIFT_FIT.py"
AUDIT_URL = BASE + "/JWST_0102_MOMZ14_STAGE3_REDSHIFT_FIT.py"


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


def fetch(url: str, path: Path) -> Path:
    if not path.exists() or path.stat().st_size < 100:
        urllib.request.urlretrieve(url, path)
    return path


def load_audit_module():
    fetch(AUDIT_URL, AUDIT_SCRIPT)
    spec = importlib.util.spec_from_file_location("jwst0102", AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to import JWST_0102 redshift functions.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artifact_filename(rank: int, product: str, hdu: int) -> str:
    stem = Path(product).stem
    return f"JWST_0098_{rank:02d}_{stem}_HDU{hdu}_NATIVE_X1D.csv"


def load_all_spectra() -> tuple[pd.DataFrame, list[tuple[str, pd.DataFrame]], Path]:
    fetch(CANDIDATE_URL, CANDIDATE_PATH)
    candidates = pd.read_csv(CANDIDATE_PATH)
    loaded: list[tuple[str, pd.DataFrame]] = []
    stage3 = None
    stage3_path = None
    for _, row in candidates.iterrows():
        rank = int(row["rank"])
        product = str(row["product"])
        hdu = int(row["hdu"])
        name = artifact_filename(rank, product, hdu)
        path = DATA / name
        fetch(ARTIFACT_BASE + "/" + name, path)
        frame = pd.read_csv(path)
        for col in ("observed_wavelength_um", "flux_native_display_units",
                    "flux_error_native_display_units"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame[np.isfinite(frame.observed_wavelength_um) &
                      np.isfinite(frame.flux_native_display_units)].copy()
        frame = frame.sort_values("observed_wavelength_um").drop_duplicates("observed_wavelength_um")
        loaded.append((name, frame))
        if "-o004_s000277193_" in name:
            stage3 = frame.copy()
            stage3_path = path
    if stage3 is None or stage3_path is None:
        raise RuntimeError("Exact Stage-3 combined source 277193 was not found.")
    return stage3.reset_index(drop=True), loaded, stage3_path


def channel_half_widths(x: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.full_like(x, np.nan)
    edges = np.empty(len(x) + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return 0.5 * (edges[1:] - edges[:-1])


def dense_line_model(audit, wave, flux, error, group, z, dense_wave):
    expected = np.array([rest * (1.0 + z) * 1e-4 for _, rest, _ in group["lines"]])
    weights = np.array([weight for _, _, weight in group["lines"]], dtype=float)
    center = float(np.average(expected, weights=weights))
    half_window = max(0.060, 50.0 * (1.0 + z) * 1e-4)
    mask = (wave >= center - half_window) & (wave <= center + half_window)
    x, y, e = wave[mask], flux[mask], error[mask]
    template = np.zeros_like(x)
    dense_template = np.zeros_like(dense_wave)
    for expected_line, relative in zip(expected, weights):
        sigma = expected_line / (audit.PRISM_R * 2.355)
        template += relative * np.exp(-0.5 * ((x - expected_line) / sigma) ** 2)
        dense_template += relative * np.exp(-0.5 * ((dense_wave - expected_line) / sigma) ** 2)
    scale = float(np.nanmax(template)) if len(template) else 1.0
    if scale > 0:
        template /= scale
        dense_template /= scale
    dx = x - center
    matrix = np.column_stack([np.ones_like(x), dx, template])
    coeff, _, _, _ = audit.weighted_linear_fit(matrix, y, e)
    dense_dx = dense_wave - center
    dense_matrix = np.column_stack([np.ones_like(dense_wave), dense_dx, dense_template])
    return dense_matrix @ coeff


def dense_break_model(audit, wave, flux, error, z, dense_wave):
    _, _, _, _, details = audit.break_fit_at_z(wave, flux, error, z)
    _, _, coeff, sigma, edge = details
    step = 0.5 * (1.0 + erf((dense_wave - edge) / (np.sqrt(2.0) * sigma)))
    dx = dense_wave - edge
    matrix = np.column_stack([np.ones_like(dense_wave), dx, step, step * dx])
    return matrix @ coeff


def plot_group(audit, group, index, z, stage3, all_frames, source_name):
    wave = stage3.observed_wavelength_um.to_numpy(float)
    flux = stage3.flux_native_display_units.to_numpy(float)
    error = stage3.flux_error_native_display_units.to_numpy(float)
    good_error = np.isfinite(error) & (error > 0)
    replacement = float(np.nanmedian(error[good_error])) if good_error.any() else 1.0
    error = np.where(good_error, error, replacement)

    low_rest = group["center_A"] - 50.0
    high_rest = group["center_A"] + 50.0
    low_obs = low_rest * (1.0 + z) * 1e-4
    high_obs = high_rest * (1.0 + z) * 1e-4
    mask = (wave >= low_obs) & (wave <= high_obs)
    x_obs, y, e = wave[mask], flux[mask], error[mask]
    x_rest = x_obs * 1e4 / (1.0 + z)
    xerr_rest = channel_half_widths(x_obs) * 1e4 / (1.0 + z)

    dense_obs = np.linspace(low_obs, high_obs, DENSE_MODEL_POINTS)
    dense_rest = dense_obs * 1e4 / (1.0 + z)
    if group["key"] == "LYMAN_ALPHA_BREAK":
        dense_y = dense_break_model(audit, wave, flux, error, z, dense_obs)
        model_label = f"Instrument-convolved break model ({DENSE_MODEL_POINTS:,} evaluations; not data)"
    else:
        dense_y = dense_line_model(audit, wave, flux, error, group, z, dense_obs)
        model_label = f"Instrument-convolved line model ({DENSE_MODEL_POINTS:,} evaluations; not data)"

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(17.8, 11.4), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]}, constrained_layout=True,
    )
    ax.errorbar(
        x_rest, y, yerr=e, xerr=xerr_rest, fmt="o", ms=5.2, capsize=2.0,
        color="#f8fbff", ecolor=group["color"], elinewidth=0.9,
        label=f"Stage-3 native channels (n={len(y)}) with channel widths",
    )
    ax.plot(dense_rest, dense_y, lw=2.1, color=group["color"], label=model_label)
    for label, rest, _ in group["lines"]:
        ax.axvline(rest, color=group["color"], lw=1.1, ls="--", alpha=0.9)
        ax.text(rest, 0.98, label, transform=ax.get_xaxis_transform(), rotation=90,
                va="top", ha="right", fontsize=7.8, color=group["color"])
    ax.axhline(0.0, lw=0.5, alpha=0.55)
    ax.grid(True, lw=0.42, alpha=0.38)
    ax.set_ylabel("Flux density [nJy]")
    ax.set_title(
        f"{TARGET} — {group['title']} | native PRISM sampling versus dense fitted model\n"
        f"rest window {low_rest:.1f}-{high_rest:.1f} Å | measured z={z:.5f}",
        fontsize=14.2,
    )
    ax.legend(loc="best", fontsize=8.3)

    exposure_sample_count = 0
    exposure_count = 0
    unique_channel_values = []
    for name, frame in all_frames:
        if "-o004_s000277193_" in name:
            continue
        w = frame.observed_wavelength_um.to_numpy(float)
        f = frame.flux_native_display_units.to_numpy(float)
        er = frame.flux_error_native_display_units.to_numpy(float)
        m = (w >= low_obs) & (w <= high_obs) & np.isfinite(f) & np.isfinite(er) & (er > 0)
        if not m.any():
            continue
        rr = w[m] * 1e4 / (1.0 + z)
        snr = f[m] / er[m]
        bx.plot(rr, snr, marker=".", ms=3.2, lw=0.45, alpha=0.48)
        exposure_sample_count += int(m.sum())
        exposure_count += 1
        unique_channel_values.extend(np.round(w[m], 7).tolist())
    bx.axhline(0.0, lw=0.65, alpha=0.7)
    bx.axhline(3.0, lw=0.55, ls="--", alpha=0.45)
    bx.axhline(-3.0, lw=0.55, ls="--", alpha=0.45)
    bx.grid(True, lw=0.38, alpha=0.35)
    bx.set_ylabel("Per-exposure flux/error")
    bx.set_xlabel(f"Rest-frame vacuum wavelength [Å], using measured z={z:.5f}")
    bx.set_xlim(low_rest, high_rest)
    unique_count = len(set(unique_channel_values))
    bx.text(
        0.012, 0.045,
        f"All individual X1D measurements: {exposure_sample_count} samples from {exposure_count} files; "
        f"unique wavelength channels ≈ {unique_count}.\n"
        "Repeated exposures improve S/N, not spectral resolution. No exposure samples were interpolated.",
        transform=bx.transAxes, fontsize=8.2, va="bottom",
    )
    top = ax.secondary_xaxis("top", functions=(
        lambda rest: rest * (1.0 + z) * 1e-4,
        lambda obs: obs * 1e4 / (1.0 + z),
    ))
    top.set_xlabel("Observed wavelength [µm]")

    path = PNG / f"{VERSION}_{index:02d}_{group['key']}_ALL_EXPOSURES.png"
    fig.savefig(path, dpi=420, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    row = {
        "region": group["key"],
        "stage3_native_channels": len(y),
        "individual_exposure_samples": exposure_sample_count,
        "individual_exposure_files": exposure_count,
        "approximately_unique_wavelength_channels": unique_count,
        "dense_model_evaluations": DENSE_MODEL_POINTS,
        "dense_model_is_measured_data": False,
        "measured_redshift": z,
        "source": source_name,
    }
    return path, row


def plot_sampling_overview(stage3: pd.DataFrame, z: float) -> Path:
    wave = stage3.observed_wavelength_um.to_numpy(float)
    flux = stage3.flux_native_display_units.to_numpy(float)
    delta = np.diff(wave)
    midpoint = 0.5 * (wave[:-1] + wave[1:])
    sampling_r = midpoint / delta
    fig, ax = plt.subplots(figsize=(17.5, 8.2), constrained_layout=True)
    ax.plot(wave, flux, lw=0.65, label=f"Complete Stage-3 spectrum ({len(wave)} measured channels)")
    ax.scatter(wave, flux, s=7, alpha=0.75)
    ax.set_xlabel("Observed wavelength [µm]")
    ax.set_ylabel("Flux density [nJy]")
    ax.grid(True, lw=0.42, alpha=0.38)
    ax.set_title(
        f"{TARGET} — complete native Stage-3 PRISM/CLEAR sampling\n"
        f"401 channels over {wave.min():.3f}-{wave.max():.3f} µm; median sampling R≈{np.nanmedian(sampling_r):.1f}",
        fontsize=14.5,
    )
    ax.legend(loc="best")
    ax.text(
        0.012, 0.025,
        "A narrow ±50 Å rest-frame window spans only about 0.155 µm observed at z≈14.5, "
        "so it naturally contains about ten native PRISM channels.",
        transform=ax.transAxes, fontsize=8.5,
    )
    path = PNG / f"{VERSION}_COMPLETE_STAGE3_NATIVE_SAMPLING.png"
    fig.savefig(path, dpi=420, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return path


def main() -> None:
    style()
    audit = load_audit_module()
    stage3, all_frames, stage3_path = load_all_spectra()

    wave = stage3.observed_wavelength_um.to_numpy(float)
    flux = stage3.flux_native_display_units.to_numpy(float)
    error = stage3.flux_error_native_display_units.to_numpy(float)
    good = np.isfinite(error) & (error > 0)
    replacement = float(np.nanmedian(error[good])) if good.any() else 1.0
    error = np.where(good, error, replacement)

    break_loglike, _, *_ = audit.break_grid(wave, flux, error)
    line_loglike, _, _ = audit.line_grid(wave, flux, error)
    combined_prob, measured_z, z_low, z_median, z_high = audit.posterior(break_loglike + line_loglike)

    overview = plot_sampling_overview(stage3, measured_z)
    outputs = []
    rows = []
    for index, group in enumerate(audit.GROUPS, 1):
        path, row = plot_group(audit, group, index, measured_z, stage3, all_frames, stage3_path.name)
        outputs.append(path)
        rows.append(row)

    table = pd.DataFrame(rows)
    table_path = CSV / f"{VERSION}_NATIVE_SAMPLE_COUNTS.csv"
    table.to_csv(table_path, index=False)
    result_path = CSV / f"{VERSION}_REDSHIFT_RESULT.csv"
    pd.DataFrame([{
        "measured_map_z": measured_z,
        "credible_low_16pct": z_low,
        "median_z": z_median,
        "credible_high_84pct": z_high,
        "published_z_comparison": PUBLISHED_Z,
        "stage3_total_channels": len(stage3),
        "all_x1d_files": len(all_frames),
    }]).to_csv(result_path, index=False)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"TARGET          {TARGET}")
    print(f"STAGE-3 SOURCE  {stage3_path}")
    print(f"ALL X1D FILES   {len(all_frames)}")
    print(f"STAGE-3 POINTS  {len(stage3)} native wavelength channels")
    print(f"MEASURED z      {measured_z:.5f}  [{z_low:.5f}, {z_high:.5f}]")
    print(f"PUBLISHED z     {PUBLISHED_Z:.5f} comparison only")
    print("MODEL GRID      2,000 evaluations per panel; not measured data")
    print(f"OVERVIEW PNG    {overview}")
    for path in outputs:
        print(f"REGION PNG      {path}")
    print(f"COUNTS CSV      {table_path}")
    print(f"RESULT CSV      {result_path}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
