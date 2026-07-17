#!/usr/bin/env python3
"""
JWST_0104_MOMZ14_G235H_SEARCH_AND_PRISM_MAX_SAMPLES.py

Search MAST for any public G235H/F170LP observation at the exact MoM-z14
position. If a real G235H X1D exists, download and plot it at native sampling.
Independently, load every uploaded PRISM/CLEAR exposure plus the exact Stage-3
combined X1D and plot every measured detector sample without interpolation.

No synthetic flux spectrum is created. G235H planning products show sampling
positions/counts only and are explicitly labeled theoretical observation design.
"""
from __future__ import annotations

import importlib.util
import math
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
        "astropy": "astropy",
        "astroquery": "astroquery",
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
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from astroquery.mast import Observations

VERSION = "JWST_0104"
TARGET = "MoM-z14"
RA_DEG = 150.0933255
DEC_DEG = 2.2731627
SOURCE_ID = 277193
PUBLISHED_Z = 14.44
LYA_A = 1215.67
PRISM_R = 100.0
G235M_R = 1000.0
G235H_R = 2700.0
PIXELS_PER_RESEL = 2.2
SEARCH_RADIUS_ARCSEC = 1.0

ROOT = Path("/content/JWST_OUTPUT")
PNG = ROOT / "PNG"
CSV = ROOT / "CSV"
DATA = ROOT / "DATA" / VERSION
FITS_DIR = DATA / "G235H_FITS"
for directory in (PNG, CSV, DATA, FITS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

RAW_BASE = (
    "https://raw.githubusercontent.com/gear66me-ui/JWST/main/"
    "artifacts/MOMZ14/JWST_0101/"
)
STAGE3_NAME = "JWST_0098_04_jw05224-o004_s000277193_nirspec_clear-prism_x1d_HDU1_NATIVE_X1D.csv"
EXPOSURE_NAMES = [
    "JWST_0098_01_jw05224004001_03101_00004_nrs2_x1d_HDU12_NATIVE_X1D.csv",
    "JWST_0098_02_jw05224004001_05101_00003_nrs2_x1d_HDU12_NATIVE_X1D.csv",
    "JWST_0098_03_jw05224004001_03101_00002_nrs2_x1d_HDU12_NATIVE_X1D.csv",
    "JWST_0098_05_jw05224004001_05101_00001_nrs2_x1d_HDU12_NATIVE_X1D.csv",
    "JWST_0098_06_jw05224004001_05101_00002_nrs2_x1d_HDU12_NATIVE_X1D.csv",
    "JWST_0098_07_jw05224004001_07101_00001_nrs2_x1d_HDU12_NATIVE_X1D.csv",
    "JWST_0098_08_jw05224004001_07101_00002_nrs2_x1d_HDU12_NATIVE_X1D.csv",
    "JWST_0098_09_jw05224004001_03101_00003_nrs2_x1d_HDU12_NATIVE_X1D.csv",
    "JWST_0098_10_jw05224004001_07101_00003_nrs2_x1d_HDU12_NATIVE_X1D.csv",
]

GROUPS = [
    {"key": "LYMAN_ALPHA_BREAK", "title": "H I Lyman-alpha break", "center_A": 1215.67,
     "color": "#ff9f1c", "lines": [("H I Ly-alpha", 1215.67)]},
    {"key": "N_IV", "title": "N IV]", "center_A": 1486.50,
     "color": "#ff4fa3", "lines": [("N IV] 1483", 1483.32), ("N IV] 1487", 1486.50)]},
    {"key": "C_IV", "title": "C IV", "center_A": 1550.00,
     "color": "#25c7f7", "lines": [("C IV 1548", 1548.20), ("C IV 1551", 1550.77)]},
    {"key": "HEII_OIII", "title": "He II + O III]", "center_A": 1650.00,
     "color": "#b978ff", "lines": [("He II 1640", 1640.42), ("O III] 1661", 1660.81), ("O III] 1666", 1666.15)]},
    {"key": "N_III", "title": "N III]", "center_A": 1750.00,
     "color": "#ef4444", "lines": [("N III] 1747", 1746.82), ("N III] 1749", 1748.65),
                                            ("N III] 1750", 1749.67), ("N III] 1752", 1752.16),
                                            ("N III] 1754", 1753.99)]},
    {"key": "C_III", "title": "C III]", "center_A": 1900.00,
     "color": "#22c55e", "lines": [("C III] 1907", 1906.68), ("C III] 1909", 1908.73)]},
]


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


def download_csv(name: str) -> Path:
    path = DATA / name
    if not path.exists() or path.stat().st_size < 500:
        urllib.request.urlretrieve(RAW_BASE + name, path)
    return path


def load_csv(name: str, role: str) -> pd.DataFrame:
    frame = pd.read_csv(download_csv(name))
    frame = frame.copy()
    frame["role"] = role
    frame["source_file"] = name
    for column in ("observed_wavelength_um", "flux_native_display_units", "flux_error_native_display_units"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[np.isfinite(frame.observed_wavelength_um) & np.isfinite(frame.flux_native_display_units)]
    return frame.sort_values("observed_wavelength_um").reset_index(drop=True)


def load_prism_data() -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    stage3 = load_csv(STAGE3_NAME, "STAGE3_COMBINED")
    exposures = [load_csv(name, "INDIVIDUAL_EXPOSURE") for name in EXPOSURE_NAMES]
    return stage3, exposures


def weighted_fit(matrix: np.ndarray, y: np.ndarray, error: np.ndarray):
    weight = 1.0 / np.maximum(error, 1e-12)
    design = matrix * weight[:, None]
    target = y * weight
    coeff, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    model = matrix @ coeff
    chi2 = float(np.sum(((y - model) / error) ** 2))
    return coeff, model, chi2


def measure_break_z(stage3: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    wave = stage3.observed_wavelength_um.to_numpy(float)
    flux = stage3.flux_native_display_units.to_numpy(float)
    error = stage3.flux_error_native_display_units.to_numpy(float)
    good_error = np.isfinite(error) & (error > 0)
    error[~good_error] = np.nanmedian(error[good_error]) if good_error.any() else 1.0
    window = (wave >= 1.70) & (wave <= 2.12)
    x, y, e = wave[window], flux[window], error[window]
    z_grid = np.linspace(13.70, 15.10, 1401)
    rows = []
    for z in z_grid:
        edge = LYA_A * (1.0 + z) * 1e-4
        best = math.inf
        for scale in np.linspace(0.45, 1.8, 12):
            sigma = scale * edge / (PRISM_R * 2.355)
            step = 0.5 * (1.0 + erf((x - edge) / (np.sqrt(2.0) * sigma)))
            dx = x - edge
            matrix = np.column_stack([np.ones_like(x), dx, step, step * dx])
            _, _, chi2 = weighted_fit(matrix, y, e)
            best = min(best, chi2)
        rows.append((z, best))
    grid = pd.DataFrame(rows, columns=["z", "chi2"])
    grid["relative_likelihood"] = np.exp(-0.5 * (grid.chi2 - grid.chi2.min()))
    best_z = float(grid.loc[grid.chi2.idxmin(), "z"])
    return best_z, grid


def search_mast_g235h() -> tuple[pd.DataFrame, list[Path], str]:
    target = SkyCoord(RA_DEG * u.deg, DEC_DEG * u.deg)
    try:
        observations = Observations.query_region(target, radius=SEARCH_RADIUS_ARCSEC * u.arcsec)
        obs_df = observations.to_pandas()
        obs_path = CSV / f"{VERSION}_MAST_REGION_OBSERVATIONS.csv"
        obs_df.to_csv(obs_path, index=False)
        if len(observations) == 0:
            return pd.DataFrame(), [], "NO_OBSERVATIONS"
        products = Observations.get_product_list(observations)
        prod_df = products.to_pandas()
        prod_df.to_csv(CSV / f"{VERSION}_MAST_ALL_PRODUCTS.csv", index=False)
        searchable = prod_df.astype(str).agg(" ".join, axis=1).str.lower()
        mode_mask = searchable.str.contains("g235h") | (
            searchable.str.contains("f170lp") & searchable.str.contains("g235")
        )
        type_mask = searchable.str.contains("x1d") | searchable.str.contains("s2d")
        candidates = prod_df[mode_mask & type_mask].copy()
        candidates.to_csv(CSV / f"{VERSION}_MAST_G235H_CANDIDATES.csv", index=False)
        if candidates.empty:
            return candidates, [], "NO_PUBLIC_G235H"
        product_names = set(candidates.get("productFilename", pd.Series(dtype=str)).astype(str))
        original_names = np.asarray(products["productFilename"]).astype(str)
        selected = products[np.isin(original_names, list(product_names))]
        manifest = Observations.download_products(selected, download_dir=str(FITS_DIR), cache=True)
        manifest_df = manifest.to_pandas()
        manifest_df.to_csv(CSV / f"{VERSION}_MAST_G235H_DOWNLOAD_MANIFEST.csv", index=False)
        paths = []
        for value in manifest_df.get("Local Path", pd.Series(dtype=str)).dropna().astype(str):
            path = Path(value)
            if path.exists() and path.suffix.lower() == ".fits":
                paths.append(path)
        return candidates, paths, "PUBLIC_G235H_FOUND"
    except Exception as exc:
        pd.DataFrame([{"error_type": type(exc).__name__, "message": str(exc)}]).to_csv(
            CSV / f"{VERSION}_MAST_G235H_ERROR.csv", index=False
        )
        return pd.DataFrame(), [], f"MAST_ERROR_{type(exc).__name__}"


def extract_x1d(path: Path) -> list[pd.DataFrame]:
    outputs = []
    with fits.open(path, memmap=False) as hdul:
        for hdu_index, hdu in enumerate(hdul[1:], 1):
            data = hdu.data
            names = [name.upper() for name in getattr(data, "names", []) or []]
            if "WAVELENGTH" not in names or "FLUX" not in names:
                continue
            wave = np.asarray(data["WAVELENGTH"], dtype=float).ravel()
            flux = np.asarray(data["FLUX"], dtype=float).ravel()
            err_name = "FLUX_ERROR" if "FLUX_ERROR" in names else "ERROR" if "ERROR" in names else None
            error = np.asarray(data[err_name], dtype=float).ravel() if err_name else np.full_like(flux, np.nan)
            finite = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
            if finite.sum() < 10:
                continue
            frame = pd.DataFrame({
                "observed_wavelength_um": wave[finite],
                "flux": flux[finite],
                "error": error[finite],
                "product": path.name,
                "hdu": hdu_index,
            }).sort_values("observed_wavelength_um")
            outputs.append(frame)
    return outputs


def robust_ylim(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    low, high = np.nanpercentile(finite, [1, 99])
    if not np.isfinite(low + high) or high <= low:
        center = float(np.nanmedian(finite))
        spread = float(np.nanstd(finite)) or 1.0
        return center - 3 * spread, center + 3 * spread
    margin = 0.15 * (high - low)
    return float(low - margin), float(high + margin)


def plot_prism_regions(stage3: pd.DataFrame, exposures: list[pd.DataFrame], best_z: float):
    outputs = []
    count_rows = []
    fig_atlas, axes = plt.subplots(3, 2, figsize=(19, 20), constrained_layout=True)
    for index, (group, atlas_ax) in enumerate(zip(GROUPS, axes.ravel()), 1):
        low, high = group["center_A"] - 50.0, group["center_A"] + 50.0
        all_y = []
        total_samples = 0
        wavelength_values = []
        for exp_index, frame in enumerate(exposures, 1):
            rest = frame.observed_wavelength_um.to_numpy(float) * 1e4 / (1.0 + best_z)
            mask = (rest >= low) & (rest <= high)
            x = rest[mask]
            y = frame.flux_native_display_units.to_numpy(float)[mask]
            if len(x) == 0:
                continue
            total_samples += len(x)
            wavelength_values.extend(x.tolist())
            all_y.extend(y.tolist())
            atlas_ax.plot(x, y, lw=0.38, alpha=0.24)
            atlas_ax.scatter(x, y, s=7, alpha=0.28)
        rest3 = stage3.observed_wavelength_um.to_numpy(float) * 1e4 / (1.0 + best_z)
        mask3 = (rest3 >= low) & (rest3 <= high)
        x3 = rest3[mask3]
        y3 = stage3.flux_native_display_units.to_numpy(float)[mask3]
        e3 = stage3.flux_error_native_display_units.to_numpy(float)[mask3]
        all_y.extend(y3.tolist())
        wavelength_values.extend(x3.tolist())
        unique_bins = len(np.unique(np.round(wavelength_values, 4)))
        independent = PRISM_R * 100.0 / group["center_A"]

        fig, ax = plt.subplots(figsize=(17, 8.5), constrained_layout=True)
        for exp_index, frame in enumerate(exposures, 1):
            rest = frame.observed_wavelength_um.to_numpy(float) * 1e4 / (1.0 + best_z)
            mask = (rest >= low) & (rest <= high)
            x = rest[mask]
            y = frame.flux_native_display_units.to_numpy(float)[mask]
            if len(x):
                ax.plot(x, y, lw=0.42, alpha=0.23)
                ax.scatter(x, y, s=10, alpha=0.28)
        ax.errorbar(x3, y3, yerr=e3, fmt="o-", ms=4.2, lw=1.05, capsize=2.0,
                    color="#f8fbff", ecolor=group["color"], label=f"Stage-3 combined (n={len(x3)})")
        atlas_ax.errorbar(x3, y3, yerr=e3, fmt="o-", ms=3.2, lw=0.9, capsize=1.2,
                          color="#f8fbff", ecolor=group["color"])
        for label, rest_line in group["lines"]:
            ax.axvline(rest_line, color=group["color"], lw=1.0, ls="--")
            atlas_ax.axvline(rest_line, color=group["color"], lw=0.8, ls="--")
        ax.set_xlim(low, high)
        atlas_ax.set_xlim(low, high)
        if all_y:
            limits = robust_ylim(np.asarray(all_y, dtype=float))
            ax.set_ylim(*limits)
            atlas_ax.set_ylim(*limits)
        ax.grid(True, lw=0.42, alpha=0.35)
        atlas_ax.grid(True, lw=0.35, alpha=0.30)
        ax.axhline(0, lw=0.5, alpha=0.5)
        atlas_ax.axhline(0, lw=0.45, alpha=0.45)
        ax.set_xlabel(f"Rest-frame wavelength [Å], z_break={best_z:.4f}")
        ax.set_ylabel("Flux density [nJy]")
        ax.set_title(f"{TARGET} — {group['title']} | every measured PRISM sample")
        ax.text(0.012, 0.025,
                f"Individual exposures: {len(exposures)}  |  total detector measurements: {total_samples}\n"
                f"Stage-3 wavelength bins: {len(x3)}  |  unique wavelength locations: {unique_bins}\n"
                f"Independent R≈100 resolution elements across 100 Å: {independent:.1f}",
                transform=ax.transAxes, fontsize=8.6, va="bottom")
        ax.legend(loc="best", fontsize=8.5)
        top = ax.secondary_xaxis("top", functions=(
            lambda rest: rest * (1.0 + best_z) * 1e-4,
            lambda obs: obs * 1e4 / (1.0 + best_z),
        ))
        top.set_xlabel("Observed wavelength [µm]")
        atlas_ax.set_title(
            f"{group['title']} | {total_samples} raw measurements | {len(x3)} Stage-3 bins",
            fontsize=10.5,
        )
        atlas_ax.set_xlabel("Rest wavelength [Å]")
        atlas_ax.set_ylabel("Flux [nJy]")
        path = PNG / f"{VERSION}_{index:02d}_{group['key']}_ALL_PRISM_MEASUREMENTS.png"
        fig.savefig(path, dpi=420, bbox_inches="tight")
        plt.show(); plt.close(fig)
        outputs.append(path)
        count_rows.append({
            "region": group["key"], "center_A": group["center_A"],
            "individual_exposures": len(exposures),
            "total_detector_measurements": total_samples,
            "stage3_bins": len(x3), "unique_wavelength_locations": unique_bins,
            "independent_resolution_elements_R100": independent,
        })
    fig_atlas.suptitle(
        f"{TARGET} — all available PRISM measurements, no interpolation\n"
        f"Nine individual X1D exposures plus exact Stage-3 combined spectrum | z_break={best_z:.4f}",
        fontsize=17,
    )
    atlas_path = PNG / f"{VERSION}_SIX_REGION_ALL_PRISM_MEASUREMENTS_ATLAS.png"
    fig_atlas.savefig(atlas_path, dpi=360, bbox_inches="tight")
    plt.show(); plt.close(fig_atlas)
    pd.DataFrame(count_rows).to_csv(CSV / f"{VERSION}_PRISM_SAMPLE_COUNTS.csv", index=False)
    return outputs, atlas_path, pd.DataFrame(count_rows)


def plot_sampling_plan() -> tuple[Path, Path, pd.DataFrame]:
    rows = []
    for group in GROUPS:
        center = group["center_A"]
        for mode, resolution in (("PRISM/CLEAR", PRISM_R), ("G235M/F170LP", G235M_R), ("G235H/F170LP", G235H_R)):
            resels = resolution * 100.0 / center
            unique_samples = PIXELS_PER_RESEL * resels
            rows.append({
                "region": group["key"], "center_A": center, "mode": mode, "R": resolution,
                "independent_resolution_elements_per_100A": resels,
                "unique_detector_samples_per_100A": unique_samples,
                "three_nod_total_detector_measurements": 3.0 * unique_samples,
            })
    table = pd.DataFrame(rows)
    csv_path = CSV / f"{VERSION}_SAMPLING_PLAN.csv"
    table.to_csv(csv_path, index=False)

    pivot = table.pivot(index="region", columns="mode", values="unique_detector_samples_per_100A")
    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
    x = np.arange(len(pivot.index))
    width = 0.24
    for offset, mode in zip((-width, 0.0, width), ("PRISM/CLEAR", "G235M/F170LP", "G235H/F170LP")):
        ax.bar(x + offset, pivot[mode].to_numpy(float), width=width, label=mode)
    ax.set_xticks(x, pivot.index)
    ax.set_ylabel("Unique detector samples per 100 Å rest-frame interval")
    ax.set_title("MoM-z14 spectral sampling capacity — instrument modes")
    ax.grid(True, axis="y", lw=0.45, alpha=0.35)
    ax.legend()
    ax.text(0.012, 0.975,
            "Computed sampling only; not observed flux. G235H requires a real follow-up observation.",
            transform=ax.transAxes, va="top", fontsize=9)
    bar_path = PNG / f"{VERSION}_MODE_SAMPLE_COUNT_COMPARISON.png"
    fig.savefig(bar_path, dpi=420, bbox_inches="tight")
    plt.show(); plt.close(fig)

    center = 1750.0
    low, high = center - 50.0, center + 50.0
    prism_n = max(2, int(round(PIXELS_PER_RESEL * PRISM_R * 100.0 / center)))
    g235h_n = max(2, int(round(PIXELS_PER_RESEL * G235H_R * 100.0 / center)))
    prism_grid = np.linspace(low, high, prism_n)
    g235h_grid = np.linspace(low, high, g235h_n)
    fig, ax = plt.subplots(figsize=(18, 5.8), constrained_layout=True)
    ax.eventplot([prism_grid, g235h_grid], lineoffsets=[1.0, 0.0], linelengths=[0.65, 0.65], linewidths=[1.1, 0.65])
    ax.set_yticks([1.0, 0.0], [f"PRISM: {prism_n} unique samples", f"G235H: {g235h_n} unique samples"])
    ax.set_xlim(low, high)
    ax.set_xlabel("Rest-frame wavelength [Å] around N III] 1750")
    ax.set_title(
        f"Sampling geometry for one 100 Å interval\n"
        f"Three G235H nod exposures ≈ {3 * g235h_n:,} total detector measurements, but {g235h_n:,} distinct wavelength positions"
    )
    ax.grid(True, axis="x", lw=0.4, alpha=0.3)
    ax.text(0.012, 0.06, "No flux values are synthesized in this panel.", transform=ax.transAxes, fontsize=9)
    grid_path = PNG / f"{VERSION}_PRISM_VS_G235H_SAMPLE_GRID.png"
    fig.savefig(grid_path, dpi=420, bbox_inches="tight")
    plt.show(); plt.close(fig)
    return bar_path, grid_path, table


def plot_real_g235h(paths: list[Path]) -> list[Path]:
    outputs = []
    rank = 0
    for path in paths:
        for frame in extract_x1d(path):
            rank += 1
            fig, ax = plt.subplots(figsize=(18, 8), constrained_layout=True)
            wave = frame.observed_wavelength_um.to_numpy(float)
            flux = frame.flux.to_numpy(float)
            error = frame.error.to_numpy(float)
            good = np.isfinite(error) & (error >= 0)
            if good.any():
                ax.fill_between(wave[good], flux[good] - error[good], flux[good] + error[good], alpha=0.13, linewidth=0)
            ax.plot(wave, flux, lw=0.58, label=f"Native G235H X1D samples: {len(wave):,}")
            ax.set_xlabel("Observed wavelength [µm]")
            ax.set_ylabel("Flux [native FITS units]")
            ax.set_title(f"{TARGET} — real public G235H/F170LP X1D\n{path.name} | HDU {int(frame.hdu.iloc[0])}")
            ax.grid(True, lw=0.42, alpha=0.35)
            ax.legend()
            output = PNG / f"{VERSION}_G235H_REAL_{rank:02d}_{path.stem}.png"
            fig.savefig(output, dpi=420, bbox_inches="tight")
            plt.show(); plt.close(fig)
            frame.to_csv(CSV / f"{VERSION}_G235H_REAL_{rank:02d}_{path.stem}.csv", index=False)
            outputs.append(output)
    return outputs


def plot_g235h_status(status: str, candidate_count: int) -> Path:
    fig, ax = plt.subplots(figsize=(15, 6.5), constrained_layout=True)
    ax.axis("off")
    title = "PUBLIC G235H DATA FOUND" if status == "PUBLIC_G235H_FOUND" else "NO PUBLIC G235H MOM-z14 X1D FOUND"
    lines = [
        title,
        f"MAST status: {status}",
        f"Exact search position: RA={RA_DEG:.7f} deg, Dec={DEC_DEG:.7f} deg",
        f"Search radius: {SEARCH_RADIUS_ARCSEC:.2f} arcsec",
        f"G235H candidate products: {candidate_count}",
        "PRISM processing below uses every real uploaded exposure and the Stage-3 combined X1D.",
        "A G235H sample-density plot is observation planning only; it is not synthetic flux.",
    ]
    ax.text(0.5, 0.72, lines[0], ha="center", va="center", fontsize=22, weight="bold")
    ax.text(0.5, 0.42, "\n".join(lines[1:]), ha="center", va="center", fontsize=12, linespacing=1.8)
    path = PNG / f"{VERSION}_G235H_PUBLIC_DATA_STATUS.png"
    fig.savefig(path, dpi=360, bbox_inches="tight")
    plt.show(); plt.close(fig)
    return path


def main() -> None:
    style()
    stage3, exposures = load_prism_data()
    best_z, redshift_grid = measure_break_z(stage3)
    redshift_path = CSV / f"{VERSION}_LYMAN_BREAK_REDSHIFT_GRID.csv"
    redshift_grid.to_csv(redshift_path, index=False)

    region_paths, atlas_path, counts = plot_prism_regions(stage3, exposures, best_z)
    sample_bar, sample_grid, plan_table = plot_sampling_plan()
    candidates, g235h_paths, mast_status = search_mast_g235h()
    real_g235h_plots = plot_real_g235h(g235h_paths) if g235h_paths else []
    status_path = plot_g235h_status(mast_status, len(candidates))

    n_iii = plan_table[(plan_table.region == "N_III") & (plan_table["mode"] == "G235H/F170LP")].iloc[0]
    print(f"CODE OUTPUT: {VERSION}")
    print(f"TARGET              {TARGET}")
    print(f"SOURCE ID           {SOURCE_ID}")
    print(f"PRISM EXPOSURES     {len(exposures)} individual + 1 Stage-3 combined")
    print(f"BREAK z             {best_z:.5f}")
    print(f"PUBLISHED z         {PUBLISHED_Z:.5f} comparison only")
    print(f"G235H MAST STATUS   {mast_status}")
    print(f"G235H PRODUCTS      {len(candidates)}")
    print(f"NIII G235H UNIQUE   {n_iii.unique_detector_samples_per_100A:.1f} samples per 100 Å")
    print(f"NIII 3-NOD TOTAL    {n_iii.three_nod_total_detector_measurements:.1f} detector measurements")
    print("SCIENCE STATUS      existing PRISM observed; G235H planning theoretical unless public FITS found")
    print(f"PRISM ATLAS PNG     {atlas_path}")
    print(f"SAMPLE BAR PNG      {sample_bar}")
    print(f"SAMPLE GRID PNG     {sample_grid}")
    print(f"G235H STATUS PNG    {status_path}")
    for path in real_g235h_plots:
        print(f"REAL G235H PNG      {path}")
    for path in region_paths:
        print(f"REGION PNG          {path}")
    print(f"COUNTS CSV          {CSV / f'{VERSION}_PRISM_SAMPLE_COUNTS.csv'}")
    print(f"PLAN CSV            {CSV / f'{VERSION}_SAMPLING_PLAN.csv'}")
    print(f"REDSHIFT CSV        {redshift_path}")
    print(f"Timestamp           {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
