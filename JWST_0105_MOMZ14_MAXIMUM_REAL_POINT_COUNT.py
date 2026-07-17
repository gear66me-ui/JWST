#!/usr/bin/env python3
"""
JWST_0105_MOMZ14_MAXIMUM_REAL_POINT_COUNT.py

Corrects JWST_0104 so the plots do not visually collapse to only the ten
Stage-3 bins.  The script:
  1. searches MAST live for an actual public MoM-z14 G235H/F170LP X1D;
  2. plots it at native sampling if one exists;
  3. otherwise plots every real uploaded PRISM/CLEAR exposure measurement
     (nine exposure X1Ds plus the Stage-3 combined X1D) in six rest-UV windows;
  4. reports total measurement instances separately from unique wavelength
     locations and independent resolution elements.

No flux is synthesized.  No interpolation is presented as measured data.
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
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from astroquery.mast import Observations

VERSION = "JWST_0105"
TARGET = "MoM-z14"
RA_DEG = 150.0933255
DEC_DEG = 2.2731627
SOURCE_ID = 277193
Z = 14.44
PRISM_R = 100.0
G235H_R = 2700.0
SEARCH_RADIUS_ARCSEC = 0.35

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

REGIONS = [
    {"key": "LYMAN_ALPHA", "title": "H I Lyman-alpha break", "center": 1215.67,
     "color": "#ff9f1c", "lines": [("H I Ly-alpha", 1215.67)]},
    {"key": "N_IV", "title": "N IV]", "center": 1486.50,
     "color": "#ff4fa3", "lines": [("N IV] 1483", 1483.32), ("N IV] 1487", 1486.50)]},
    {"key": "C_IV", "title": "C IV", "center": 1550.00,
     "color": "#25c7f7", "lines": [("C IV 1548", 1548.20), ("C IV 1551", 1550.77)]},
    {"key": "HEII_OIII", "title": "He II + O III]", "center": 1650.00,
     "color": "#b978ff", "lines": [("He II 1640", 1640.42),
                                      ("O III] 1661", 1660.81),
                                      ("O III] 1666", 1666.15)]},
    {"key": "N_III", "title": "N III]", "center": 1750.00,
     "color": "#ef4444", "lines": [("N III] 1747", 1746.82),
                                      ("N III] 1749", 1748.65),
                                      ("N III] 1750", 1749.67),
                                      ("N III] 1752", 1752.16),
                                      ("N III] 1754", 1753.99)]},
    {"key": "C_III", "title": "C III]", "center": 1900.00,
     "color": "#22c55e", "lines": [("C III] 1907", 1906.68),
                                      ("C III] 1909", 1908.73)]},
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


def download_csv(name: str) -> Path:
    path = DATA / name
    if not path.exists() or path.stat().st_size < 500:
        urllib.request.urlretrieve(RAW_BASE + name, path)
    return path


def load_csv(name: str, role: str, exposure_index: int) -> pd.DataFrame:
    frame = pd.read_csv(download_csv(name)).copy()
    frame["role"] = role
    frame["exposure_index"] = exposure_index
    frame["source_file"] = name
    for column in ("observed_wavelength_um", "flux_native_display_units",
                   "flux_error_native_display_units"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    keep = np.isfinite(frame["observed_wavelength_um"]) & np.isfinite(frame["flux_native_display_units"])
    return frame.loc[keep].sort_values("observed_wavelength_um").reset_index(drop=True)


def load_all_prism() -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    stage3 = load_csv(STAGE3_NAME, "STAGE3_COMBINED", 0)
    exposures = [load_csv(name, "INDIVIDUAL_EXPOSURE", idx)
                 for idx, name in enumerate(EXPOSURE_NAMES, 1)]
    return stage3, exposures


def robust_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    low, high = np.nanpercentile(finite, [1.0, 99.0])
    if not np.isfinite(low + high) or high <= low:
        center = float(np.nanmedian(finite))
        spread = float(np.nanstd(finite)) or 1.0
        return center - 3.0 * spread, center + 3.0 * spread
    margin = 0.12 * (high - low)
    return float(low - margin), float(high + margin)


def prism_region_data(region: dict, stage3: pd.DataFrame,
                      exposures: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    low, high = region["center"] - 50.0, region["center"] + 50.0
    exposure_parts = []
    for frame in exposures:
        rest = frame["observed_wavelength_um"].to_numpy(float) * 1e4 / (1.0 + Z)
        mask = (rest >= low) & (rest <= high)
        part = frame.loc[mask].copy()
        part["rest_wavelength_A"] = rest[mask]
        exposure_parts.append(part)
    all_exp = pd.concat(exposure_parts, ignore_index=True) if exposure_parts else pd.DataFrame()
    rest3 = stage3["observed_wavelength_um"].to_numpy(float) * 1e4 / (1.0 + Z)
    mask3 = (rest3 >= low) & (rest3 <= high)
    combined = stage3.loc[mask3].copy()
    combined["rest_wavelength_A"] = rest3[mask3]
    return all_exp, combined


def plot_prism_maximum(stage3: pd.DataFrame, exposures: list[pd.DataFrame]):
    outputs = []
    count_rows = []
    atlas, axes = plt.subplots(3, 2, figsize=(20, 20), constrained_layout=True)
    marker_cycle = ["o", "s", "^", "v", "D", "P", "X", "<", ">"]

    for index, (region, atlas_ax) in enumerate(zip(REGIONS, axes.ravel()), 1):
        low, high = region["center"] - 50.0, region["center"] + 50.0
        all_exp, combined = prism_region_data(region, stage3, exposures)
        total_raw = len(all_exp)
        stage3_n = len(combined)
        unique_raw = len(np.unique(np.round(all_exp.get("rest_wavelength_A", pd.Series(dtype=float)), 5)))
        total_with_stage3 = total_raw + stage3_n
        independent = PRISM_R * 100.0 / region["center"]

        fig, ax = plt.subplots(figsize=(18, 9), constrained_layout=True)
        y_for_limits = []
        for exp_index in range(1, len(exposures) + 1):
            part = all_exp[all_exp["exposure_index"] == exp_index]
            if part.empty:
                continue
            x = part["rest_wavelength_A"].to_numpy(float)
            y = part["flux_native_display_units"].to_numpy(float)
            y_for_limits.extend(y.tolist())
            marker = marker_cycle[(exp_index - 1) % len(marker_cycle)]
            ax.scatter(x, y, s=28, marker=marker, alpha=0.78, linewidths=0.35,
                       label=f"Exposure {exp_index}: n={len(part)}")
            atlas_ax.scatter(x, y, s=13, marker=marker, alpha=0.58, linewidths=0.2)

        x3 = combined["rest_wavelength_A"].to_numpy(float)
        y3 = combined["flux_native_display_units"].to_numpy(float)
        e3 = combined["flux_error_native_display_units"].to_numpy(float)
        y_for_limits.extend(y3.tolist())
        ax.errorbar(x3, y3, yerr=e3, fmt="o-", ms=5.2, lw=1.45, capsize=2.4,
                    color="#ffffff", ecolor=region["color"], zorder=8,
                    label=f"Stage-3 combined bins: n={stage3_n}")
        atlas_ax.errorbar(x3, y3, yerr=e3, fmt="o-", ms=3.6, lw=1.0, capsize=1.3,
                          color="#ffffff", ecolor=region["color"], zorder=8)

        for label, rest_line in region["lines"]:
            ax.axvline(rest_line, color=region["color"], lw=1.15, ls="--", alpha=0.92)
            atlas_ax.axvline(rest_line, color=region["color"], lw=0.85, ls="--", alpha=0.88)

        if y_for_limits:
            limits = robust_limits(np.asarray(y_for_limits, dtype=float))
            ax.set_ylim(*limits)
            atlas_ax.set_ylim(*limits)
        ax.set_xlim(low, high)
        atlas_ax.set_xlim(low, high)
        ax.grid(True, lw=0.42, alpha=0.34)
        atlas_ax.grid(True, lw=0.34, alpha=0.30)
        ax.axhline(0.0, lw=0.55, alpha=0.55)
        atlas_ax.axhline(0.0, lw=0.45, alpha=0.50)
        ax.set_xlabel(f"Rest-frame wavelength [Å], using z={Z:.2f}")
        ax.set_ylabel("Flux density [native X1D units]")
        ax.set_title(
            f"{TARGET} — {region['title']} | maximum real PRISM measurement count\n"
            f"{total_raw} exposure measurements + {stage3_n} combined bins = {total_with_stage3} plotted measurements"
        )
        ax.text(
            0.012, 0.022,
            f"Distinct wavelength locations in individual exposures: {unique_raw}\n"
            f"Independent R≈100 resolution elements across this 100 Å window: {independent:.2f}\n"
            "Every marker is a real X1D measurement. Repeated wavelengths are separate exposures, not extra resolution.",
            transform=ax.transAxes, fontsize=8.9, va="bottom",
            bbox={"facecolor": "#050812", "edgecolor": "#64748b", "alpha": 0.86, "pad": 5},
        )
        ax.legend(loc="upper right", ncol=2, fontsize=7.8, framealpha=0.90)
        top = ax.secondary_xaxis("top", functions=(
            lambda rest: rest * (1.0 + Z) * 1e-4,
            lambda obs: obs * 1e4 / (1.0 + Z),
        ))
        top.set_xlabel("Observed wavelength [µm]")

        atlas_ax.set_title(
            f"{region['title']} | {total_with_stage3} real measurements | {stage3_n} unique Stage-3 bins",
            fontsize=10.5,
        )
        atlas_ax.set_xlabel("Rest wavelength [Å]")
        atlas_ax.set_ylabel("Flux")

        path = PNG / f"{VERSION}_{index:02d}_{region['key']}_MAX_REAL_PRISM_POINTS.png"
        fig.savefig(path, dpi=420, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        outputs.append(path)

        all_out = all_exp.copy()
        all_out["region"] = region["key"]
        combined_out = combined.copy()
        combined_out["region"] = region["key"]
        pd.concat([all_out, combined_out], ignore_index=True).to_csv(
            CSV / f"{VERSION}_{index:02d}_{region['key']}_ALL_REAL_MEASUREMENTS.csv", index=False
        )
        count_rows.append({
            "region": region["key"],
            "center_A": region["center"],
            "individual_exposure_measurements": total_raw,
            "stage3_combined_bins": stage3_n,
            "total_plotted_measurements": total_with_stage3,
            "distinct_individual_exposure_wavelengths": unique_raw,
            "independent_resolution_elements_R100": independent,
        })

    atlas.suptitle(
        f"{TARGET} — maximum real PRISM point count\n"
        "Nine individual exposure X1Ds plus exact Stage-3 combined X1D; no synthetic flux and no interpolated measurements",
        fontsize=17,
    )
    atlas_path = PNG / f"{VERSION}_SIX_REGION_MAX_REAL_PRISM_POINTS_ATLAS.png"
    atlas.savefig(atlas_path, dpi=360, bbox_inches="tight")
    plt.show()
    plt.close(atlas)
    counts = pd.DataFrame(count_rows)
    counts_path = CSV / f"{VERSION}_REAL_POINT_COUNTS.csv"
    counts.to_csv(counts_path, index=False)
    return outputs, atlas_path, counts_path, counts


def search_real_g235h() -> tuple[str, pd.DataFrame, list[Path]]:
    target = SkyCoord(RA_DEG * u.deg, DEC_DEG * u.deg)
    try:
        observations = Observations.query_region(target, radius=SEARCH_RADIUS_ARCSEC * u.arcsec)
        obs_df = observations.to_pandas()
        obs_df.to_csv(CSV / f"{VERSION}_MAST_OBSERVATIONS.csv", index=False)
        if len(observations) == 0:
            return "NO_OBSERVATIONS", pd.DataFrame(), []
        products = Observations.get_product_list(observations)
        prod_df = products.to_pandas()
        prod_df.to_csv(CSV / f"{VERSION}_MAST_ALL_PRODUCTS.csv", index=False)
        text = prod_df.astype(str).agg(" ".join, axis=1).str.lower()
        mode = text.str.contains("g235h") & text.str.contains("f170lp")
        x1d = text.str.contains("x1d")
        candidates = prod_df[mode & x1d].copy()
        candidates.to_csv(CSV / f"{VERSION}_MAST_G235H_X1D_CANDIDATES.csv", index=False)
        if candidates.empty:
            return "NO_PUBLIC_G235H_X1D", candidates, []
        names = set(candidates.get("productFilename", pd.Series(dtype=str)).astype(str))
        product_names = np.asarray(products["productFilename"]).astype(str)
        selected = products[np.isin(product_names, list(names))]
        manifest = Observations.download_products(selected, download_dir=str(FITS_DIR), cache=True)
        manifest_df = manifest.to_pandas()
        manifest_df.to_csv(CSV / f"{VERSION}_MAST_G235H_DOWNLOAD_MANIFEST.csv", index=False)
        paths = []
        for value in manifest_df.get("Local Path", pd.Series(dtype=str)).dropna().astype(str):
            path = Path(value)
            if path.exists() and path.suffix.lower() == ".fits":
                paths.append(path)
        return "PUBLIC_G235H_X1D_FOUND", candidates, paths
    except Exception as exc:
        pd.DataFrame([{"error_type": type(exc).__name__, "message": str(exc)}]).to_csv(
            CSV / f"{VERSION}_MAST_G235H_ERROR.csv", index=False
        )
        return f"MAST_ERROR_{type(exc).__name__}", pd.DataFrame(), []


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
            if finite.sum() < 20:
                continue
            outputs.append(pd.DataFrame({
                "observed_wavelength_um": wave[finite],
                "flux": flux[finite],
                "error": error[finite],
                "product": path.name,
                "hdu": hdu_index,
            }).sort_values("observed_wavelength_um"))
    return outputs


def plot_real_g235h(paths: list[Path]) -> list[Path]:
    outputs = []
    rank = 0
    for path in paths:
        for frame in extract_x1d(path):
            rank += 1
            rest = frame["observed_wavelength_um"].to_numpy(float) * 1e4 / (1.0 + Z)
            for region_index, region in enumerate(REGIONS, 1):
                low, high = region["center"] - 50.0, region["center"] + 50.0
                mask = (rest >= low) & (rest <= high)
                if mask.sum() < 3:
                    continue
                x = rest[mask]
                y = frame["flux"].to_numpy(float)[mask]
                e = frame["error"].to_numpy(float)[mask]
                fig, ax = plt.subplots(figsize=(18, 8.8), constrained_layout=True)
                good = np.isfinite(e) & (e >= 0)
                if good.any():
                    ax.fill_between(x[good], y[good] - e[good], y[good] + e[good], alpha=0.14, linewidth=0)
                ax.plot(x, y, lw=0.72, marker=".", ms=3.2,
                        label=f"Real native G235H samples: n={mask.sum()}")
                for label, rest_line in region["lines"]:
                    ax.axvline(rest_line, color=region["color"], lw=1.0, ls="--")
                ax.set_xlim(low, high)
                ax.set_xlabel(f"Rest-frame wavelength [Å], using z={Z:.2f}")
                ax.set_ylabel("Flux [native FITS units]")
                ax.set_title(f"{TARGET} — REAL G235H/F170LP — {region['title']}\n{path.name} | HDU {int(frame.hdu.iloc[0])}")
                ax.grid(True, lw=0.42, alpha=0.34)
                ax.legend()
                output = PNG / f"{VERSION}_G235H_REAL_{rank:02d}_{region_index:02d}_{region['key']}.png"
                fig.savefig(output, dpi=420, bbox_inches="tight")
                plt.show()
                plt.close(fig)
                pd.DataFrame({
                    "rest_wavelength_A": x,
                    "observed_wavelength_um": frame["observed_wavelength_um"].to_numpy(float)[mask],
                    "flux": y,
                    "error": e,
                }).to_csv(CSV / f"{VERSION}_G235H_REAL_{rank:02d}_{region_index:02d}_{region['key']}.csv", index=False)
                outputs.append(output)
    return outputs


def make_status_plot(status: str, candidate_count: int, counts: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)
    ax.axis("off")
    max_real = int(counts["total_plotted_measurements"].max()) if len(counts) else 0
    min_real = int(counts["total_plotted_measurements"].min()) if len(counts) else 0
    title = "REAL PUBLIC G235H DATA FOUND" if status == "PUBLIC_G235H_X1D_FOUND" else "NO REAL PUBLIC G235H X1D FOUND"
    body = [
        title,
        f"MAST status: {status}",
        f"G235H/F170LP candidate X1D products: {candidate_count}",
        f"Current PRISM plots now show {min_real}–{max_real} real measurement instances per 100 Å window, not only ten Stage-3 bins.",
        "Repeated exposure measurements increase point count and statistical information, but not independent wavelength resolution.",
        "A genuine ~300-point unique-wavelength spectrum requires an actual G235H observation.",
    ]
    ax.text(0.5, 0.75, body[0], ha="center", va="center", fontsize=22, weight="bold")
    ax.text(0.5, 0.43, "\n".join(body[1:]), ha="center", va="center", fontsize=12.3, linespacing=1.8)
    path = PNG / f"{VERSION}_G235H_AND_PRISM_POINT_STATUS.png"
    fig.savefig(path, dpi=360, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return path


def main() -> None:
    set_style()
    stage3, exposures = load_all_prism()
    prism_paths, atlas_path, counts_path, counts = plot_prism_maximum(stage3, exposures)
    status, candidates, g235h_paths = search_real_g235h()
    real_g235h_outputs = plot_real_g235h(g235h_paths) if g235h_paths else []
    status_path = make_status_plot(status, len(candidates), counts)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"TARGET          {TARGET}")
    print(f"PRISM INPUT     9 individual exposure X1Ds + 1 Stage-3 combined X1D")
    print(f"PRISM COUNTS    {counts_path}")
    print(f"PRISM ATLAS     {atlas_path}")
    print(f"G235H STATUS    {status}")
    print(f"G235H PRODUCTS  {len(candidates)}")
    print(f"G235H PLOTS     {len(real_g235h_outputs)}")
    print(f"STATUS PNG      {status_path}")
    for path in prism_paths:
        print(f"REGION PNG      {path}")
    for path in real_g235h_outputs:
        print(f"G235H PNG       {path}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
