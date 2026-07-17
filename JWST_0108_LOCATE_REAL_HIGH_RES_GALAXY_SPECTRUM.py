#!/usr/bin/env python3
"""
JWST_0108_LOCATE_REAL_HIGH_RES_GALAXY_SPECTRUM.py

Locate and download a real public JWST/NIRSpec high-resolution galaxy spectrum.
Target: SPURS-A2744-7 / Gz9p3 / DHZ1, z=9.31102, observed by GLASS ERS-1324.
Accepted modes only: G140H/F100LP, G235H/F170LP, G395H/F290LP (R~2700).
PRISM and medium-resolution products are rejected. No interpolation, no synthetic
flux, no model spectrum, and no AI-generated images.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
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

VERSION = "JWST_0108"
TARGET = "SPURS-A2744-7 / Gz9p3 / DHZ1"
RA_DEG = 3.6171694
DEC_DEG = -30.4255494
Z = 9.31102
SEARCH_RADIUS = 2.0 * u.arcsec
HIGH_RES_MODES = {
    "G140H": "F100LP",
    "G235H": "F170LP",
    "G395H": "F290LP",
}

ROOT = Path("/content/JWST_OUTPUT")
PNG = ROOT / "PNG"
CSV = ROOT / "CSV"
DATA = ROOT / "DATA" / VERSION
FITS_DIR = DATA / "FITS"
for directory in (PNG, CSV, DATA, FITS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

REGIONS = [
    {"key": "LYMAN_ALPHA", "title": "H I Lyman-alpha", "center_A": 1215.67,
     "lines": [("H I Ly-alpha", 1215.67)]},
    {"key": "N_IV", "title": "N IV]", "center_A": 1486.50,
     "lines": [("N IV] 1483", 1483.32), ("N IV] 1487", 1486.50)]},
    {"key": "C_IV", "title": "C IV", "center_A": 1550.00,
     "lines": [("C IV 1548", 1548.20), ("C IV 1551", 1550.77)]},
    {"key": "HEII_OIII", "title": "He II + O III]", "center_A": 1650.00,
     "lines": [("He II 1640", 1640.42), ("O III] 1661", 1660.81),
               ("O III] 1666", 1666.15)]},
    {"key": "N_III", "title": "N III]", "center_A": 1750.00,
     "lines": [("N III] 1747", 1746.82), ("N III] 1749", 1748.65),
               ("N III] 1750", 1749.67), ("N III] 1752", 1752.16),
               ("N III] 1754", 1753.99)]},
    {"key": "C_III", "title": "C III]", "center_A": 1900.00,
     "lines": [("C III] 1907", 1906.68), ("C III] 1909", 1908.73)]},
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


def text_columns(frame: pd.DataFrame) -> pd.Series:
    return frame.astype(str).agg(" ".join, axis=1).str.lower()


def query_mast() -> tuple[pd.DataFrame, pd.DataFrame]:
    Observations.disable_cloud_dataset()
    coord = SkyCoord(RA_DEG * u.deg, DEC_DEG * u.deg)
    observations = Observations.query_region(coord, radius=SEARCH_RADIUS)
    if len(observations) == 0:
        raise RuntimeError("MAST returned no observations at the verified target position.")

    obs_df = observations.to_pandas()
    obs_df.to_csv(CSV / f"{VERSION}_MAST_ALL_OBSERVATIONS.csv", index=False)
    obs_text = text_columns(obs_df)
    keep = obs_text.str.contains("nirspec") & (
        obs_text.str.contains("g140h") |
        obs_text.str.contains("g235h") |
        obs_text.str.contains("g395h") |
        obs_text.str.contains("glass-jwst") |
        obs_text.str.contains("1324")
    )
    high_obs_df = obs_df[keep].copy()
    high_obs_df.to_csv(CSV / f"{VERSION}_MAST_HIGH_RES_OBSERVATIONS.csv", index=False)

    if high_obs_df.empty:
        raise RuntimeError(
            "No high-resolution NIRSpec observation rows were found at the target position. "
            "The script will not fall back to PRISM."
        )

    obs_ids = set(high_obs_df["obsid"].astype(str)) if "obsid" in high_obs_df.columns else set()
    if not obs_ids:
        raise RuntimeError("High-resolution observation rows do not expose obsid values.")

    selected_obs = observations[np.isin(np.asarray(observations["obsid"]).astype(str), list(obs_ids))]
    products = Observations.get_product_list(selected_obs)
    product_df = products.to_pandas()
    product_df.to_csv(CSV / f"{VERSION}_MAST_ALL_HIGH_RES_PRODUCTS.csv", index=False)

    ptext = text_columns(product_df)
    fits_mask = ptext.str.contains("fits")
    spectrum_mask = (
        ptext.str.contains("x1d") |
        ptext.str.contains("spec.fits") |
        ptext.str.contains("s2d")
    )
    high_mask = (
        ptext.str.contains("g140h") |
        ptext.str.contains("g235h") |
        ptext.str.contains("g395h") |
        ptext.str.contains("glass-jwst")
    )
    reject_mask = ptext.str.contains("prism") | ptext.str.contains("g140m") | ptext.str.contains("g235m") | ptext.str.contains("g395m")
    candidates = product_df[fits_mask & spectrum_mask & high_mask & ~reject_mask].copy()
    candidates.to_csv(CSV / f"{VERSION}_MAST_HIGH_RES_CANDIDATE_PRODUCTS.csv", index=False)
    if candidates.empty:
        raise RuntimeError(
            "MAST observations exist, but no downloadable high-resolution X1D/SPEC/S2D FITS products were found. "
            "The script refuses to substitute PRISM data."
        )
    return high_obs_df, candidates


def download_candidates(candidates: pd.DataFrame) -> tuple[pd.DataFrame, list[Path]]:
    product_names = set(candidates.get("productFilename", pd.Series(dtype=str)).astype(str))
    if not product_names:
        raise RuntimeError("Candidate product table lacks productFilename values.")

    coord = SkyCoord(RA_DEG * u.deg, DEC_DEG * u.deg)
    observations = Observations.query_region(coord, radius=SEARCH_RADIUS)
    products = Observations.get_product_list(observations)
    original_names = np.asarray(products["productFilename"]).astype(str)
    selected = products[np.isin(original_names, list(product_names))]
    manifest = Observations.download_products(
        selected,
        download_dir=str(FITS_DIR),
        cache=True,
        mrp_only=False,
    )
    manifest_df = manifest.to_pandas()
    manifest_df.to_csv(CSV / f"{VERSION}_DOWNLOAD_MANIFEST.csv", index=False)

    paths = []
    for value in manifest_df.get("Local Path", pd.Series(dtype=str)).dropna().astype(str):
        path = Path(value)
        if path.exists() and path.suffix.lower() == ".fits":
            paths.append(path)
    if not paths:
        raise RuntimeError("MAST download completed without any local FITS files.")
    return manifest_df, paths


def header_value(headers: list[fits.Header], keys: list[str]) -> str:
    for header in headers:
        for key in keys:
            value = header.get(key)
            if value not in (None, "", "N/A", "UNKNOWN"):
                return str(value).strip()
    return ""


def unit_scale_to_um(values: np.ndarray, unit_text: str) -> np.ndarray:
    unit = unit_text.lower().strip()
    wave = np.asarray(values, dtype=float)
    if "angstrom" in unit or unit in {"a", "aa"}:
        return wave * 1e-4
    if "nm" in unit:
        return wave * 1e-3
    if "meter" in unit or unit == "m":
        return wave * 1e6
    if np.nanmedian(wave) > 1000:
        return wave * 1e-4
    if np.nanmedian(wave) > 20:
        return wave * 1e-3
    return wave


def find_column(names: list[str], options: list[str]) -> str | None:
    upper = {name.upper(): name for name in names}
    for option in options:
        if option in upper:
            return upper[option]
    return None


def extract_spectra(path: Path) -> list[pd.DataFrame]:
    outputs = []
    with fits.open(path, memmap=False) as hdul:
        primary = hdul[0].header
        for hdu_index, hdu in enumerate(hdul[1:], 1):
            data = hdu.data
            names = list(getattr(data, "names", []) or [])
            if not names:
                continue
            wave_col = find_column(names, ["WAVELENGTH", "WAVE", "LAMBDA", "LAM"])
            flux_col = find_column(names, ["FLUX", "FNU", "FLAM", "SPEC1D", "SCI"])
            if wave_col is None or flux_col is None:
                continue
            err_col = find_column(names, ["FLUX_ERROR", "ERROR", "ERR", "E_FLUX", "SIGMA"])
            wave_raw = np.asarray(data[wave_col], dtype=float).ravel()
            flux = np.asarray(data[flux_col], dtype=float).ravel()
            error = np.asarray(data[err_col], dtype=float).ravel() if err_col else np.full_like(flux, np.nan)
            n = min(wave_raw.size, flux.size, error.size)
            wave_raw, flux, error = wave_raw[:n], flux[:n], error[:n]
            unit_text = ""
            try:
                unit_text = str(hdu.columns[wave_col].unit or "")
            except Exception:
                pass
            wave_um = unit_scale_to_um(wave_raw, unit_text)
            finite = np.isfinite(wave_um) & np.isfinite(flux) & (wave_um > 0)
            if finite.sum() < 40:
                continue

            headers = [hdu.header, primary]
            grating = header_value(headers, ["GRATING", "DISPERSR", "OPT_ELEM"]).upper()
            filt = header_value(headers, ["FILTER", "FWA_POS"]).upper()
            combined_text = f"{path.name} {grating} {filt}".upper()
            mode = next((item for item in HIGH_RES_MODES if item in combined_text), "")
            if mode not in HIGH_RES_MODES:
                continue
            if "PRISM" in combined_text or any(item in combined_text for item in ("G140M", "G235M", "G395M")):
                continue

            frame = pd.DataFrame({
                "observed_wavelength_um": wave_um[finite],
                "flux_native": flux[finite],
                "flux_error_native": error[finite],
            }).sort_values("observed_wavelength_um").drop_duplicates("observed_wavelength_um")
            frame["rest_wavelength_A"] = frame.observed_wavelength_um * 1e4 / (1.0 + Z)
            frame["grating"] = mode
            frame["filter"] = filt or HIGH_RES_MODES[mode]
            frame["source_file"] = path.name
            frame["hdu"] = hdu_index
            frame["wavelength_unit_original"] = unit_text
            outputs.append(frame.reset_index(drop=True))
    return outputs


def robust_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    low, high = np.nanpercentile(finite, [1, 99])
    if not np.isfinite(low + high) or high <= low:
        med = float(np.nanmedian(finite))
        sig = float(np.nanstd(finite)) or 1.0
        return med - 3 * sig, med + 3 * sig
    margin = 0.15 * (high - low)
    return float(low - margin), float(high + margin)


def choose_best_window(spectra: list[pd.DataFrame], low: float, high: float) -> tuple[pd.DataFrame, int]:
    ranked = []
    for frame in spectra:
        mask = (frame.rest_wavelength_A >= low) & (frame.rest_wavelength_A <= high)
        count = int(mask.sum())
        if count > 0:
            ranked.append((count, frame, mask))
    if not ranked:
        raise RuntimeError(f"No high-resolution spectrum covers rest-frame {low:.1f}-{high:.1f} A.")
    count, frame, mask = max(ranked, key=lambda item: item[0])
    return frame.loc[mask].copy(), count


def plot_overviews(spectra: list[pd.DataFrame]) -> list[Path]:
    paths = []
    for mode in HIGH_RES_MODES:
        candidates = [frame for frame in spectra if str(frame.grating.iloc[0]).upper() == mode]
        if not candidates:
            continue
        frame = max(candidates, key=len)
        fig, ax = plt.subplots(figsize=(18, 8), constrained_layout=True)
        ax.plot(frame.observed_wavelength_um, frame.flux_native, lw=0.55,
                label=f"Native {mode} samples: {len(frame):,}")
        ax.set_xlabel("Observed wavelength [um]")
        ax.set_ylabel("Flux [native FITS units]")
        ax.set_title(f"{TARGET} — real JWST/NIRSpec {mode}/{frame['filter'].iloc[0]} spectrum")
        ax.grid(True, lw=0.4, alpha=0.32)
        ax.legend()
        path = PNG / f"{VERSION}_{mode}_FULL_NATIVE_SPECTRUM.png"
        fig.savefig(path, dpi=420, bbox_inches="tight")
        plt.show(); plt.close(fig)
        paths.append(path)
    return paths


def plot_regions(spectra: list[pd.DataFrame]) -> tuple[list[Path], pd.DataFrame, pd.DataFrame]:
    paths = []
    audit_rows = []
    samples = []
    fig_atlas, axes = plt.subplots(3, 2, figsize=(20, 20), constrained_layout=True)

    for ax_atlas, region in zip(axes.ravel(), REGIONS):
        low = region["center_A"] - 50.0
        high = region["center_A"] + 50.0
        window, count = choose_best_window(spectra, low, high)
        mode = str(window.grating.iloc[0])
        source_file = str(window.source_file.iloc[0])
        x = window.rest_wavelength_A.to_numpy(float)
        y = window.flux_native.to_numpy(float)
        e = window.flux_error_native.to_numpy(float)

        fig, ax = plt.subplots(figsize=(18, 8.5), constrained_layout=True)
        for target_ax in (ax, ax_atlas):
            good_err = np.isfinite(e) & (e >= 0)
            if good_err.any():
                target_ax.fill_between(x[good_err], y[good_err] - e[good_err], y[good_err] + e[good_err],
                                       alpha=0.10, linewidth=0)
            target_ax.plot(x, y, lw=0.62)
            target_ax.scatter(x, y, s=5.5, alpha=0.72)
            for label, rest in region["lines"]:
                target_ax.axvline(rest, lw=0.85, ls="--", alpha=0.9)
            target_ax.set_xlim(low, high)
            target_ax.set_ylim(*robust_limits(y))
            target_ax.grid(True, lw=0.38, alpha=0.30)
            target_ax.set_xlabel("Rest-frame wavelength [A]")
            target_ax.set_ylabel("Flux [native FITS units]")

        ax.set_title(f"{TARGET} — {region['title']} | real {mode} native samples: {count}")
        ax.legend([f"{mode} native detector samples: {count}"], loc="best")
        ax.text(0.012, 0.022,
                f"Source FITS: {source_file}\nNo interpolation, no PRISM, no synthetic spectrum.",
                transform=ax.transAxes, fontsize=8.3, va="bottom")
        ax_atlas.set_title(f"{region['title']} | {mode} | n={count}", fontsize=10.5)

        path = PNG / f"{VERSION}_{region['key']}_{mode}_RAW_NATIVE.png"
        fig.savefig(path, dpi=500, bbox_inches="tight")
        plt.show(); plt.close(fig)
        paths.append(path)

        saved = window.copy()
        saved["region"] = region["key"]
        samples.append(saved)
        audit_rows.append({
            "region": region["key"],
            "window_low_A": low,
            "window_high_A": high,
            "native_samples": count,
            "grating": mode,
            "filter": str(window["filter"].iloc[0]),
            "source_file": source_file,
            "hdu": int(window.hdu.iloc[0]),
            "redshift": Z,
        })

    fig_atlas.suptitle(
        f"{TARGET} — six real high-resolution JWST/NIRSpec windows\n"
        "G140H/G235H/G395H only | native samples only | no interpolation",
        fontsize=17,
    )
    atlas_path = PNG / f"{VERSION}_SIX_REGION_REAL_HIGH_RES_ATLAS.png"
    fig_atlas.savefig(atlas_path, dpi=450, bbox_inches="tight")
    plt.show(); plt.close(fig_atlas)
    paths.append(atlas_path)
    return paths, pd.DataFrame(audit_rows), pd.concat(samples, ignore_index=True)


def main() -> None:
    set_style()
    obs_df, candidates = query_mast()
    manifest_df, fits_paths = download_candidates(candidates)

    spectra = []
    file_audit = []
    for path in fits_paths:
        extracted = extract_spectra(path)
        for frame in extracted:
            spectra.append(frame)
            file_audit.append({
                "file": path.name,
                "hdu": int(frame.hdu.iloc[0]),
                "grating": str(frame.grating.iloc[0]),
                "filter": str(frame["filter"].iloc[0]),
                "native_samples": len(frame),
                "wavelength_min_um": float(frame.observed_wavelength_um.min()),
                "wavelength_max_um": float(frame.observed_wavelength_um.max()),
            })

    if not spectra:
        raise RuntimeError(
            "Downloaded FITS files contained no validated G140H/G235H/G395H 1D spectra. "
            "No PRISM fallback was used."
        )

    file_audit_df = pd.DataFrame(file_audit).sort_values(["grating", "native_samples"], ascending=[True, False])
    file_audit_path = CSV / f"{VERSION}_VALIDATED_HIGH_RES_FITS_AUDIT.csv"
    file_audit_df.to_csv(file_audit_path, index=False)

    overview_paths = plot_overviews(spectra)
    region_paths, region_audit, sample_table = plot_regions(spectra)
    region_audit_path = CSV / f"{VERSION}_REGION_NATIVE_SAMPLE_COUNTS.csv"
    sample_path = CSV / f"{VERSION}_SIX_REGION_RAW_NATIVE_SAMPLES.csv"
    region_audit.to_csv(region_audit_path, index=False)
    sample_table.to_csv(sample_path, index=False)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"TARGET             {TARGET}")
    print(f"COORDINATES        RA={RA_DEG:.7f} deg  Dec={DEC_DEG:.7f} deg")
    print(f"REDSHIFT           {Z:.5f}")
    print("ACCEPTED MODES     G140H/F100LP, G235H/F170LP, G395H/F290LP")
    print("REJECTED           PRISM and all medium-resolution gratings")
    print(f"MAST OBSERVATIONS  {len(obs_df)} high-resolution rows")
    print(f"CANDIDATE FILES    {len(candidates)}")
    print(f"DOWNLOADED FITS    {len(fits_paths)}")
    print(f"VALID 1D SPECTRA   {len(spectra)}")
    print("REGION SAMPLE COUNTS")
    for row in region_audit.itertuples(index=False):
        print(f"  {row.region:<14} {row.grating:<6} n={row.native_samples:5d}  {row.source_file}")
    for path in overview_paths:
        print(f"OVERVIEW PNG       {path}")
    for path in region_paths:
        print(f"REGION PNG         {path}")
    print(f"FITS AUDIT CSV     {file_audit_path}")
    print(f"REGION AUDIT CSV   {region_audit_path}")
    print(f"RAW SAMPLE CSV     {sample_path}")
    print(f"Timestamp          {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
