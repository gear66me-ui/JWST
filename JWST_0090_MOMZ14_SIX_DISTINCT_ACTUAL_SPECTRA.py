#!/usr/bin/env python3
"""
JWST_0090_MOMZ14_SIX_DISTINCT_ACTUAL_SPECTRA.py

Replot the cached, coordinate-verified MoM-z14 JWST/NIRSpec X1D spectrum as:
  1) one full raw spectrum with an element-color reference table, and
  2) five distinct, populated, nonidentical raw spectral-region plots.

Actual detector samples only. No smoothing, interpolation, synthetic profiles,
or AI imagery. Matplotlib only.
"""
from __future__ import annotations

import importlib.util
import math
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

VERSION = "JWST_0090"
GALAXY = "MoM-z14"
Z = 14.44
TARGET_RA = 150.0933255
TARGET_DEC = 2.2731627
MIN_SAMPLES = 18

ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
DATA_DIR = ROOT / "DATA"
for directory in (PNG_DIR, CSV_DIR, DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)

ELEMENT_COLORS = {
    "Nitrogen": "#ff4d6d",
    "Carbon": "#00d4ff",
    "Helium": "#c77dff",
    "Oxygen": "#2f80ed",
}

REGIONS = [
    {
        "key": "N_IV",
        "title": "N IV] rest-UV region",
        "element": "Nitrogen",
        "window_A": (1425.0, 1525.0),
        "lines": [("N IV] 1483", "Nitrogen", 1483.32),
                  ("N IV] 1487", "Nitrogen", 1486.50)],
    },
    {
        "key": "C_IV",
        "title": "C IV rest-UV region",
        "element": "Carbon",
        "window_A": (1525.0, 1600.0),
        "lines": [("C IV 1548", "Carbon", 1548.20),
                  ("C IV 1551", "Carbon", 1550.77)],
    },
    {
        "key": "HEII_OIII",
        "title": "He II + O III] rest-UV blend region",
        "element": "Blend",
        "window_A": (1600.0, 1705.0),
        "lines": [("He II 1640", "Helium", 1640.42),
                  ("O III] 1661", "Oxygen", 1660.81),
                  ("O III] 1666", "Oxygen", 1666.15)],
    },
    {
        "key": "N_III",
        "title": "N III] rest-UV region",
        "element": "Nitrogen",
        "window_A": (1705.0, 1810.0),
        "lines": [("N III] 1747", "Nitrogen", 1746.82),
                  ("N III] 1749", "Nitrogen", 1748.65),
                  ("N III] 1750", "Nitrogen", 1749.67),
                  ("N III] 1752", "Nitrogen", 1752.16),
                  ("N III] 1754", "Nitrogen", 1753.99)],
    },
    {
        "key": "C_III",
        "title": "C III] rest-UV region",
        "element": "Carbon",
        "window_A": (1810.0, 1995.0),
        "lines": [("C III] 1907", "Carbon", 1906.68),
                  ("C III] 1909", "Carbon", 1908.73)],
    },
]


def reset_style() -> None:
    plt.close("all")
    matplotlib.rcdefaults()
    matplotlib.rcParams.update({
        "text.usetex": False,
        "figure.facecolor": "#050712",
        "axes.facecolor": "#07101f",
        "axes.edgecolor": "#8ca3b8",
        "axes.labelcolor": "#f1f5f9",
        "xtick.color": "#dbeafe",
        "ytick.color": "#dbeafe",
        "text.color": "#f8fafc",
        "font.size": 10,
    })


def to_rest_A(wavelength_um: np.ndarray) -> np.ndarray:
    return np.asarray(wavelength_um, dtype=float) * 1.0e4 / (1.0 + Z)


def to_observed_um(rest_A: np.ndarray | float) -> np.ndarray:
    return np.asarray(rest_A, dtype=float) * (1.0 + Z) * 1.0e-4


def find_cached_csv() -> Path | None:
    preferred = [
        CSV_DIR / "JWST_0087_MOMZ14_ACTUAL_NIRSPEC_X1D_RAW.csv",
        CSV_DIR / "JWST_0088_MOMZ14_ACTUAL_NIRSPEC_X1D_RAW.csv",
        CSV_DIR / "JWST_0089_MOMZ14_ACTUAL_NIRSPEC_X1D_RAW.csv",
    ]
    for path in preferred:
        if path.exists() and path.stat().st_size > 1000:
            return path
    patterns = [
        "*MOMZ14*ACTUAL*NIRSPEC*X1D*RAW*.csv",
        "*MOMZ14*X1D*RAW*.csv",
        "*MOMZ14*EXACT*JWST*.csv",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(CSV_DIR.glob(pattern))
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def read_cached_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    frame = pd.read_csv(path)
    lookup = {str(column).lower(): column for column in frame.columns}

    wave_col = next((lookup[name] for name in (
        "observed_wavelength_um", "wavelength_um", "wave_um") if name in lookup), None)
    flux_col = next((lookup[name] for name in (
        "flux_display_units", "flux", "original_flux") if name in lookup), None)
    err_col = next((lookup[name] for name in (
        "flux_error_display_units", "flux_error", "error", "original_flux_error") if name in lookup), None)

    if wave_col is None or flux_col is None:
        raise RuntimeError(f"Cached CSV lacks wavelength/flux columns: {path}")

    wave = pd.to_numeric(frame[wave_col], errors="coerce").to_numpy(float)
    flux = pd.to_numeric(frame[flux_col], errors="coerce").to_numpy(float)
    error = (pd.to_numeric(frame[err_col], errors="coerce").to_numpy(float)
             if err_col is not None else np.full_like(flux, np.nan))
    valid = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    order = np.argsort(wave[valid])
    wave, flux, error = wave[valid][order], flux[valid][order], error[valid][order]
    if len(wave) < 100:
        raise RuntimeError(f"Cached spectrum contains only {len(wave)} valid samples: {path}")
    return wave, flux, error, str(path)


def table_field(data, names):
    if data is None or not getattr(data, "names", None):
        return None, None
    lookup = {str(name).upper(): name for name in data.names}
    for name in names:
        if name.upper() in lookup:
            actual = lookup[name.upper()]
            return actual, data[actual]
    return None, None


def unit_to_um(values, unit) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    text = str(unit or "").lower()
    if "angstrom" in text or text.strip() in {"a", "aa"}:
        return array * 1.0e-4
    if "nm" in text:
        return array * 1.0e-3
    if "um" in text or "micron" in text:
        return array
    median = float(np.nanmedian(array))
    return array * 1.0e-4 if median > 1000 else (array * 1.0e-3 if median > 10 else array)


def header_coordinates(headers):
    pairs = [
        ("SRCRA", "SRCDEC"), ("RA_OBJ", "DEC_OBJ"),
        ("OBJ_RA", "OBJ_DEC"), ("SLIT_RA", "SLIT_DEC"),
        ("TARG_RA", "TARG_DEC"), ("RA_TARG", "DEC_TARG"),
    ]
    for header in headers:
        for ra_key, dec_key in pairs:
            try:
                ra = float(header.get(ra_key))
                dec = float(header.get(dec_key))
                if math.isfinite(ra) and math.isfinite(dec):
                    return ra, dec
            except Exception:
                continue
    return None, None


def read_existing_fits() -> tuple[np.ndarray, np.ndarray, np.ndarray, str] | None:
    target = SkyCoord(TARGET_RA * u.deg, TARGET_DEC * u.deg)
    fits_files = sorted(DATA_DIR.rglob("*.fits"), key=lambda p: p.stat().st_mtime, reverse=True)
    candidates = []
    for path in fits_files:
        try:
            with fits.open(path, memmap=False) as hdul:
                primary = hdul[0].header
                for hdu_index, hdu in enumerate(hdul[1:], 1):
                    wave_name, wave_values = table_field(hdu.data, ["WAVELENGTH", "WAVE"])
                    flux_name, flux_values = table_field(hdu.data, ["FLUX"])
                    if wave_values is None or flux_values is None:
                        continue
                    try:
                        wave_unit = hdu.columns[wave_name].unit
                    except Exception:
                        wave_unit = None
                    wave = unit_to_um(wave_values, wave_unit)
                    flux = np.asarray(flux_values, dtype=float).ravel()
                    _, err_values = table_field(hdu.data, ["FLUX_ERROR", "ERROR", "ERR"])
                    error = np.asarray(err_values, dtype=float).ravel() if err_values is not None else np.full_like(flux, np.nan)
                    _, dq_values = table_field(hdu.data, ["DQ", "QUALITY"])
                    dq = np.asarray(dq_values).ravel() if dq_values is not None else np.zeros_like(flux, dtype=int)
                    valid = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
                    if dq.size == valid.size:
                        valid &= dq == 0
                    if valid.sum() < 100:
                        continue
                    ra, dec = header_coordinates([hdu.header, primary])
                    separation = math.inf
                    if ra is not None:
                        separation = float(SkyCoord(ra * u.deg, dec * u.deg).separation(target).arcsec)
                    order = np.argsort(wave[valid])
                    candidates.append((separation, wave[valid][order], flux[valid][order], error[valid][order], f"{path} HDU {hdu_index}"))
        except Exception:
            continue
    verified = [candidate for candidate in candidates if math.isfinite(candidate[0]) and candidate[0] <= 1.5]
    if not verified:
        return None
    verified.sort(key=lambda item: (item[0], -len(item[1])))
    _, wave, flux, error, source = verified[0]
    finite = np.abs(flux[np.isfinite(flux)])
    if finite.size and np.nanmedian(finite) < 1.0e-3:
        flux = flux * 1.0e9
        error = error * 1.0e9
    return wave, flux, error, source


def load_actual_spectrum():
    cached = find_cached_csv()
    if cached is not None:
        return read_cached_csv(cached)
    from_fits = read_existing_fits()
    if from_fits is not None:
        return from_fits
    raise RuntimeError(
        "No cached MoM-z14 X1D CSV or coordinate-verified FITS file exists in /content/JWST_OUTPUT. "
        "Run JWST_0087 once in this Colab runtime to create the actual spectrum cache."
    )


def robust_limits(values: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    low, high = np.nanpercentile(finite, [1.5, 98.5])
    if np.isfinite(errors).any():
        finite_err = errors[np.isfinite(errors) & (errors >= 0)]
        if finite_err.size:
            pad = min(float(np.nanmedian(finite_err)), max(float(high - low), 1.0))
            low -= pad
            high += pad
    if high <= low:
        center = float(np.nanmedian(finite))
        spread = float(np.nanstd(finite)) or 1.0
        low, high = center - 3.0 * spread, center + 3.0 * spread
    margin = 0.12 * (high - low)
    return float(low - margin), float(high + margin)


def style_axis(axis) -> None:
    axis.grid(True, color="#334155", linewidth=0.55, alpha=0.58)
    for spine in axis.spines.values():
        spine.set_color("#94a3b8")


def all_reference_rows() -> list[dict]:
    rows = []
    index = 1
    for region in REGIONS:
        for label, element, rest_wave in region["lines"]:
            rows.append({
                "id": index,
                "region": region["key"],
                "line": label,
                "element": element,
                "rest_wavelength_A": rest_wave,
                "observed_wavelength_um_z14p44": float(to_observed_um(rest_wave)),
                "color_hex": ELEMENT_COLORS[element],
            })
            index += 1
    return rows


def select_real_samples(rest_A: np.ndarray, region: dict) -> np.ndarray:
    low, high = region["window_A"]
    indices = np.flatnonzero((rest_A >= low) & (rest_A <= high))
    center = float(np.mean([item[2] for item in region["lines"]]))
    center_index = int(np.argmin(np.abs(rest_A - center)))

    if len(indices) < MIN_SAMPLES:
        half = MIN_SAMPLES // 2
        start = max(0, center_index - half)
        stop = min(len(rest_A), start + MIN_SAMPLES)
        start = max(0, stop - MIN_SAMPLES)
        indices = np.arange(start, stop, dtype=int)

    if len(indices) < 8:
        raise RuntimeError(f"{region['key']} contains only {len(indices)} actual samples.")
    return indices


def draw_reference_lines(axis, region: dict) -> None:
    for line_number, (label, element, rest_wave) in enumerate(region["lines"]):
        color = ELEMENT_COLORS[element]
        axis.axvline(rest_wave, color=color, linewidth=1.6, linestyle="--", alpha=0.96, zorder=5)
        axis.text(
            rest_wave,
            0.98 - 0.11 * (line_number % 4),
            label,
            rotation=90,
            transform=axis.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=8.5,
            color=color,
            bbox={"facecolor": "#050712", "alpha": 0.76, "edgecolor": "none", "pad": 1.5},
        )


def trace_color(region: dict) -> str:
    if region["element"] in ELEMENT_COLORS:
        return ELEMENT_COLORS[region["element"]]
    return "#e2e8f0"


def make_full_plot(wave_um, rest_A, flux, error) -> Path:
    reference_rows = all_reference_rows()
    figure = plt.figure(figsize=(18, 12), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=[5.7, 2.3])
    axis = figure.add_subplot(grid[0, 0])
    table_axis = figure.add_subplot(grid[1, 0])

    axis.plot(wave_um, flux, color="#e2e8f0", linewidth=0.9,
              drawstyle="steps-mid", label="Actual NIRSpec X1D flux samples", zorder=3)
    good = np.isfinite(error) & (error >= 0)
    if good.any():
        axis.fill_between(wave_um[good], flux[good] - error[good], flux[good] + error[good],
                          color="#38bdf8", alpha=0.12, step="mid",
                          label="1-sigma uncertainty", zorder=1)

    for row in reference_rows:
        axis.axvline(row["observed_wavelength_um_z14p44"],
                     color=row["color_hex"], linewidth=1.25,
                     linestyle="--", alpha=0.93, zorder=4)

    axis.set_xlabel("Observed wavelength [micrometers]")
    axis.set_ylabel("Flux density [display units from cached X1D]")
    axis.set_title(
        "MoM-z14 — actual public JWST/NIRSpec prism X1D spectrum\n"
        "Raw jagged detector samples; element-colored laboratory reference positions",
        fontsize=17, pad=14,
    )
    axis.set_ylim(*robust_limits(flux, error))
    style_axis(axis)
    axis.legend(loc="upper right", fontsize=9, facecolor="#020617", edgecolor="#475569")
    top = axis.secondary_xaxis("top", functions=(to_rest_A, to_observed_um))
    top.set_xlabel("Rest-frame wavelength [angstrom] at z = 14.44")

    table_axis.axis("off")
    table_axis.set_title("Reference-line color key", loc="left", fontsize=13, pad=7)
    table_rows = [[
        row["id"], row["element"], row["line"],
        f"{row['rest_wavelength_A']:.2f}",
        f"{row['observed_wavelength_um_z14p44']:.5f}",
    ] for row in reference_rows]
    table = table_axis.table(
        cellText=table_rows,
        colLabels=["ID", "Element", "Transition", "Rest A", "Observed um"],
        cellLoc="left", colLoc="left",
        bbox=[0.00, 0.00, 0.72, 0.93],
        colWidths=[0.05, 0.13, 0.27, 0.12, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#475569")
        cell.set_linewidth(0.45)
        if row_index == 0:
            cell.set_facecolor("#223047")
            cell.set_text_props(color="#ffffff", weight="bold")
        else:
            item = reference_rows[row_index - 1]
            cell.set_facecolor("#111827" if row_index % 2 else "#172033")
            cell.set_text_props(color="#e5edf5")
            if column_index == 0:
                cell.set_facecolor(item["color_hex"])
                cell.set_text_props(color="#050712", weight="bold")

    table_axis.text(
        0.76, 0.82,
        "ELEMENT COLORS\n\n"
        "Nitrogen  — magenta/red\n"
        "Carbon    — cyan\n"
        "Helium    — violet\n"
        "Oxygen    — blue\n\n"
        "White curve: actual X1D flux\n"
        "Blue band: actual 1-sigma error\n"
        "No smoothing or interpolation",
        transform=table_axis.transAxes, ha="left", va="top", fontsize=10.2,
        linespacing=1.48,
        bbox={"facecolor": "#0b1220", "edgecolor": "#475569", "boxstyle": "round,pad=0.7"},
    )

    output = PNG_DIR / f"{VERSION}_01_FULL_ACTUAL_NIRSPEC_ELEMENT_COLORS.png"
    figure.savefig(output, dpi=360, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.show()
    plt.close(figure)
    return output


def make_region_plot(wave_um, rest_A, flux, error, region: dict, plot_number: int):
    indices = select_real_samples(rest_A, region)
    x_rest = rest_A[indices]
    x_obs = wave_um[indices]
    y = flux[indices]
    e = error[indices]

    figure, axis = plt.subplots(figsize=(16.5, 8.3), constrained_layout=True)
    color = trace_color(region)
    axis.plot(x_rest, y, color=color, linewidth=1.35,
              drawstyle="steps-mid", label=f"Actual X1D samples: n={len(indices)}", zorder=3)
    axis.scatter(x_rest, y, s=18, color=color, edgecolors="#020617",
                 linewidths=0.45, zorder=4)
    good = np.isfinite(e) & (e >= 0)
    if good.any():
        axis.fill_between(x_rest[good], y[good] - e[good], y[good] + e[good],
                          color="#38bdf8", alpha=0.16, step="mid",
                          label="1-sigma uncertainty", zorder=1)

    draw_reference_lines(axis, region)
    axis.set_xlim(float(x_rest.min()), float(x_rest.max()))
    axis.set_ylim(*robust_limits(y, e))
    axis.set_xlabel("Rest-frame vacuum wavelength [angstrom]")
    axis.set_ylabel("Flux density [display units from cached X1D]")
    axis.set_title(
        f"MoM-z14 — {region['title']}\n"
        f"Distinct actual-data window {x_rest.min():.2f}-{x_rest.max():.2f} A | "
        f"observed {x_obs.min():.5f}-{x_obs.max():.5f} um | n={len(indices)}",
        fontsize=15, pad=12,
    )
    style_axis(axis)
    axis.legend(loc="upper right", fontsize=9, facecolor="#020617", edgecolor="#475569")
    top = axis.secondary_xaxis("top", functions=(to_observed_um, to_rest_A))
    top.set_xlabel("Observed wavelength [micrometers]")

    output = PNG_DIR / f"{VERSION}_{plot_number:02d}_{region['key']}_ACTUAL_RAW_SPECTRUM.png"
    figure.savefig(output, dpi=360, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.show()
    plt.close(figure)

    sample_frame = pd.DataFrame({
        "region": region["key"],
        "sample_index": indices,
        "observed_wavelength_um": x_obs,
        "rest_wavelength_angstrom_z14p44": x_rest,
        "flux_display_units": y,
        "flux_error_display_units": e,
        "actual_x1d_sample": True,
        "smoothed": False,
        "interpolated": False,
    })
    csv_path = CSV_DIR / f"{VERSION}_{plot_number:02d}_{region['key']}_ACTUAL_SAMPLES.csv"
    sample_frame.to_csv(csv_path, index=False)
    return output, csv_path, indices


def main() -> None:
    reset_style()
    wave_um, flux, error, source = load_actual_spectrum()
    rest_A = to_rest_A(wave_um)

    full_png = make_full_plot(wave_um, rest_A, flux, error)
    outputs = []
    used_signatures = set()

    for plot_number, region in enumerate(REGIONS, start=2):
        png_path, csv_path, indices = make_region_plot(
            wave_um, rest_A, flux, error, region, plot_number
        )
        signature = (int(indices[0]), int(indices[-1]))
        if signature in used_signatures:
            raise RuntimeError(
                f"Duplicate sample window detected for {region['key']}: {signature}. "
                "Refusing to emit repeated plots."
            )
        used_signatures.add(signature)
        outputs.append((region, png_path, csv_path, indices))

    reference_csv = CSV_DIR / f"{VERSION}_REFERENCE_LINE_COLOR_KEY.csv"
    pd.DataFrame(all_reference_rows()).to_csv(reference_csv, index=False)

    index_rows = []
    for region, png_path, csv_path, indices in outputs:
        index_rows.append({
            "plot": region["key"],
            "element": region["element"],
            "png": str(png_path),
            "csv": str(csv_path),
            "sample_count": len(indices),
            "first_sample_index": int(indices[0]),
            "last_sample_index": int(indices[-1]),
            "rest_min_A": float(rest_A[indices].min()),
            "rest_max_A": float(rest_A[indices].max()),
            "observed_min_um": float(wave_um[indices].min()),
            "observed_max_um": float(wave_um[indices].max()),
        })
    index_csv = CSV_DIR / f"{VERSION}_SIX_PLOT_INDEX.csv"
    pd.DataFrame(index_rows).to_csv(index_csv, index=False)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"SOURCE          {source}")
    print("DATA            actual cached JWST/NIRSpec X1D detector samples")
    print("PLOTS           1 full spectrum + 5 distinct populated regions")
    print("SMOOTHING       none")
    print("INTERPOLATION   none")
    print("SYNTHETIC DATA  none")
    print("ELEMENT COLORS  N=magenta/red  C=cyan  He=violet  O=blue")
    print(f"FULL PNG        {full_png}")
    for region, png_path, csv_path, indices in outputs:
        print(f"{region['key']:<12} n={len(indices):>3}  {rest_A[indices].min():8.2f}-{rest_A[indices].max():8.2f} A")
        print(f"PNG             {png_path}")
        print(f"CSV             {csv_path}")
    print(f"LINE KEY CSV    {reference_csv}")
    print(f"PLOT INDEX CSV  {index_csv}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
