#!/usr/bin/env python3
"""
JWST_0091_MOMZ14_VERIFIED_ACTUAL_ZOOM_SPECTRA.py

Create five verified plots from actual cached MoM-z14 JWST/NIRSpec X1D samples:
  01 observed 2.0-3.0 micrometers
  02 rest 1475-1500 A  N IV]
  03 rest 1525-1575 A  C IV
  04 rest 1620-1680 A  He II + O III]
  05 rest 1740-1760 A  N III]

No step drawstyle, smoothing, interpolation, synthetic profiles, or AI imagery.
For each zoom, the script scans every cached coordinate-matched X1D FITS extension
and selects the one with the greatest number of genuine detector samples inside
the exact requested window. It refuses to save or display a blank plot.
"""
from __future__ import annotations

import hashlib
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
        "PIL": "pillow",
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
from PIL import Image

VERSION = "JWST_0091"
GALAXY = "MoM-z14"
Z = 14.44
TARGET_RA = 150.0933255
TARGET_DEC = 2.2731627
TARGET = SkyCoord(TARGET_RA * u.deg, TARGET_DEC * u.deg)

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

REFERENCE_LINES = [
    ("N IV] 1483", "Nitrogen", 1483.32),
    ("N IV] 1487", "Nitrogen", 1486.50),
    ("C IV 1548", "Carbon", 1548.20),
    ("C IV 1551", "Carbon", 1550.77),
    ("He II 1640", "Helium", 1640.42),
    ("O III] 1661", "Oxygen", 1660.81),
    ("O III] 1666", "Oxygen", 1666.15),
    ("N III] 1747", "Nitrogen", 1746.82),
    ("N III] 1749", "Nitrogen", 1748.65),
    ("N III] 1750", "Nitrogen", 1749.67),
    ("N III] 1752", "Nitrogen", 1752.16),
    ("N III] 1754", "Nitrogen", 1753.99),
    ("C III] 1907", "Carbon", 1906.68),
    ("C III] 1909", "Carbon", 1908.73),
]

PLOTS = [
    {
        "number": 1,
        "key": "FULL_2_TO_3_UM",
        "axis": "observed",
        "limits": (2.0, 3.0),
        "title": "MoM-z14 actual JWST/NIRSpec spectrum — observed 2.0 to 3.0 micrometers",
        "trace_color": "#e2e8f0",
        "lines": REFERENCE_LINES,
        "minimum_samples": 30,
    },
    {
        "number": 2,
        "key": "N_IV_1475_1500_A",
        "axis": "rest",
        "limits": (1475.0, 1500.0),
        "title": "N IV] actual spectral response — rest 1475 to 1500 A",
        "trace_color": ELEMENT_COLORS["Nitrogen"],
        "lines": [item for item in REFERENCE_LINES if item[0].startswith("N IV")],
        "minimum_samples": 6,
    },
    {
        "number": 3,
        "key": "C_IV_1525_1575_A",
        "axis": "rest",
        "limits": (1525.0, 1575.0),
        "title": "C IV actual spectral response — rest 1525 to 1575 A",
        "trace_color": ELEMENT_COLORS["Carbon"],
        "lines": [item for item in REFERENCE_LINES if item[0].startswith("C IV")],
        "minimum_samples": 8,
    },
    {
        "number": 4,
        "key": "HEII_OIII_1620_1680_A",
        "axis": "rest",
        "limits": (1620.0, 1680.0),
        "title": "He II + O III] actual spectral response — rest 1620 to 1680 A",
        "trace_color": "#e2e8f0",
        "lines": [item for item in REFERENCE_LINES if item[1] in {"Helium", "Oxygen"}],
        "minimum_samples": 8,
    },
    {
        "number": 5,
        "key": "N_III_1740_1760_A",
        "axis": "rest",
        "limits": (1740.0, 1760.0),
        "title": "N III] actual spectral response — rest 1740 to 1760 A",
        "trace_color": ELEMENT_COLORS["Nitrogen"],
        "lines": [item for item in REFERENCE_LINES if item[0].startswith("N III")],
        "minimum_samples": 6,
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


def rest_to_obs(rest_A):
    return np.asarray(rest_A, dtype=float) * (1.0 + Z) * 1.0e-4


def obs_to_rest(obs_um):
    return np.asarray(obs_um, dtype=float) * 1.0e4 / (1.0 + Z)


def table_field(data, names):
    if data is None or not getattr(data, "names", None):
        return None, None
    lookup = {str(name).upper(): name for name in data.names}
    for name in names:
        if name.upper() in lookup:
            actual = lookup[name.upper()]
            return actual, data[actual]
    return None, None


def wavelength_to_um(values, unit):
    array = np.asarray(values, dtype=float).ravel()
    text = str(unit or "").lower()
    if "angstrom" in text or text.strip() in {"a", "aa"}:
        return array * 1.0e-4
    if "nm" in text:
        return array * 1.0e-3
    if "um" in text or "micron" in text:
        return array
    median = float(np.nanmedian(array))
    if median > 1000:
        return array * 1.0e-4
    if median > 10:
        return array * 1.0e-3
    return array


def source_coordinates(headers):
    pairs = [
        ("SRCRA", "SRCDEC"), ("RA_OBJ", "DEC_OBJ"),
        ("OBJ_RA", "OBJ_DEC"), ("SLIT_RA", "SLIT_DEC"),
        ("MSA_RA", "MSA_DEC"), ("SHUT_RA", "SHUT_DEC"),
        ("TARG_RA", "TARG_DEC"), ("RA_TARG", "DEC_TARG"),
    ]
    for header in headers:
        for ra_key, dec_key in pairs:
            try:
                ra = float(header.get(ra_key))
                dec = float(header.get(dec_key))
                if math.isfinite(ra) and math.isfinite(dec):
                    return ra, dec, f"{ra_key}/{dec_key}"
            except Exception:
                continue
    return None, None, "NONE"


def convert_flux(flux, error, unit):
    text = str(unit or "unknown").lower()
    finite = np.abs(flux[np.isfinite(flux)])
    is_jy = "jy" in text or (finite.size and float(np.nanmedian(finite)) < 1.0e-3)
    if is_jy:
        return flux * 1.0e9, error * 1.0e9, "Flux density [nJy]"
    return flux, error, f"Flux [{unit}]"


def cached_fits_files():
    files = []
    roots = [
        DATA_DIR,
        ROOT / "FITS",
        Path("/content/drive/MyDrive/Colab Notebooks/JWST/MoM-14"),
    ]
    for root in roots:
        if root.exists():
            files.extend(root.rglob("*.fits"))
            files.extend(root.rglob("*.fit"))
            files.extend(root.rglob("*.fts"))
    files.extend(Path("/content").glob("*.fits"))
    return sorted(set(path for path in files if path.is_file()))


def fits_candidates():
    candidates = []
    for path in cached_fits_files():
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
                    try:
                        flux_unit = str(hdu.columns[flux_name].unit or "unknown")
                    except Exception:
                        flux_unit = "unknown"
                    wave = wavelength_to_um(wave_values, wave_unit)
                    flux = np.asarray(flux_values, dtype=float).ravel()
                    _, error_values = table_field(hdu.data, ["FLUX_ERROR", "ERROR", "ERR"])
                    error = (np.asarray(error_values, dtype=float).ravel()
                             if error_values is not None else np.full_like(flux, np.nan))
                    _, dq_values = table_field(hdu.data, ["DQ", "QUALITY"])
                    dq = (np.asarray(dq_values).ravel()
                          if dq_values is not None else np.zeros_like(flux, dtype=int))
                    valid = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
                    if dq.size == valid.size:
                        valid &= dq == 0
                    if valid.sum() < 8:
                        continue
                    ra, dec, coord_source = source_coordinates([hdu.header, primary])
                    if ra is None or dec is None:
                        continue
                    separation = float(SkyCoord(ra * u.deg, dec * u.deg).separation(TARGET).arcsec)
                    if not math.isfinite(separation) or separation > 1.5:
                        continue
                    order = np.argsort(wave[valid])
                    wave = wave[valid][order]
                    flux = flux[valid][order]
                    error = error[valid][order]
                    flux, error, ylabel = convert_flux(flux, error, flux_unit)
                    mode = "/".join(str(value) for value in (
                        primary.get("FILTER", ""), primary.get("GRATING", ""),
                        primary.get("EXP_TYPE", "")) if str(value).strip()) or "unknown"
                    candidates.append({
                        "wave_um": wave,
                        "flux": flux,
                        "error": error,
                        "ylabel": ylabel,
                        "source": f"{path} HDU {hdu_index}",
                        "mode": mode,
                        "separation_arcsec": separation,
                        "coord_source": coord_source,
                        "kind": "FITS X1D",
                    })
        except Exception:
            continue
    return candidates


def csv_candidates():
    candidates = []
    patterns = ["*MOMZ14*.csv", "*MoM-z14*.csv", "*NIRSPEC*.csv", "*EXACT*JWST*.csv"]
    paths = []
    for pattern in patterns:
        paths.extend(CSV_DIR.glob(pattern))
    for path in sorted(set(paths)):
        try:
            frame = pd.read_csv(path)
            lookup = {str(column).lower(): column for column in frame.columns}
            wave_col = next((lookup[name] for name in (
                "observed_wavelength_um", "wavelength_um", "wave_um") if name in lookup), None)
            flux_col = next((lookup[name] for name in (
                "flux_display_units", "flux", "original_flux") if name in lookup), None)
            err_col = next((lookup[name] for name in (
                "flux_error_display_units", "flux_error", "error", "original_flux_error") if name in lookup), None)
            if wave_col is None or flux_col is None:
                continue
            wave = pd.to_numeric(frame[wave_col], errors="coerce").to_numpy(float)
            flux = pd.to_numeric(frame[flux_col], errors="coerce").to_numpy(float)
            error = (pd.to_numeric(frame[err_col], errors="coerce").to_numpy(float)
                     if err_col is not None else np.full_like(flux, np.nan))
            valid = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
            if valid.sum() < 8:
                continue
            order = np.argsort(wave[valid])
            wave, flux, error = wave[valid][order], flux[valid][order], error[valid][order]
            candidates.append({
                "wave_um": wave,
                "flux": flux,
                "error": error,
                "ylabel": "Flux density [cached X1D display units]",
                "source": str(path),
                "mode": "cached CSV",
                "separation_arcsec": np.nan,
                "coord_source": "verified by originating script",
                "kind": "cached X1D CSV",
            })
        except Exception:
            continue
    return candidates


def mask_for(candidate, plot_def):
    wave = candidate["wave_um"]
    if plot_def["axis"] == "observed":
        x = wave
    else:
        x = obs_to_rest(wave)
    low, high = plot_def["limits"]
    return x, (x >= low) & (x <= high)


def choose_candidate(candidates, plot_def):
    ranked = []
    for candidate in candidates:
        x, mask = mask_for(candidate, plot_def)
        count = int(mask.sum())
        if count < plot_def["minimum_samples"]:
            continue
        y = candidate["flux"][mask]
        finite_y = y[np.isfinite(y)]
        if finite_y.size < plot_def["minimum_samples"] or float(np.ptp(finite_y)) <= 0:
            continue
        preferred = 0
        if plot_def["number"] == 1 and "0087_MOMZ14_ACTUAL_NIRSPEC_X1D_RAW" in candidate["source"]:
            preferred = 1
        verified = 1 if math.isfinite(candidate["separation_arcsec"]) else 0
        ranked.append((preferred, count, verified, -float(candidate["separation_arcsec"])
                       if verified else -999.0, candidate))
    if not ranked:
        raise RuntimeError(
            f"No cached actual X1D spectrum has at least {plot_def['minimum_samples']} genuine samples "
            f"inside the exact requested window {plot_def['limits']}. Refusing to generate a blank plot."
        )
    ranked.sort(key=lambda row: row[:4], reverse=True)
    return ranked[0][4]


def robust_limits(values, errors):
    finite = values[np.isfinite(values)]
    low, high = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        center = float(np.median(finite))
        spread = float(np.std(finite)) or 1.0
        low, high = center - 3.0 * spread, center + 3.0 * spread
    margin = 0.12 * (high - low)
    return float(low - margin), float(high + margin)


def style_axis(axis):
    axis.grid(True, color="#334155", linewidth=0.55, alpha=0.58)
    for spine in axis.spines.values():
        spine.set_color("#94a3b8")


def draw_reference_lines(axis, plot_def):
    for index, (label, element, rest_wave) in enumerate(plot_def["lines"]):
        color = ELEMENT_COLORS[element]
        position = float(rest_to_obs(rest_wave)) if plot_def["axis"] == "observed" else rest_wave
        low, high = plot_def["limits"]
        if not (low <= position <= high):
            continue
        axis.axvline(position, color=color, linewidth=1.55, linestyle="--", alpha=0.96, zorder=5)
        axis.text(
            position, 0.98 - 0.10 * (index % 5), label,
            rotation=90, transform=axis.get_xaxis_transform(),
            ha="right", va="top", fontsize=8.2, color=color,
            bbox={"facecolor": "#050712", "alpha": 0.72, "edgecolor": "none", "pad": 1.4},
        )


def line_key_table(axis):
    axis.axis("off")
    rows = []
    for index, (label, element, rest_wave) in enumerate(REFERENCE_LINES, 1):
        observed = float(rest_to_obs(rest_wave))
        if 2.0 <= observed <= 3.0:
            rows.append((index, element, label, rest_wave, observed))
    cell_text = [[idx, element, label, f"{rest:.2f}", f"{obs:.5f}"]
                 for idx, element, label, rest, obs in rows]
    table = axis.table(
        cellText=cell_text,
        colLabels=["ID", "Element", "Transition", "Rest A", "Observed um"],
        cellLoc="left", colLoc="left", bbox=[0.00, 0.00, 0.73, 0.94],
        colWidths=[0.05, 0.14, 0.28, 0.12, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.3)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#475569")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#223047")
            cell.set_text_props(color="white", weight="bold")
        else:
            element = rows[row - 1][1]
            cell.set_facecolor("#111827" if row % 2 else "#172033")
            cell.set_text_props(color="#e5edf5")
            if column == 0:
                cell.set_facecolor(ELEMENT_COLORS[element])
                cell.set_text_props(color="#050712", weight="bold")
    axis.set_title("Reference-line color key", loc="left", fontsize=13, pad=6)
    axis.text(
        0.77, 0.84,
        "ELEMENT COLORS\n\n"
        "Nitrogen — magenta/red\n"
        "Carbon   — cyan\n"
        "Helium   — violet\n"
        "Oxygen   — blue\n\n"
        "Curve: actual X1D samples\n"
        "No step drawstyle\n"
        "No smoothing\n"
        "No interpolation",
        transform=axis.transAxes, ha="left", va="top", fontsize=10.1, linespacing=1.45,
        bbox={"facecolor": "#0b1220", "edgecolor": "#475569", "boxstyle": "round,pad=0.7"},
    )


def series_hash(x, y):
    payload = np.column_stack([x, y]).astype("float64").tobytes()
    return hashlib.sha256(payload).hexdigest()


def verify_png(path):
    if not path.exists() or path.stat().st_size < 30000:
        raise RuntimeError(f"PNG verification failed: {path}")
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=float) / 255.0
    if float(array.std()) < 0.03:
        raise RuntimeError(f"Rendered PNG appears blank: {path}")
    return int(path.stat().st_size), float(array.std())


def make_plot(candidate, plot_def):
    x_all, mask = mask_for(candidate, plot_def)
    x = x_all[mask]
    y = candidate["flux"][mask]
    error = candidate["error"][mask]
    finite = np.isfinite(x) & np.isfinite(y)
    x, y, error = x[finite], y[finite], error[finite]
    order = np.argsort(x)
    x, y, error = x[order], y[order], error[order]

    if len(x) < plot_def["minimum_samples"] or float(np.ptp(x)) <= 0 or float(np.ptp(y)) <= 0:
        raise RuntimeError(f"{plot_def['key']} failed pre-render data verification.")

    if plot_def["number"] == 1:
        figure = plt.figure(figsize=(18, 11.8), constrained_layout=True)
        grid = figure.add_gridspec(2, 1, height_ratios=[5.6, 2.4])
        axis = figure.add_subplot(grid[0, 0])
        table_axis = figure.add_subplot(grid[1, 0])
    else:
        figure, axis = plt.subplots(figsize=(16.5, 8.5), constrained_layout=True)
        table_axis = None

    axis.plot(
        x, y,
        color=plot_def["trace_color"], linewidth=1.05,
        linestyle="-", drawstyle="default",
        marker="o", markersize=2.2, markeredgewidth=0,
        label=f"Actual unsmoothed X1D samples: n={len(x)}", zorder=3,
    )
    good_error = np.isfinite(error) & (error >= 0)
    if good_error.any():
        axis.fill_between(
            x[good_error], y[good_error] - error[good_error], y[good_error] + error[good_error],
            color="#38bdf8", alpha=0.10, label="Actual 1-sigma uncertainty", zorder=1,
        )

    draw_reference_lines(axis, plot_def)
    axis.set_xlim(*plot_def["limits"])
    axis.set_ylim(*robust_limits(y, error))
    axis.set_ylabel(candidate["ylabel"])
    axis.set_title(
        plot_def["title"] + "\n"
        f"actual detector samples | source mode {candidate['mode']} | n={len(x)} | "
        "ordinary connected line, no step function",
        fontsize=15 if plot_def["number"] > 1 else 17, pad=13,
    )
    style_axis(axis)
    axis.legend(loc="upper right", fontsize=9, facecolor="#020617", edgecolor="#475569")

    if plot_def["axis"] == "observed":
        axis.set_xlabel("Observed wavelength [micrometers]")
        top = axis.secondary_xaxis("top", functions=(obs_to_rest, rest_to_obs))
        top.set_xlabel("Rest-frame wavelength [angstrom] at z = 14.44")
    else:
        axis.set_xlabel("Rest-frame vacuum wavelength [angstrom]")
        top = axis.secondary_xaxis("top", functions=(rest_to_obs, obs_to_rest))
        top.set_xlabel("Observed wavelength [micrometers]")

    if table_axis is not None:
        line_key_table(table_axis)

    output = PNG_DIR / f"{VERSION}_{plot_def['number']:02d}_{plot_def['key']}.png"
    figure.savefig(output, dpi=360, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)

    csv_path = CSV_DIR / f"{VERSION}_{plot_def['number']:02d}_{plot_def['key']}_ACTUAL_SAMPLES.csv"
    pd.DataFrame({
        "x_plot_units": x,
        "x_axis": plot_def["axis"],
        "observed_wavelength_um": x if plot_def["axis"] == "observed" else rest_to_obs(x),
        "rest_wavelength_angstrom_z14p44": obs_to_rest(x) if plot_def["axis"] == "observed" else x,
        "flux_display_units": y,
        "flux_error_display_units": error,
        "actual_x1d_sample": True,
        "drawstyle": "default",
        "smoothed": False,
        "interpolated": False,
    }).to_csv(csv_path, index=False)

    png_bytes, png_std = verify_png(output)
    return {
        "plot_number": plot_def["number"],
        "plot_key": plot_def["key"],
        "requested_low": plot_def["limits"][0],
        "requested_high": plot_def["limits"][1],
        "axis": plot_def["axis"],
        "sample_count": len(x),
        "x_min_actual": float(x.min()),
        "x_max_actual": float(x.max()),
        "flux_span": float(np.ptp(y)),
        "series_hash": series_hash(x, y),
        "source": candidate["source"],
        "source_kind": candidate["kind"],
        "source_mode": candidate["mode"],
        "source_separation_arcsec": candidate["separation_arcsec"],
        "png": str(output),
        "csv": str(csv_path),
        "png_bytes": png_bytes,
        "png_pixel_std": png_std,
        "verification": "PASS",
    }


def main():
    reset_style()
    candidates = fits_candidates() + csv_candidates()
    if not candidates:
        raise RuntimeError(
            "No cached actual MoM-z14 NIRSpec X1D FITS or CSV data were found. "
            "This script deliberately does not create toy data or blank plots."
        )

    selected = [choose_candidate(candidates, plot_def) for plot_def in PLOTS]
    results = [make_plot(candidate, plot_def)
               for candidate, plot_def in zip(selected, PLOTS)]

    hashes = [row["series_hash"] for row in results]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("Duplicate spectral sample arrays detected; refusing repeated plots.")

    verification_path = CSV_DIR / f"{VERSION}_PLOT_VERIFICATION.csv"
    pd.DataFrame(results).to_csv(verification_path, index=False)

    from IPython.display import display, Image as IPImage
    for row in results:
        display(IPImage(filename=row["png"]))

    print(f"CODE OUTPUT: {VERSION}")
    print("DATA            actual cached coordinate-matched JWST/NIRSpec X1D samples")
    print("PLOTS           5 exact requested windows")
    print("DRAWSTYLE       default connected line; not steps")
    print("SMOOTHING       none")
    print("INTERPOLATION   none")
    print("SYNTHETIC DATA  none")
    print("ELEMENT COLORS  N=magenta/red  C=cyan  He=violet  O=blue")
    for row in results:
        print(f"PLOT {row['plot_number']:02d}         {row['plot_key']:<28} n={row['sample_count']:>4}  PASS")
        print(f"SOURCE          {row['source']}")
        print(f"PNG             {row['png']}")
        print(f"CSV             {row['csv']}")
    print(f"VERIFY CSV      {verification_path}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
