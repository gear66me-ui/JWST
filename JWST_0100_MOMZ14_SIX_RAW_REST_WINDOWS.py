#!/usr/bin/env python3
"""
JWST_0100_MOMZ14_SIX_RAW_REST_WINDOWS.py

Load the highest-resolution native MoM-z14 X1D CSV produced by JWST_0098 and
plot six independent rest-frame windows, each centered on a redshift feature
with +/-50 Angstrom coverage. No smoothing, interpolation, rebinning, fitting,
or synthetic spectra. Reference markers use six distinct group colors.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def ensure_packages() -> None:
    required = {"numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib"}
    missing = [pip_name for module, pip_name in required.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_packages()

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

VERSION = "JWST_0100"
Z = 14.44
ROOT = Path("/content/JWST_OUTPUT")
PNG = ROOT / "PNG"
CSV = ROOT / "CSV"
for directory in (PNG, CSV):
    directory.mkdir(parents=True, exist_ok=True)

SOURCE_VERSION = "JWST_0098"
SOURCE_RUNNER = "/content/JWST_0098_MOMZ14_MAST_MAX_NATIVE_SPECTRUM.py"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/JWST/main/"
    "JWST_0098_MOMZ14_MAST_MAX_NATIVE_SPECTRUM.py"
)

REGIONS = [
    {
        "key": "LYMAN_ALPHA_BREAK",
        "center_A": 1215.67,
        "title": "H I Lyman-alpha break",
        "color": "#ff9f1c",
        "lines": [("H I Ly-alpha", 1215.67)],
    },
    {
        "key": "N_IV",
        "center_A": 1486.50,
        "title": "N IV] doublet",
        "color": "#ff4fa3",
        "lines": [("N IV] 1483", 1483.32), ("N IV] 1487", 1486.50)],
    },
    {
        "key": "C_IV",
        "center_A": 1550.00,
        "title": "C IV doublet",
        "color": "#25c7f7",
        "lines": [("C IV 1548", 1548.20), ("C IV 1551", 1550.77)],
    },
    {
        "key": "HEII_OIII",
        "center_A": 1650.00,
        "title": "He II + O III] complex",
        "color": "#b978ff",
        "lines": [("He II 1640", 1640.42), ("O III] 1661", 1660.81), ("O III] 1666", 1666.15)],
    },
    {
        "key": "N_III",
        "center_A": 1750.00,
        "title": "N III] multiplet",
        "color": "#ef4444",
        "lines": [
            ("N III] 1747", 1746.82),
            ("N III] 1749", 1748.65),
            ("N III] 1750", 1749.67),
            ("N III] 1752", 1752.16),
            ("N III] 1754", 1753.99),
        ],
    },
    {
        "key": "C_III",
        "center_A": 1900.00,
        "title": "C III] doublet",
        "color": "#22c55e",
        "lines": [("C III] 1907", 1906.68), ("C III] 1909", 1908.73)],
    },
]


def reset_style() -> None:
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


def ensure_source_data() -> None:
    files = list(CSV.glob(f"{SOURCE_VERSION}_*_NATIVE_X1D.csv"))
    if files:
        return
    subprocess.check_call([
        "bash", "-lc",
        f"curl -fsSL -o {SOURCE_RUNNER} {SOURCE_URL} && python {SOURCE_RUNNER}"
    ])


def load_best_native_csv() -> tuple[pd.DataFrame, Path]:
    ensure_source_data()
    candidates = []
    for path in CSV.glob(f"{SOURCE_VERSION}_*_NATIVE_X1D.csv"):
        try:
            frame = pd.read_csv(path)
            if "observed_wavelength_um" not in frame or "flux_native_display_units" not in frame:
                continue
            sample_count = len(frame)
            grating = str(frame.get("grating", pd.Series([""])).iloc[0]).upper()
            nominal = 2700 if "H" in grating else 1000 if "M" in grating else 100 if "PRISM" in grating else 0
            separation = pd.to_numeric(frame.get("separation_arcsec", pd.Series([np.nan])), errors="coerce").iloc[0]
            candidates.append((nominal, sample_count, -float(separation) if np.isfinite(separation) else -9999.0, path, frame))
        except Exception:
            continue
    if not candidates:
        raise RuntimeError("No JWST_0098 native X1D CSV files were found.")
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    _, _, _, path, frame = candidates[0]
    return frame, path


def obs_um_to_rest_A(values) -> np.ndarray:
    return np.asarray(values, dtype=float) * 1.0e4 / (1.0 + Z)


def rest_A_to_obs_um(values) -> np.ndarray:
    return np.asarray(values, dtype=float) * (1.0 + Z) * 1.0e-4


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


def draw_region(region: dict, frame: pd.DataFrame, source_name: str, index: int) -> tuple[Path, dict, pd.DataFrame]:
    wave_obs = pd.to_numeric(frame["observed_wavelength_um"], errors="coerce").to_numpy(float)
    flux = pd.to_numeric(frame["flux_native_display_units"], errors="coerce").to_numpy(float)
    error = pd.to_numeric(frame.get("flux_error_native_display_units", np.nan), errors="coerce").to_numpy(float)
    dq = pd.to_numeric(frame.get("dq", 0), errors="coerce").fillna(0).to_numpy(int)
    rest_A = obs_um_to_rest_A(wave_obs)

    low = region["center_A"] - 50.0
    high = region["center_A"] + 50.0
    mask = np.isfinite(rest_A) & np.isfinite(flux) & (rest_A >= low) & (rest_A <= high)
    x = rest_A[mask]
    y = flux[mask]
    e = error[mask]
    q = dq[mask]
    obs = wave_obs[mask]
    if len(x) < 3:
        raise RuntimeError(f"{region['key']} contains only {len(x)} native samples in {low:.2f}-{high:.2f} A.")

    order = np.argsort(x)
    x, y, e, q, obs = x[order], y[order], e[order], q[order], obs[order]
    color = region["color"]

    fig, ax = plt.subplots(figsize=(17, 8.4), constrained_layout=True)
    valid_error = np.isfinite(e) & (e >= 0)
    if valid_error.any():
        ax.fill_between(x[valid_error], y[valid_error] - e[valid_error], y[valid_error] + e[valid_error],
                        alpha=0.12, linewidth=0, color=color, label="Native 1-sigma uncertainty")
    ax.plot(x, y, linewidth=0.62, color="#e7f4ff", alpha=0.98,
            label=f"Native X1D samples (n={len(x)})")
    ax.scatter(x, y, s=8, color=color, alpha=0.82, linewidths=0)
    bad = q != 0
    if bad.any():
        ax.scatter(x[bad], y[bad], s=20, facecolors="none", edgecolors="#f8fafc",
                   linewidths=0.6, alpha=0.75, label=f"DQ != 0 ({bad.sum()})")

    ymin, ymax = robust_limits(y)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(low, high)
    span = ymax - ymin
    for line_index, (label, wavelength) in enumerate(region["lines"]):
        ax.axvline(wavelength, color=color, linewidth=1.05, linestyle="--", alpha=0.96)
        text_y = ymax - (0.06 + 0.11 * (line_index % 4)) * span
        ax.annotate(f"{label}\n{wavelength:.2f} A",
                    xy=(wavelength, text_y - 0.025 * span), xytext=(wavelength, text_y),
                    ha="center", va="bottom", rotation=90, fontsize=8.2, color=color,
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.55},
                    bbox={"facecolor": "#050812", "edgecolor": "none", "alpha": 0.72, "pad": 1.0})

    ax.axhline(0.0, linewidth=0.45, alpha=0.55, color="#94a3b8")
    ax.grid(True, linewidth=0.42, alpha=0.38, color="#41546c")
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")
    ax.set_xlabel("Rest-frame vacuum wavelength [A]")
    ax.set_ylabel("Flux [native JWST_0098 display units]")
    ax.set_title(
        f"MoM-z14 — {region['title']}\n"
        f"center={region['center_A']:.2f} A | window={low:.2f}-{high:.2f} A | z={Z:.2f}",
        fontsize=14.5, pad=12,
    )
    top = ax.secondary_xaxis("top", functions=(rest_A_to_obs_um, obs_um_to_rest_A))
    top.set_xlabel("Observed wavelength [micron]")
    ax.legend(loc="best", fontsize=8.2, framealpha=0.88)

    output = PNG / f"{VERSION}_{index:02d}_{region['key']}_RAW_NATIVE.png"
    fig.suptitle(f"Source: {source_name} | no smoothing, interpolation, rebinning, or fitting", fontsize=10.5)
    fig.savefig(output, dpi=420, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    samples = pd.DataFrame({
        "region": region["key"],
        "rest_wavelength_A": x,
        "observed_wavelength_um": obs,
        "flux_native_display_units": y,
        "flux_error_native_display_units": e,
        "dq": q,
        "smoothed": False,
        "interpolated": False,
        "rebinned": False,
    })
    summary = {
        "region": region["key"],
        "center_A": region["center_A"],
        "low_A": low,
        "high_A": high,
        "samples": len(x),
        "reference_lines": len(region["lines"]),
        "color": color,
    }
    return output, summary, samples


def make_atlas(frame: pd.DataFrame, source_name: str) -> Path:
    wave_obs = pd.to_numeric(frame["observed_wavelength_um"], errors="coerce").to_numpy(float)
    flux = pd.to_numeric(frame["flux_native_display_units"], errors="coerce").to_numpy(float)
    rest_A = obs_um_to_rest_A(wave_obs)

    fig, axes = plt.subplots(6, 1, figsize=(19, 25), constrained_layout=True)
    for ax, region in zip(axes, REGIONS):
        low, high = region["center_A"] - 50.0, region["center_A"] + 50.0
        mask = np.isfinite(rest_A) & np.isfinite(flux) & (rest_A >= low) & (rest_A <= high)
        x, y = rest_A[mask], flux[mask]
        order = np.argsort(x)
        x, y = x[order], y[order]
        ax.plot(x, y, linewidth=0.56, color="#e7f4ff")
        ax.scatter(x, y, s=6, color=region["color"], alpha=0.78, linewidths=0)
        for label, wavelength in region["lines"]:
            ax.axvline(wavelength, color=region["color"], linewidth=0.9, linestyle="--", alpha=0.95)
        ax.set_xlim(low, high)
        ax.set_ylim(*robust_limits(y))
        ax.grid(True, linewidth=0.38, alpha=0.35, color="#41546c")
        ax.set_ylabel("Flux")
        ax.set_title(f"{region['title']} | {low:.1f}-{high:.1f} A | n={len(x)}", loc="left", fontsize=11.4)
    axes[-1].set_xlabel("Rest-frame vacuum wavelength [A]")
    fig.suptitle(
        "MoM-z14 — six native X1D redshift windows\n"
        f"Six-color reference groups | source={source_name} | z={Z:.2f}",
        fontsize=18,
    )
    output = PNG / f"{VERSION}_MOMZ14_SIX_REGION_RAW_NATIVE_ATLAS.png"
    fig.savefig(output, dpi=360, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return output


def main() -> None:
    reset_style()
    frame, source_path = load_best_native_csv()
    plot_paths = []
    summaries = []
    sample_frames = []
    for index, region in enumerate(REGIONS, 1):
        path, summary, samples = draw_region(region, frame, source_path.name, index)
        plot_paths.append(path)
        summaries.append(summary)
        sample_frames.append(samples)

    atlas_path = make_atlas(frame, source_path.name)
    summary_path = CSV / f"{VERSION}_SIX_REGION_SUMMARY.csv"
    samples_path = CSV / f"{VERSION}_SIX_REGION_NATIVE_SAMPLES.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    pd.concat(sample_frames, ignore_index=True).to_csv(samples_path, index=False)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"SOURCE CSV      {source_path}")
    print(f"REDSHIFT        {Z:.2f}")
    print("WINDOWS         six independent rest-frame regions, each +/-50 A")
    print("PROCESSING      no smoothing; no interpolation; no rebinning; no fitting")
    print(f"ATLAS PNG       {atlas_path}")
    for path in plot_paths:
        print(f"REGION PNG      {path}")
    print(f"SAMPLES CSV     {samples_path}")
    print(f"SUMMARY CSV     {summary_path}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
