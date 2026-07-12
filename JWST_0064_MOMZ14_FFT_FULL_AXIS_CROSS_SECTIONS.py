#!/usr/bin/env python3
"""
JWST_0064_MOMZ14_FFT_FULL_AXIS_CROSS_SECTIONS.py

Rebuild the MoM-z14 four-filter FFT dashboard with complete, signed central
cross-sections across the entire 2-D FFT square.

Unlike JWST_0052, no positive/negative folding is performed.

Rows
----
1. Calibrated NIRCam mosaic.
2. Full shifted 2-D FFT power square with center-spanning x/y crosshairs.
3. Complete horizontal FFT-axis cut from negative to positive Nyquist.
4. Complete vertical FFT-axis cut from negative to positive Nyquist.

No AI-generated imagery is used. FITS data are retrieved through the existing
JWST_0048/JWST_0051 workflow and all figures are rendered with Matplotlib.
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
        "astropy": "astropy",
        "requests": "requests",
    }
    missing = [pip_name for module, pip_name in required.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_packages()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

VERSION = "JWST_0064"
BASE_0051 = "JWST_0051_MOMZ14_FFT_RASTER_ARTIFACT_ANALYSIS.py"
BASE_URL = f"https://raw.githubusercontent.com/gear66me-ui/JWST/main/{BASE_0051}"
BASE_PATH = Path("/content") / BASE_0051

ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
FITS_DIR = ROOT / "FITS"
for directory in (PNG_DIR, CSV_DIR, FITS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

FREQ_MIN = 0.025
FREQ_MAX = 0.46
PERIOD_GUIDES_PIX = [32, 16, 8, 4, 2]


def load_analysis_module():
    urllib.request.urlretrieve(BASE_URL, BASE_PATH)
    spec = importlib.util.spec_from_file_location("jwst0051", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to import JWST_0051")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pixel_scale_arcsec(header) -> float:
    try:
        scales = np.asarray(proj_plane_pixel_scales(WCS(header).celestial)) * 3600.0
        scales = scales[np.isfinite(scales) & (scales > 0) & (scales < 10)]
        if scales.size:
            return float(np.median(scales))
    except Exception:
        pass
    pixar = header.get("PIXAR_A2")
    if pixar is not None and 0 < float(pixar) < 100:
        return float(np.sqrt(float(pixar)))
    raise RuntimeError("No plausible celestial pixel scale in FITS WCS")


def analyze_full_axis(windowed: np.ndarray) -> dict:
    ny, nx = windowed.shape
    fft_complex = np.fft.fftshift(np.fft.fft2(windowed))
    power = np.abs(fft_complex) ** 2

    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=1.0))
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=1.0))
    fxx, fyy = np.meshgrid(fx, fy)
    radius = np.hypot(fxx, fyy)

    annulus = (
        np.isfinite(power)
        & (radius >= FREQ_MIN)
        & (radius <= FREQ_MAX)
    )
    annular_median = max(
        float(np.nanmedian(power[annulus])),
        np.finfo(float).tiny,
    )
    normalized = power / annular_median

    cy = int(np.argmin(np.abs(fy)))
    cx = int(np.argmin(np.abs(fx)))

    # Exact, complete central cuts. Nothing is folded, cropped, averaged,
    # mirrored, or reduced to positive frequencies.
    horizontal_full = normalized[cy, :].copy()
    vertical_full = normalized[:, cx].copy()

    x_mask = (
        np.isfinite(horizontal_full)
        & (np.abs(fx) >= FREQ_MIN)
        & (np.abs(fx) <= FREQ_MAX)
    )
    y_mask = (
        np.isfinite(vertical_full)
        & (np.abs(fy) >= FREQ_MIN)
        & (np.abs(fy) <= FREQ_MAX)
    )
    x_peak_index = int(np.flatnonzero(x_mask)[np.argmax(horizontal_full[x_mask])])
    y_peak_index = int(np.flatnonzero(y_mask)[np.argmax(vertical_full[y_mask])])

    return {
        "fft_complex": fft_complex,
        "power": power,
        "normalized": normalized,
        "fx": fx,
        "fy": fy,
        "radius": radius,
        "cx": cx,
        "cy": cy,
        "horizontal_full": horizontal_full,
        "vertical_full": vertical_full,
        "x_peak_index": x_peak_index,
        "y_peak_index": y_peak_index,
        "x_peak_frequency": float(fx[x_peak_index]),
        "y_peak_frequency": float(fy[y_peak_index]),
        "x_peak_power": float(horizontal_full[x_peak_index]),
        "y_peak_power": float(vertical_full[y_peak_index]),
    }


def add_signed_period_guides(ax: plt.Axes) -> None:
    for period in PERIOD_GUIDES_PIX:
        frequency = 1.0 / period
        for sign in (-1.0, 1.0):
            x = sign * frequency
            ax.axvline(x, linewidth=0.55, linestyle="--", alpha=0.24)
        ax.text(
            frequency,
            0.97,
            f"{period}px",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="right",
            va="top",
            fontsize=7,
            alpha=0.70,
        )
        ax.text(
            -frequency,
            0.97,
            f"{period}px",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="left",
            va="top",
            fontsize=7,
            alpha=0.70,
        )


def positive_log_limits(values: np.ndarray) -> tuple[float, float]:
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return 1.0e-6, 1.0
    low = max(float(np.percentile(positive, 1.0)), np.finfo(float).tiny)
    high = float(np.percentile(positive, 99.8))
    if high <= low:
        high = low * 10.0
    return low, high


def make_dashboard(records: list[dict]) -> Path:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(4, 4, figsize=(23, 18), constrained_layout=True)

    for col, record in enumerate(records):
        entry = record["entry"]
        image = record["prep"]["image"]
        analysis = record["analysis"]
        scale = record["pixel_scale_arcsec"]

        image_lo, image_hi = np.nanpercentile(image, [1.0, 99.5])
        ax = axes[0, col]
        ax.imshow(
            image,
            origin="lower",
            cmap="gray",
            vmin=image_lo,
            vmax=image_hi,
            interpolation="nearest",
        )
        ax.set_title(f"{entry['name']} | {entry['lambda_um']:.2f} µm", fontsize=13)
        ax.set_xlabel(f"calibrated mosaic | {scale:.5f} arcsec/pixel", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        power = analysis["power"]
        positive = power[power > 0]
        fft_vmin = max(float(np.percentile(positive, 35.0)), np.finfo(float).tiny)
        fft_vmax = float(np.percentile(positive, 99.97))
        extent = [
            analysis["fx"][0], analysis["fx"][-1],
            analysis["fy"][0], analysis["fy"][-1],
        ]

        ax = axes[1, col]
        ax.imshow(
            power,
            origin="lower",
            cmap="magma",
            norm=LogNorm(vmin=fft_vmin, vmax=fft_vmax),
            extent=extent,
            interpolation="nearest",
            aspect="equal",
        )
        # These two lines span the complete FFT square from edge to edge.
        ax.plot([-0.5, 0.5], [0.0, 0.0], linewidth=1.7, label="full horizontal cut")
        ax.plot([0.0, 0.0], [-0.5, 0.5], linewidth=1.7, label="full vertical cut")
        ax.scatter(
            [analysis["x_peak_frequency"]], [0.0],
            s=28, zorder=4,
        )
        ax.scatter(
            [0.0], [analysis["y_peak_frequency"]],
            s=28, zorder=4,
        )
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("fx [cycles/pixel]", fontsize=9)
        ax.set_ylabel("fy [cycles/pixel]", fontsize=9)
        ax.set_title("Full 2-D FFT square + complete x/y cuts", fontsize=11)
        ax.grid(alpha=0.10)

        ax = axes[2, col]
        profile = analysis["horizontal_full"]
        profile_safe = np.maximum(profile, np.finfo(float).tiny)
        ax.plot(analysis["fx"], profile_safe, linewidth=1.35)
        ax.scatter(
            [analysis["x_peak_frequency"]], [analysis["x_peak_power"]],
            s=34, zorder=4,
        )
        add_signed_period_guides(ax)
        ylo, yhi = positive_log_limits(profile_safe)
        ax.set_yscale("log")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(ylo, yhi * 1.35)
        ax.set_xlabel("signed fx [cycles/pixel] — full negative-to-positive axis")
        ax.set_ylabel("power / annular median")
        ax.set_title(
            "Complete horizontal FFT-axis cross-section\n"
            f"peak f={analysis['x_peak_frequency']:+.5f} cyc/pix",
            fontsize=10,
        )
        ax.grid(alpha=0.18, which="both")

        ax = axes[3, col]
        profile = analysis["vertical_full"]
        profile_safe = np.maximum(profile, np.finfo(float).tiny)
        ax.plot(analysis["fy"], profile_safe, linewidth=1.35)
        ax.scatter(
            [analysis["y_peak_frequency"]], [analysis["y_peak_power"]],
            s=34, zorder=4,
        )
        add_signed_period_guides(ax)
        ylo, yhi = positive_log_limits(profile_safe)
        ax.set_yscale("log")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(ylo, yhi * 1.35)
        ax.set_xlabel("signed fy [cycles/pixel] — full negative-to-positive axis")
        ax.set_ylabel("power / annular median")
        ax.set_title(
            "Complete vertical FFT-axis cross-section\n"
            f"peak f={analysis['y_peak_frequency']:+.5f} cyc/pix",
            fontsize=10,
        )
        ax.grid(alpha=0.18, which="both")

    fig.suptitle(
        "MoM-z14 four-filter FFT — complete signed central cross-sections\n"
        "Rows 3 and 4 now traverse the entire FFT square from −Nyquist to +Nyquist; no folding or half-axis truncation",
        fontsize=18,
    )
    output = PNG_DIR / f"{VERSION}_MOMZ14_FFT_FULL_AXIS_CROSS_SECTIONS.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def make_summary_table(summary: pd.DataFrame) -> Path:
    shown = summary.copy()
    for column in [
        "pixel_scale_arcsec_per_pix",
        "x_peak_frequency_cycles_per_pix",
        "x_peak_period_pix",
        "y_peak_frequency_cycles_per_pix",
        "y_peak_period_pix",
    ]:
        shown[column] = shown[column].map(lambda value: f"{value:.6f}")
    shown = shown[[
        "filter",
        "pixel_scale_arcsec_per_pix",
        "x_peak_frequency_cycles_per_pix",
        "x_peak_period_pix",
        "y_peak_frequency_cycles_per_pix",
        "y_peak_period_pix",
    ]]
    shown.columns = [
        "Filter",
        "Scale [arcsec/pix]",
        "Horizontal-cut peak f",
        "Horizontal period [pix]",
        "Vertical-cut peak f",
        "Vertical period [pix]",
    ]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(17, 4.8), constrained_layout=True)
    ax.axis("off")
    table = ax.table(
        cellText=shown.values,
        colLabels=shown.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.75)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#536879")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#18344f")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#111923" if row % 2 else "#17212c")
            cell.set_text_props(color="#edf3f8")
    ax.set_title(
        "MoM-z14 complete signed FFT-axis peak summary\n"
        "Peak search excludes the central |f| < 0.025 cycles/pixel region",
        fontsize=15,
        pad=18,
    )
    output = PNG_DIR / f"{VERSION}_MOMZ14_FFT_FULL_AXIS_TABLE.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def main() -> None:
    module = load_analysis_module()
    base = module.load_base_module()
    base.VERSION = VERSION
    base.FITS_DIR = FITS_DIR
    session = base.build_session()

    records: list[dict] = []
    summary_rows: list[dict] = []
    profile_rows: list[dict] = []

    for entry in base.FILTERS:
        path, source_url, byte_count = base.download_channel(session, entry)
        data, header = base.load_image(path)
        scale = pixel_scale_arcsec(header)
        prep = module.preprocess(data)
        analysis = analyze_full_axis(prep["windowed"])

        records.append({
            "entry": entry,
            "path": path,
            "source_url": source_url,
            "download_bytes": byte_count,
            "pixel_scale_arcsec": scale,
            "prep": prep,
            "analysis": analysis,
        })

        x_frequency = analysis["x_peak_frequency"]
        y_frequency = analysis["y_peak_frequency"]
        summary_rows.append({
            "filter": entry["name"],
            "lambda_um": entry["lambda_um"],
            "pixel_scale_arcsec_per_pix": scale,
            "x_peak_frequency_cycles_per_pix": x_frequency,
            "x_peak_period_pix": 1.0 / abs(x_frequency),
            "x_peak_period_arcsec": scale / abs(x_frequency),
            "y_peak_frequency_cycles_per_pix": y_frequency,
            "y_peak_period_pix": 1.0 / abs(y_frequency),
            "y_peak_period_arcsec": scale / abs(y_frequency),
            "fits_path": str(path),
            "source_url": source_url,
        })

        for axis_name, frequency, power_values in [
            ("FULL_HORIZONTAL_FX", analysis["fx"], analysis["horizontal_full"]),
            ("FULL_VERTICAL_FY", analysis["fy"], analysis["vertical_full"]),
        ]:
            for f_value, p_value in zip(frequency, power_values):
                profile_rows.append({
                    "filter": entry["name"],
                    "axis_profile": axis_name,
                    "signed_frequency_cycles_per_pix": float(f_value),
                    "absolute_frequency_cycles_per_pix": float(abs(f_value)),
                    "period_pix": float(1.0 / abs(f_value)) if f_value != 0 else np.inf,
                    "power_over_annular_median": float(p_value),
                })

    summary = pd.DataFrame(summary_rows)
    profiles = pd.DataFrame(profile_rows)

    dashboard_path = make_dashboard(records)
    table_path = make_summary_table(summary)
    summary_csv = CSV_DIR / f"{VERSION}_MOMZ14_FFT_FULL_AXIS_SUMMARY.csv"
    profiles_csv = CSV_DIR / f"{VERSION}_MOMZ14_FFT_FULL_AXIS_PROFILES.csv"
    summary.to_csv(summary_csv, index=False)
    profiles.to_csv(profiles_csv, index=False)

    print(f"CODE OUTPUT: {VERSION}")
    print("Target         MoM-z14 calibrated NIRCam FITS mosaics")
    print("Correction     Full signed FFT cuts; no positive-half folding")
    print("Horizontal cut fy=0, traversing fx from negative to positive Nyquist")
    print("Vertical cut   fx=0, traversing fy from negative to positive Nyquist")
    print("Filter   X peak [cyc/pix]   X period [pix]   Y peak [cyc/pix]   Y period [pix]")
    for row in summary.itertuples(index=False):
        print(
            f"{row.filter:<7} {row.x_peak_frequency_cycles_per_pix:>+17.6f}"
            f" {row.x_peak_period_pix:>16.4f}"
            f" {row.y_peak_frequency_cycles_per_pix:>+18.6f}"
            f" {row.y_peak_period_pix:>16.4f}"
        )
    print(f"Plot PNG       {dashboard_path}")
    print(f"Table PNG      {table_path}")
    print(f"Summary CSV    {summary_csv}")
    print(f"Profiles CSV   {profiles_csv}")
    print(f"Timestamp      {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
