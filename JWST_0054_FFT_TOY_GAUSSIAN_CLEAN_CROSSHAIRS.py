#!/usr/bin/env python3
"""
JWST_0054_FFT_TOY_GAUSSIAN_CLEAN_CROSSHAIRS.py

Corrected toy FFT learning widget.

Changes from JWST_0053:
- Removes full-width guide lines from the FFT panels.
- Uses one small, identical center marker in every FFT panel.
- Replaces the hard rectangular crop with a soft Hann window so the third-row
  FFT is not dominated by a bright sinc cross and grid.
- Keeps all plots generated only from NumPy and Matplotlib.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def ensure_packages() -> None:
    required = {"numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib"}
    missing = [pip for module, pip in required.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_packages()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

VERSION = "JWST_0054"
ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
for directory in (PNG_DIR, CSV_DIR):
    directory.mkdir(parents=True, exist_ok=True)

N = 256
SIGMA_PIX = 14.0
HIGHPASS_SIGMA_PIX = 26.0


def gaussian2d(n: int, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices((n, n), dtype=float)
    center = (n - 1) / 2.0
    xx -= center
    yy -= center
    image = np.exp(-0.5 * (xx * xx + yy * yy) / (sigma * sigma))
    return image, xx, yy


def build_cases() -> dict:
    base, xx, yy = gaussian2d(N, SIGMA_PIX)
    radius = np.hypot(xx, yy)
    broad = np.exp(-0.5 * (radius / HIGHPASS_SIGMA_PIX) ** 2)
    hann_2d = np.outer(np.hanning(N), np.hanning(N))

    return {
        "PURE": {
            "title": "Pure Gaussian lamp",
            "image": base,
            "note": "Ideal Gaussian source: smooth Gaussian image and smooth central FFT power.",
        },
        "HIGHPASS": {
            "title": "High-pass Gaussian",
            "image": base - 0.65 * broad / broad.max(),
            "note": "Broad structure is subtracted, suppressing the lowest spatial frequencies.",
        },
        "SOFT_WINDOW": {
            "title": "Soft-windowed Gaussian",
            "image": base * hann_2d,
            "note": "A Hann window tapers the edges smoothly and avoids the hard-crop sinc cross.",
        },
    }


def fft_products(image: np.ndarray) -> dict:
    fft = np.fft.fftshift(np.fft.fft2(image))
    amplitude = np.abs(fft)
    power = amplitude ** 2
    fx = np.fft.fftshift(np.fft.fftfreq(image.shape[1]))
    fy = np.fft.fftshift(np.fft.fftfreq(image.shape[0]))
    cy = image.shape[0] // 2
    cx = image.shape[1] // 2
    return {
        "power": power,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "image_cut": image[cy, :],
        "x_power": power[cy, :],
        "y_power": power[:, cx],
    }


def radial_profile(power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices(power.shape, dtype=float)
    cy = (power.shape[0] - 1) / 2.0
    cx = (power.shape[1] - 1) / 2.0
    rr = np.floor(np.hypot(xx - cx, yy - cy)).astype(int)
    radius = np.arange(rr.max() + 1, dtype=float)
    profile = np.full(radius.shape, np.nan)
    for k in range(len(radius)):
        mask = rr == k
        if np.any(mask):
            profile[k] = np.mean(power[mask])
    return radius, profile


def add_small_center_marker(ax) -> None:
    ax.plot(0.0, 0.0, marker="+", markersize=9, markeredgewidth=0.8,
            color="white", alpha=0.9, linestyle="None", zorder=10)


def make_overview(results: dict) -> Path:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 4, figsize=(21, 14), constrained_layout=True)

    for row, payload in enumerate(results.values()):
        image = payload["image"]
        product = payload["product"]
        fx = product["fx"]
        fy = product["fy"]

        ax = axes[row, 0]
        ax.imshow(image, origin="lower", cmap="gray", interpolation="nearest")
        ax.axhline(product["cy"], linewidth=0.45, alpha=0.45, color="white")
        ax.axvline(product["cx"], linewidth=0.45, alpha=0.45, color="white")
        ax.set_title(payload["title"], fontsize=13)
        ax.text(0.02, -0.12, payload["note"], transform=ax.transAxes,
                fontsize=9, va="top")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[row, 1]
        ax.plot(np.arange(N) - product["cx"], product["image_cut"], linewidth=1.8)
        ax.set_title("Image-space center line", fontsize=11)
        ax.set_xlabel("pixel offset")
        ax.set_ylabel("intensity")
        ax.grid(alpha=0.22)

        positive = product["power"][product["power"] > 0]
        vmin = max(float(np.percentile(positive, 20.0)), np.finfo(float).tiny)
        vmax = float(np.percentile(positive, 99.95))
        extent = [fx[0], fx[-1], fy[0], fy[-1]]
        ax = axes[row, 2]
        ax.imshow(product["power"], origin="lower", cmap="magma",
                  norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent,
                  interpolation="nearest", aspect="equal")
        add_small_center_marker(ax)
        ax.set_title("2-D FFT power — clean center marker", fontsize=11)
        ax.set_xlabel("fx [cycles/pixel]")
        ax.set_ylabel("fy [cycles/pixel]")

        frequency = np.abs(fx)
        order = np.argsort(frequency)
        ax = axes[row, 3]
        ax.plot(frequency[order], product["x_power"][order], linewidth=1.6)
        ax.set_yscale("log")
        ax.set_xlim(0, 0.5)
        ax.set_title("FFT power cut through center", fontsize=11)
        ax.set_xlabel("|f| [cycles/pixel]")
        ax.set_ylabel("power")
        ax.grid(alpha=0.22)

    fig.suptitle(
        "Toy Gaussian FFT demo — corrected clean markers\n"
        "The small white plus is only a display marker and is not part of the FFT data",
        fontsize=18,
    )
    output = PNG_DIR / f"{VERSION}_TOY_GAUSSIAN_CLEAN_CROSSHAIRS.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def make_radial_plot(results: dict) -> Path:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for payload in results.values():
        radius, profile = radial_profile(payload["product"]["power"])
        valid = (radius > 0) & np.isfinite(profile)
        ax.plot(radius[valid], profile[valid], linewidth=1.8,
                label=payload["title"])
    ax.set_yscale("log")
    ax.set_xlabel("radius from FFT center [Fourier-plane pixels]")
    ax.set_ylabel("mean power")
    ax.set_title("Radial FFT power comparison", fontsize=15)
    ax.grid(alpha=0.22)
    ax.legend()
    output = PNG_DIR / f"{VERSION}_TOY_GAUSSIAN_RADIAL_POWER.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def make_summary(results: dict) -> tuple[Path, Path]:
    rows = []
    for payload in results.values():
        product = payload["product"]
        frequency = np.abs(product["fx"])
        order = np.argsort(frequency)
        frequency = frequency[order]
        power = product["x_power"][order]
        valid = frequency > 0
        index = int(np.argmax(power[valid]))
        peak_frequency = float(frequency[valid][index])
        rows.append({
            "case": payload["title"],
            "dominant_frequency_cycles_per_pixel": peak_frequency,
            "equivalent_period_pixels": 1.0 / peak_frequency,
            "center_power": float(product["x_power"][product["cx"]]),
            "note": payload["note"],
        })

    frame = pd.DataFrame(rows)
    csv_path = CSV_DIR / f"{VERSION}_TOY_GAUSSIAN_SUMMARY.csv"
    frame.to_csv(csv_path, index=False)

    shown = frame.copy()
    shown["dominant_frequency_cycles_per_pixel"] = shown[
        "dominant_frequency_cycles_per_pixel"
    ].map(lambda value: f"{value:.5f}")
    shown["equivalent_period_pixels"] = shown[
        "equivalent_period_pixels"
    ].map(lambda value: f"{value:.2f}")
    shown["center_power"] = shown["center_power"].map(lambda value: f"{value:.6g}")
    shown.columns = ["Case", "Dominant f [cyc/pix]", "Equivalent period [pix]",
                     "Center power", "Interpretation"]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(18, 4.8), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=shown.values, colLabels=shown.columns,
                     cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.85)
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
        "Corrected toy FFT summary\n"
        "All FFT panels use the same small center marker; no full-width guide lines are drawn",
        fontsize=15, pad=16,
    )
    png_path = PNG_DIR / f"{VERSION}_TOY_GAUSSIAN_SUMMARY_TABLE.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return csv_path, png_path


def main() -> None:
    cases = build_cases()
    results = {
        key: {**payload, "product": fft_products(np.asarray(payload["image"], dtype=float))}
        for key, payload in cases.items()
    }

    overview = make_overview(results)
    radial = make_radial_plot(results)
    summary_csv, summary_png = make_summary(results)

    print(f"CODE OUTPUT: {VERSION}")
    print("Correction   removed full-width FFT guide lines")
    print("Marker       identical small white plus at FFT center in all three rows")
    print("Row 3        soft Hann window replaces the hard rectangular crop")
    print(f"Plot PNG     {overview}")
    print(f"Radial PNG   {radial}")
    print(f"Table PNG    {summary_png}")
    print(f"CSV          {summary_csv}")
    print(f"Timestamp    {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
