#!/usr/bin/env python3
"""
JWST_0055_FFT_TOY_GAUSSIAN_2ROW_CLEAN.py

Two-row toy FFT learning dashboard:
1) pure Gaussian lamp,
2) high-pass Gaussian.

The previous hard-windowed third row is removed. FFT panels use only very fine
center guide lines with no glow, halo, or thick crosshair overlay.
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

VERSION = "JWST_0055"
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


def build_cases() -> list[dict]:
    base, xx, yy = gaussian2d(N, SIGMA_PIX)
    radius = np.hypot(xx, yy)
    broad = np.exp(-0.5 * (radius / HIGHPASS_SIGMA_PIX) ** 2)
    return [
        {
            "name": "PURE_GAUSSIAN",
            "title": "Pure Gaussian lamp",
            "image": base,
            "note": "Smooth hotspot: compact smooth Fourier peak at zero frequency.",
        },
        {
            "name": "HIGHPASS_GAUSSIAN",
            "title": "High-pass Gaussian",
            "image": base - 0.65 * broad / broad.max(),
            "note": "Broad structure removed: low-frequency center is suppressed and may form a ring.",
        },
    ]


def fft_products(image: np.ndarray) -> dict:
    fft = np.fft.fftshift(np.fft.fft2(image))
    power = np.abs(fft) ** 2
    fx = np.fft.fftshift(np.fft.fftfreq(image.shape[1]))
    fy = np.fft.fftshift(np.fft.fftfreq(image.shape[0]))
    cy, cx = image.shape[0] // 2, image.shape[1] // 2
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
    radius_index = np.floor(np.hypot(xx - cx, yy - cy)).astype(int)
    radius = np.arange(radius_index.max() + 1, dtype=float)
    values = np.full(radius.shape, np.nan)
    for index in range(len(radius)):
        mask = radius_index == index
        if np.any(mask):
            values[index] = np.mean(power[mask])
    return radius, values


def make_dashboard(results: list[dict]) -> Path:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 4, figsize=(21, 9.5), constrained_layout=True)

    for row, record in enumerate(results):
        image = record["image"]
        product = record["fft"]
        fx, fy = product["fx"], product["fy"]

        ax = axes[row, 0]
        ax.imshow(image, origin="lower", cmap="gray", interpolation="nearest")
        ax.axhline(product["cy"], color="white", linewidth=0.45, alpha=0.55)
        ax.axvline(product["cx"], color="white", linewidth=0.45, alpha=0.55)
        ax.set_title(record["title"], fontsize=13)
        ax.text(0.02, -0.12, record["note"], transform=ax.transAxes,
                fontsize=9, va="top")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[row, 1]
        ax.plot(np.arange(N) - product["cx"], product["image_cut"], linewidth=1.8)
        ax.set_title("Image-space center line", fontsize=11)
        ax.set_xlabel("pixel offset")
        ax.set_ylabel("intensity")
        ax.grid(alpha=0.20)

        positive = product["power"][product["power"] > 0]
        vmin = max(float(np.percentile(positive, 8.0)), np.finfo(float).tiny)
        vmax = float(np.percentile(positive, 99.98))
        extent = [fx[0], fx[-1], fy[0], fy[-1]]
        ax = axes[row, 2]
        ax.imshow(product["power"], origin="lower", cmap="magma",
                  norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent,
                  interpolation="nearest", aspect="equal")
        ax.axhline(0.0, color="white", linewidth=0.35, alpha=0.42, zorder=5)
        ax.axvline(0.0, color="white", linewidth=0.35, alpha=0.42, zorder=5)
        ax.set_title("2-D FFT power — fine center guides", fontsize=11)
        ax.set_xlabel("fx [cycles/pixel]")
        ax.set_ylabel("fy [cycles/pixel]")

        frequency = np.abs(fx)
        order = np.argsort(frequency)
        ax = axes[row, 3]
        ax.plot(frequency[order], product["x_power"][order], linewidth=1.6)
        ax.set_yscale("log")
        ax.set_xlim(0, 0.5)
        ax.set_title("FFT center cross-section", fontsize=11)
        ax.set_xlabel("|f| [cycles/pixel]")
        ax.set_ylabel("power")
        ax.grid(alpha=0.20)

    fig.suptitle(
        "Toy Gaussian FFT demo — two-row clean revision\n"
        "Third row removed; Column 3 uses only thin, non-glowing center guides",
        fontsize=18,
    )
    output = PNG_DIR / f"{VERSION}_TOY_GAUSSIAN_2ROW_CLEAN_DASHBOARD.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def make_radial_plot(results: list[dict]) -> Path:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for record in results:
        radius, power = radial_profile(record["fft"]["power"])
        mask = (radius > 0) & np.isfinite(power)
        ax.plot(radius[mask], power[mask], linewidth=1.8, label=record["title"])
    ax.set_yscale("log")
    ax.set_xlabel("radius from FFT center [Fourier-plane pixels]")
    ax.set_ylabel("mean power")
    ax.set_title("Radial FFT power comparison", fontsize=15)
    ax.grid(alpha=0.20)
    ax.legend()
    output = PNG_DIR / f"{VERSION}_TOY_GAUSSIAN_2ROW_RADIAL.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def make_summary(results: list[dict]) -> tuple[Path, Path]:
    rows = []
    for record in results:
        product = record["fft"]
        frequency = np.abs(product["fx"])
        order = np.argsort(frequency)
        frequency = frequency[order]
        power = product["x_power"][order]
        mask = frequency > 0
        index = int(np.argmax(power[mask]))
        peak_frequency = float(frequency[mask][index])
        rows.append({
            "case": record["title"],
            "dominant_frequency_cycles_per_pixel": peak_frequency,
            "equivalent_period_pixels": 1.0 / peak_frequency,
            "center_power": float(product["x_power"][product["cx"]]),
            "interpretation": record["note"],
        })

    dataframe = pd.DataFrame(rows)
    csv_path = CSV_DIR / f"{VERSION}_TOY_GAUSSIAN_2ROW_SUMMARY.csv"
    dataframe.to_csv(csv_path, index=False)

    shown = dataframe.copy()
    shown["dominant_frequency_cycles_per_pixel"] = shown[
        "dominant_frequency_cycles_per_pixel"
    ].map(lambda value: f"{value:.5f}")
    shown["equivalent_period_pixels"] = shown[
        "equivalent_period_pixels"
    ].map(lambda value: f"{value:.2f}")
    shown["center_power"] = shown["center_power"].map(lambda value: f"{value:.6g}")
    shown.columns = [
        "Case", "Dominant f [cyc/pix]", "Equivalent period [pix]",
        "Center power", "Interpretation",
    ]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(18, 4.1), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=shown.values, colLabels=shown.columns,
                     cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#536879")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#18344f")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#111923" if row % 2 else "#17212c")
            cell.set_text_props(color="#edf3f8")
    ax.set_title("Toy Gaussian FFT two-row summary", fontsize=15, pad=16)
    png_path = PNG_DIR / f"{VERSION}_TOY_GAUSSIAN_2ROW_TABLE.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return csv_path, png_path


def main() -> None:
    results = []
    for case in build_cases():
        image = np.asarray(case["image"], dtype=float)
        results.append({**case, "fft": fft_products(image)})

    dashboard = make_dashboard(results)
    radial = make_radial_plot(results)
    summary_csv, table_png = make_summary(results)

    print(f"CODE OUTPUT: {VERSION}")
    print("Purpose      Two-row toy Gaussian FFT learning dashboard")
    print("Change       Removed third row entirely")
    print("Guides       Thin non-glowing lines only in Column 3")
    print(f"Plot PNG     {dashboard}")
    print(f"Radial PNG   {radial}")
    print(f"Table PNG    {table_png}")
    print(f"CSV          {summary_csv}")
    print(f"Timestamp    {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
