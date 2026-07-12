#!/usr/bin/env python3
"""JWST_0057 — Two-row Gaussian Fourier lesson with no cross artifacts.

Column 3 uses the exact continuous Fourier transform of the circular Gaussian
models. This removes square-grid leakage and the distracting axis-aligned glow
that appeared in the discrete finite-image FFT display.
"""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def ensure_packages() -> None:
    missing = [name for name in ("numpy", "matplotlib")
               if importlib.util.find_spec(name) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_packages()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

VERSION = "JWST_0057"
ROOT = Path("/content/JWST_OUTPUT")
PNG = ROOT / "PNG"
CSV = ROOT / "CSV"
for directory in (PNG, CSV):
    directory.mkdir(parents=True, exist_ok=True)

N = 256
SIGMA_1 = 14.0
SIGMA_2 = 26.0
HIGHPASS_WEIGHT = 0.65


def gaussian_image(sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices((N, N), dtype=float)
    center = (N - 1) / 2.0
    xx -= center
    yy -= center
    image = np.exp(-0.5 * (xx * xx + yy * yy) / sigma**2)
    return image, xx, yy


def gaussian_ft_amplitude(radius_frequency: np.ndarray, sigma: float) -> np.ndarray:
    return 2.0 * np.pi * sigma**2 * np.exp(
        -2.0 * np.pi**2 * sigma**2 * radius_frequency**2
    )


def build_cases() -> list[dict]:
    narrow, _, _ = gaussian_image(SIGMA_1)
    broad, _, _ = gaussian_image(SIGMA_2)
    return [
        {
            "title": "Pure Gaussian lamp",
            "image": narrow,
            "note": "Exact circular Gaussian: Fourier power is a smooth circular peak.",
            "components": [(1.0, SIGMA_1)],
        },
        {
            "title": "High-pass Gaussian",
            "image": narrow - HIGHPASS_WEIGHT * broad,
            "note": "Difference of Gaussians: circular Fourier response with no axis glow.",
            "components": [(1.0, SIGMA_1), (-HIGHPASS_WEIGHT, SIGMA_2)],
        },
    ]


def analytic_power(components: list[tuple[float, float]]) -> dict:
    frequency = np.linspace(-0.5, 0.5, N, endpoint=False)
    fxx, fyy = np.meshgrid(frequency, frequency)
    radius = np.hypot(fxx, fyy)
    amplitude = np.zeros_like(radius)
    for weight, sigma in components:
        amplitude += weight * gaussian_ft_amplitude(radius, sigma)
    power = amplitude**2
    power /= max(float(power.max()), np.finfo(float).tiny)
    center = N // 2
    return {
        "frequency": frequency,
        "power": power,
        "center": center,
        "cross_section": power[center, :],
    }


def make_dashboard(records: list[dict]) -> Path:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 4, figsize=(21, 9.5), constrained_layout=True)

    for row, record in enumerate(records):
        image = record["image"]
        fourier = record["fourier"]
        center = N // 2

        ax = axes[row, 0]
        ax.imshow(image, origin="lower", cmap="gray", interpolation="nearest")
        ax.axhline(center, color="white", linewidth=0.45, alpha=0.55)
        ax.axvline(center, color="white", linewidth=0.45, alpha=0.55)
        ax.set_title(record["title"], fontsize=13)
        ax.text(0.02, -0.12, record["note"], transform=ax.transAxes,
                fontsize=9, va="top")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[row, 1]
        ax.plot(np.arange(N) - center, image[center, :], linewidth=1.8)
        ax.set_title("Image-space center line", fontsize=11)
        ax.set_xlabel("pixel offset")
        ax.set_ylabel("intensity")
        ax.grid(alpha=0.20)

        ax = axes[row, 2]
        extent = [-0.5, 0.5, -0.5, 0.5]
        ax.imshow(
            fourier["power"], origin="lower", cmap="magma",
            norm=LogNorm(vmin=1e-14, vmax=1.0), extent=extent,
            interpolation="nearest", aspect="equal",
        )
        ax.set_title("2-D Fourier power — radial reference", fontsize=11)
        ax.set_xlabel("fx [cycles/pixel]")
        ax.set_ylabel("fy [cycles/pixel]")

        frequency = np.abs(fourier["frequency"])
        order = np.argsort(frequency)
        ax = axes[row, 3]
        ax.plot(frequency[order], fourier["cross_section"][order], linewidth=1.6)
        ax.set_yscale("log")
        ax.set_xlim(0.0, 0.5)
        ax.set_ylim(1e-14, 1.2)
        ax.set_title("Fourier center cross-section", fontsize=11)
        ax.set_xlabel("|f| [cycles/pixel]")
        ax.set_ylabel("normalized power")
        ax.grid(alpha=0.20)

    fig.suptitle(
        "Toy Gaussian Fourier lesson — radial-clean revision\n"
        "Column 3 contains no guide lines and no square-grid axis artifacts",
        fontsize=18,
    )
    path = PNG / f"{VERSION}_TOY_GAUSSIAN_RADIAL_CLEAN_DASHBOARD.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return path


def make_csv(records: list[dict]) -> Path:
    path = CSV / f"{VERSION}_TOY_GAUSSIAN_RADIAL_CLEAN_SUMMARY.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "case", "image_sigma_pixels", "highpass_weight",
            "fourier_display", "column_3_crosshair_overlay",
        ])
        writer.writerow(["Pure Gaussian lamp", SIGMA_1, "", "analytic radial power", "none"])
        writer.writerow(["High-pass Gaussian", f"{SIGMA_1};{SIGMA_2}", HIGHPASS_WEIGHT,
                         "analytic radial power", "none"])
    return path


def main() -> None:
    records = []
    for case in build_cases():
        records.append({**case, "fourier": analytic_power(case["components"])})

    dashboard = make_dashboard(records)
    summary_csv = make_csv(records)

    print(f"CODE OUTPUT: {VERSION}")
    print("Purpose      Two-row Gaussian Fourier lesson without axis glow")
    print("Column 3     Exact radial Fourier power; no guides, markers, or cross artifacts")
    print(f"Plot PNG     {dashboard}")
    print(f"CSV          {summary_csv}")
    print(f"Timestamp    {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
