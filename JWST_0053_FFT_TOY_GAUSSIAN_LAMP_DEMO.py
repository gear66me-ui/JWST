#!/usr/bin/env python3
"""
JWST_0053_FFT_TOY_GAUSSIAN_LAMP_DEMO.py

Toy 2-D FFT learning module for the JWST workflow.
Creates three computed images only:
1) pure Gaussian lamp,
2) high-pass Gaussian,
3) hard-windowed Gaussian.

Shows image-space line cuts, 2-D FFT power, FFT crosshair profiles,
radial power, a styled summary table, CSV output, and concise terminal output.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def ensure_packages() -> None:
    req = {"numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib"}
    missing = [pip for mod, pip in req.items() if importlib.util.find_spec(mod) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_packages()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

VERSION = "JWST_0053"
ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
for d in (PNG_DIR, CSV_DIR):
    d.mkdir(parents=True, exist_ok=True)

N = 256
SIGMA_PIX = 14.0
HIGHPASS_SIGMA_PIX = 26.0
WINDOW_HALF_WIDTH_PIX = 70


def gaussian2d(n: int, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices((n, n), dtype=float)
    c = (n - 1) / 2.0
    xx -= c
    yy -= c
    image = np.exp(-0.5 * (xx * xx + yy * yy) / (sigma * sigma))
    return image, xx, yy


def fft_products(image: np.ndarray) -> dict:
    fft = np.fft.fftshift(np.fft.fft2(image))
    amp = np.abs(fft)
    power = amp ** 2
    fx = np.fft.fftshift(np.fft.fftfreq(image.shape[1]))
    fy = np.fft.fftshift(np.fft.fftfreq(image.shape[0]))
    cy, cx = image.shape[0] // 2, image.shape[1] // 2
    return {
        "amp": amp,
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
    values = np.full(radius.shape, np.nan)
    for k in range(len(radius)):
        q = rr == k
        if np.any(q):
            values[k] = np.mean(power[q])
    return radius, values


def build_cases() -> dict:
    base, xx, yy = gaussian2d(N, SIGMA_PIX)
    rr = np.hypot(xx, yy)
    broad = np.exp(-0.5 * (rr / HIGHPASS_SIGMA_PIX) ** 2)
    box = ((np.abs(xx) <= WINDOW_HALF_WIDTH_PIX) &
           (np.abs(yy) <= WINDOW_HALF_WIDTH_PIX)).astype(float)
    return {
        "PURE_GAUSSIAN": {
            "title": "Pure Gaussian lamp",
            "image": base,
            "note": "Gaussian image -> Gaussian Fourier amplitude and Gaussian-like power.",
        },
        "HIGHPASS_GAUSSIAN": {
            "title": "High-pass Gaussian",
            "image": base - 0.65 * broad / broad.max(),
            "note": "Low frequencies are suppressed; the FFT center can dip or become ring-like.",
        },
        "WINDOWED_GAUSSIAN": {
            "title": "Hard-windowed Gaussian",
            "image": base * box,
            "note": "Sharp crop edges introduce sinc-like wings and ripples in Fourier space.",
        },
    }


def make_overview(results: dict) -> Path:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 4, figsize=(21, 14), constrained_layout=True)

    for row, payload in enumerate(results.values()):
        image = payload["image"]
        prod = payload["prod"]
        fx, fy = prod["fx"], prod["fy"]

        ax = axes[row, 0]
        ax.imshow(image, origin="lower", cmap="gray", interpolation="nearest")
        ax.axhline(prod["cy"], linewidth=0.8)
        ax.axvline(prod["cx"], linewidth=0.8)
        ax.set_title(payload["title"], fontsize=13)
        ax.text(0.02, -0.12, payload["note"], transform=ax.transAxes,
                fontsize=9, va="top")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[row, 1]
        ax.plot(np.arange(N) - prod["cx"], prod["image_cut"], linewidth=1.8)
        ax.set_title("Image-space center line", fontsize=11)
        ax.set_xlabel("pixel offset")
        ax.set_ylabel("intensity")
        ax.grid(alpha=0.22)

        positive = prod["power"][prod["power"] > 0]
        vmin = max(float(np.percentile(positive, 5.0)), np.finfo(float).tiny)
        vmax = float(np.percentile(positive, 99.98))
        extent = [fx[0], fx[-1], fy[0], fy[-1]]
        ax = axes[row, 2]
        ax.imshow(prod["power"], origin="lower", cmap="magma",
                  norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent,
                  interpolation="nearest", aspect="equal")
        ax.axhline(0.0, linewidth=1.0)
        ax.axvline(0.0, linewidth=1.0)
        ax.set_title("2-D FFT power", fontsize=11)
        ax.set_xlabel("fx [cycles/pixel]")
        ax.set_ylabel("fy [cycles/pixel]")

        freq = np.abs(fx)
        order = np.argsort(freq)
        ax = axes[row, 3]
        ax.plot(freq[order], prod["x_power"][order], linewidth=1.6)
        ax.set_yscale("log")
        ax.set_xlim(0, 0.5)
        ax.set_title("FFT crosshair through center", fontsize=11)
        ax.set_xlabel("|f| [cycles/pixel]")
        ax.set_ylabel("power")
        ax.grid(alpha=0.22)

    fig.suptitle(
        "Toy Gaussian hotspot FFT demo\n"
        "Pure Gaussian gives a smooth central Fourier peak; filtering and hard edges add structure",
        fontsize=18,
    )
    out = PNG_DIR / f"{VERSION}_TOY_GAUSSIAN_FFT_OVERVIEW.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return out


def make_radial_plot(results: dict) -> Path:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for payload in results.values():
        radius, power = radial_profile(payload["prod"]["power"])
        q = (radius > 0) & np.isfinite(power)
        ax.plot(radius[q], power[q], linewidth=1.8, label=payload["title"])
    ax.set_yscale("log")
    ax.set_xlabel("radius from FFT center [Fourier-plane pixels]")
    ax.set_ylabel("mean power")
    ax.set_title("Radial FFT power comparison", fontsize=15)
    ax.grid(alpha=0.22)
    ax.legend()
    out = PNG_DIR / f"{VERSION}_TOY_GAUSSIAN_FFT_RADIAL.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return out


def make_summary(results: dict) -> tuple[pd.DataFrame, Path, Path]:
    rows = []
    for payload in results.values():
        prod = payload["prod"]
        freq = np.abs(prod["fx"])
        order = np.argsort(freq)
        freq = freq[order]
        power = prod["x_power"][order]
        q = freq > 0
        peak = int(np.argmax(power[q]))
        f_peak = float(freq[q][peak])
        rows.append({
            "case": payload["title"],
            "dominant_frequency_cycles_per_pixel": f_peak,
            "equivalent_period_pixels": 1.0 / f_peak,
            "center_power": float(prod["x_power"][prod["cx"]]),
            "interpretation": payload["note"],
        })

    df = pd.DataFrame(rows)
    csv_path = CSV_DIR / f"{VERSION}_TOY_GAUSSIAN_FFT_SUMMARY.csv"
    df.to_csv(csv_path, index=False)

    shown = df.copy()
    shown["dominant_frequency_cycles_per_pixel"] = shown[
        "dominant_frequency_cycles_per_pixel"
    ].map(lambda x: f"{x:.5f}")
    shown["equivalent_period_pixels"] = shown[
        "equivalent_period_pixels"
    ].map(lambda x: f"{x:.2f}")
    shown["center_power"] = shown["center_power"].map(lambda x: f"{x:.6g}")
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
        "Toy Gaussian FFT summary\n"
        "Brightness controls Fourier power; spatial scale controls distance from the FFT center",
        fontsize=15, pad=16,
    )
    png_path = PNG_DIR / f"{VERSION}_TOY_GAUSSIAN_FFT_TABLE.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return df, csv_path, png_path


def main() -> None:
    cases = build_cases()
    results = {}
    for key, payload in cases.items():
        results[key] = {
            **payload,
            "prod": fft_products(np.asarray(payload["image"], dtype=float)),
        }

    overview = make_overview(results)
    radial = make_radial_plot(results)
    _, summary_csv, table_png = make_summary(results)

    print(f"CODE OUTPUT: {VERSION}")
    print("Purpose      Toy Gaussian lamp FFT learning demo")
    print("Cases        pure Gaussian | high-pass Gaussian | hard-windowed Gaussian")
    print("Key idea     Fourier radius tracks spatial frequency, not brightness alone")
    print(f"Plot PNG     {overview}")
    print(f"Radial PNG   {radial}")
    print(f"Table PNG    {table_png}")
    print(f"CSV          {summary_csv}")
    print(f"Timestamp    {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
