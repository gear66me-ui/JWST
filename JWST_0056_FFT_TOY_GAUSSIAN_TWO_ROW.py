#!/usr/bin/env python3
"""
JWST_0056_FFT_TOY_GAUSSIAN_TWO_ROW.py

Two-row toy FFT learning dashboard:
1) pure Gaussian lamp,
2) high-pass Gaussian.

The hard-windowed third case has been removed. Column 3 keeps only thin,
non-glowing reference lines at fx=0 and fy=0.
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
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

VERSION = "JWST_0056"
ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
for directory in (PNG_DIR, CSV_DIR):
    directory.mkdir(parents=True, exist_ok=True)

N = 256
SIGMA_PIX = 14.0
HIGHPASS_SIGMA_PIX = 26.0
GUIDE_COLOR = "#d8dde6"
GUIDE_LW = 0.45
GUIDE_ALPHA = 0.55


def gaussian2d(n: int, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices((n, n), dtype=float)
    center = (n - 1) / 2.0
    xx -= center
    yy -= center
    image = np.exp(-0.5 * (xx * xx + yy * yy) / (sigma * sigma))
    return image, xx, yy


def build_cases() -> dict:
    base, xx, yy = gaussian2d(N, SIGMA_PIX)
    rr = np.hypot(xx, yy)
    broad = np.exp(-0.5 * (rr / HIGHPASS_SIGMA_PIX) ** 2)
    return {
        "PURE_GAUSSIAN": {
            "title": "Pure Gaussian lamp",
            "image": base,
            "note": "Single smooth hotspot: compact smooth central FFT power peak.",
        },
        "HIGHPASS_GAUSSIAN": {
            "title": "High-pass Gaussian",
            "image": base - 0.65 * broad / broad.max(),
            "note": "Broad structure removed: low-frequency power is suppressed and the center can ring.",
        },
    }


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
    }


def make_dashboard(results: dict) -> Path:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 4, figsize=(21, 10), constrained_layout=True)

    for row, payload in enumerate(results.values()):
        image = payload["image"]
        prod = payload["prod"]
        fx, fy = prod["fx"], prod["fy"]

        ax = axes[row, 0]
        ax.imshow(image, origin="lower", cmap="gray", interpolation="nearest")
        ax.axhline(prod["cy"], color=GUIDE_COLOR, linewidth=GUIDE_LW, alpha=GUIDE_ALPHA)
        ax.axvline(prod["cx"], color=GUIDE_COLOR, linewidth=GUIDE_LW, alpha=GUIDE_ALPHA)
        ax.set_title(payload["title"], fontsize=13)
        ax.text(0.02, -0.13, payload["note"], transform=ax.transAxes, fontsize=9, va="top")
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
        ax.axhline(0.0, color=GUIDE_COLOR, linewidth=GUIDE_LW, alpha=GUIDE_ALPHA)
        ax.axvline(0.0, color=GUIDE_COLOR, linewidth=GUIDE_LW, alpha=GUIDE_ALPHA)
        ax.set_title("2-D FFT power (fine guide lines)", fontsize=11)
        ax.set_xlabel("fx [cycles/pixel]")
        ax.set_ylabel("fy [cycles/pixel]")

        freq = np.abs(fx)
        order = np.argsort(freq)
        ax = axes[row, 3]
        ax.plot(freq[order], prod["x_power"][order], linewidth=1.6)
        ax.set_yscale("log")
        ax.set_xlim(0, 0.5)
        ax.set_title("FFT center cross-section", fontsize=11)
        ax.set_xlabel("|f| [cycles/pixel]")
        ax.set_ylabel("power")
        ax.grid(alpha=0.22)

    fig.suptitle(
        "Toy Gaussian hotspot FFT demo — two-row revision\n"
        "Pure Gaussian versus high-pass Gaussian; hard-windowed third case removed",
        fontsize=18,
    )
    output = PNG_DIR / f"{VERSION}_TOY_GAUSSIAN_FFT_TWO_ROW_DASHBOARD.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def make_summary(results: dict) -> tuple[Path, Path]:
    rows = []
    for payload in results.values():
        prod = payload["prod"]
        freq = np.abs(prod["fx"])
        order = np.argsort(freq)
        freq = freq[order]
        power = prod["x_power"][order]
        mask = freq > 0
        peak_index = int(np.argmax(power[mask]))
        peak_frequency = float(freq[mask][peak_index])
        rows.append({
            "case": payload["title"],
            "dominant_frequency_cycles_per_pixel": peak_frequency,
            "equivalent_period_pixels": 1.0 / peak_frequency,
            "center_power": float(prod["x_power"][prod["cx"]]),
            "interpretation": payload["note"],
        })

    dataframe = pd.DataFrame(rows)
    csv_path = CSV_DIR / f"{VERSION}_TOY_GAUSSIAN_FFT_TWO_ROW_SUMMARY.csv"
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
        "Case",
        "Dominant f [cyc/pix]",
        "Equivalent period [pix]",
        "Center power",
        "Interpretation",
    ]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(18, 4.2), constrained_layout=True)
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
    ax.set_title("Toy Gaussian FFT two-row summary", fontsize=15, pad=16)
    png_path = PNG_DIR / f"{VERSION}_TOY_GAUSSIAN_FFT_TWO_ROW_TABLE.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return csv_path, png_path


def main() -> None:
    cases = build_cases()
    results = {
        key: {**payload, "prod": fft_products(np.asarray(payload["image"], dtype=float))}
        for key, payload in cases.items()
    }

    dashboard = make_dashboard(results)
    summary_csv, table_png = make_summary(results)

    print(f"CODE OUTPUT: {VERSION}")
    print("Purpose      Two-row toy Gaussian FFT comparison")
    print("Rows         pure Gaussian | high-pass Gaussian")
    print("Removed      hard-windowed third row")
    print("Guides       thin non-glowing center lines in Column 3")
    print(f"Plot PNG     {dashboard}")
    print(f"Table PNG    {table_png}")
    print(f"CSV          {summary_csv}")
    print(f"Timestamp    {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
