#!/usr/bin/env python3
"""
JWST_0073_FFT_16X16_EXTENDED_Y_AXIS.py

High-resolution 16 x 16 FFT comparison with blank tiles and extra vertical
headroom so the normalized peaks are not clipped at the top of the log axis.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VERSION = "JWST_0073"
N = 16
PAD = 2048
MAX_LEVEL = 64.0
ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
PNG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)


def build_binary_checkerboard() -> np.ndarray:
    y, x = np.indices((N, N))
    return ((x + y) % 2).astype(float)


def build_gray_tile() -> np.ndarray:
    # Same complementary gray progression used in the prior comparison.
    # Each adjacent pair sums to 64, with no text rendered into the cells.
    left_values = list(range(65)) + list(range(63, 0, -1))
    flat: list[float] = []
    for left in left_values:
        flat.extend((float(left), float(64 - left)))
    return np.asarray(flat, dtype=float).reshape(N, N)


def fft_products(image: np.ndarray):
    centered = image - image.mean()

    exact_fft = np.fft.fftshift(np.fft.fft2(centered))
    exact_power = np.abs(exact_fft) ** 2
    exact_power /= max(float(exact_power.max()), np.finfo(float).tiny)
    exact_freq = np.fft.fftshift(np.fft.fftfreq(N, d=1.0))
    exact_diag = np.diag(exact_power).copy()

    dense_fft = np.fft.fftshift(np.fft.fft2(centered, s=(PAD, PAD)))
    dense_power = np.abs(dense_fft) ** 2
    dense_power /= max(float(dense_power.max()), np.finfo(float).tiny)
    dense_freq = np.fft.fftshift(np.fft.fftfreq(PAD, d=1.0))
    dense_diag = np.diag(dense_power).copy()

    return exact_power, exact_freq, exact_diag, dense_freq, dense_diag


def draw_blank_tile(ax: plt.Axes, image: np.ndarray, title: str, vmax: float) -> None:
    ax.imshow(
        image,
        cmap="gray",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        origin="upper",
    )
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_fft_map(ax: plt.Axes, power: np.ndarray, title: str) -> None:
    ax.imshow(
        np.log10(power + 1.0e-12),
        cmap="magma",
        origin="lower",
        extent=[-0.5, 0.5, -0.5, 0.5],
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel("fx [cycles/pixel]")
    ax.set_ylabel("fy [cycles/pixel]")
    ax.grid(alpha=0.14)


def make_pattern_dashboard(
    checker: np.ndarray,
    gray: np.ndarray,
    checker_power: np.ndarray,
    gray_power: np.ndarray,
) -> Path:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), constrained_layout=True)

    draw_blank_tile(axes[0, 0], checker, "16 x 16 binary checkerboard", 1.0)
    draw_fft_map(axes[0, 1], checker_power, "Binary checkerboard 2-D FFT power")
    draw_blank_tile(
        axes[1, 0],
        gray,
        "16 x 16 complementary gray sequence\nblank tiles only",
        MAX_LEVEL,
    )
    draw_fft_map(axes[1, 1], gray_power, "Gray-sequence 2-D FFT power")

    fig.suptitle(
        "16 x 16 Alternating Gray Levels — FFT Comparison",
        fontsize=21,
        y=1.02,
    )

    output = PNG_DIR / f"{VERSION}_16X16_PATTERN_AND_2D_FFT.png"
    fig.savefig(output, dpi=360, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def make_curve_comparison(
    checker_exact_f: np.ndarray,
    checker_exact_p: np.ndarray,
    checker_dense_f: np.ndarray,
    checker_dense_p: np.ndarray,
    gray_exact_f: np.ndarray,
    gray_exact_p: np.ndarray,
    gray_dense_f: np.ndarray,
    gray_dense_p: np.ndarray,
) -> Path:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 1, figsize=(18, 14), constrained_layout=True)

    rows = [
        (
            axes[0],
            checker_exact_f,
            checker_exact_p,
            checker_dense_f,
            checker_dense_p,
            "Binary checkerboard",
        ),
        (
            axes[1],
            gray_exact_f,
            gray_exact_p,
            gray_dense_f,
            gray_dense_p,
            "Complementary gray sequence",
        ),
    ]

    for ax, exact_f, exact_p, dense_f, dense_p, title in rows:
        ax.plot(
            dense_f,
            np.maximum(dense_p, 1.0e-12),
            linewidth=2.5,
            label="dense diagonal FFT power curve",
        )
        ax.scatter(
            exact_f,
            np.maximum(exact_p, 1.0e-12),
            s=44,
            zorder=4,
            label="exact 16 x 16 FFT bins",
        )
        ax.set_yscale("log")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(1.0e-12, 1.0e1)
        ax.set_title(title, fontsize=16, pad=14)
        ax.set_ylabel("normalized FFT power")
        ax.grid(alpha=0.20, which="both")
        ax.legend(loc="upper center")

    axes[1].set_xlabel("diagonal spatial frequency, fx = fy [cycles/pixel]")

    fig.suptitle(
        "High-Resolution Diagonal FFT Power-Spectrum Comparison\n"
        "Extended y-axis to 10¹ so the normalized peaks remain fully visible",
        fontsize=21,
        y=1.025,
    )

    output = PNG_DIR / f"{VERSION}_16X16_EXTENDED_Y_AXIS_CURVES.png"
    fig.savefig(output, dpi=360, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def save_csv(
    frequency: np.ndarray,
    checker_curve: np.ndarray,
    gray_curve: np.ndarray,
) -> Path:
    output = CSV_DIR / f"{VERSION}_16X16_DIAGONAL_FFT_CURVES.csv"
    pd.DataFrame(
        {
            "frequency_cycles_per_pixel": frequency,
            "binary_checkerboard_normalized_power": checker_curve,
            "gray_sequence_normalized_power": gray_curve,
        }
    ).to_csv(output, index=False)
    return output


def main() -> None:
    checker = build_binary_checkerboard()
    gray = build_gray_tile()

    checker_power, checker_exact_f, checker_exact_p, checker_dense_f, checker_dense_p = fft_products(checker)
    gray_power, gray_exact_f, gray_exact_p, gray_dense_f, gray_dense_p = fft_products(gray)

    dashboard = make_pattern_dashboard(checker, gray, checker_power, gray_power)
    curves = make_curve_comparison(
        checker_exact_f,
        checker_exact_p,
        checker_dense_f,
        checker_dense_p,
        gray_exact_f,
        gray_exact_p,
        gray_dense_f,
        gray_dense_p,
    )
    csv_path = save_csv(checker_dense_f, checker_dense_p, gray_dense_p)

    print(f"CODE OUTPUT: {VERSION}")
    print("Y-AXIS RANGE     1e-12 to 1e+1 normalized FFT power")
    print("DISPLAY          blank tiles; no overlaid cell numbers")
    print(f"PATTERN PNG      {dashboard}")
    print(f"CURVE PNG        {curves}")
    print(f"CSV              {csv_path}")
    print(f"Timestamp        {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
