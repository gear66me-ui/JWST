#!/usr/bin/env python3
"""
JWST_0068_FFT_8X8_CHECKERBOARD_SINGLE_CURVE.py

Small 8 x 8 checkerboard plus one continuous-looking diagonal FFT power curve.
The curve is obtained by zero-padding the exact finite checkerboard before the
FFT, which densely samples the same discrete-time Fourier transform without
adding new information.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VERSION = "JWST_0068"
N = 8
FFT_GRID = 2048
ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
PNG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)


def build_checkerboard() -> np.ndarray:
    yy, xx = np.indices((N, N))
    return ((xx + yy) % 2).astype(float)


def compute_diagonal_curve(checker: np.ndarray):
    centered = checker - checker.mean()

    # Dense sampling of the exact 8 x 8 checkerboard transform.
    fft_dense = np.fft.fftshift(
        np.fft.fft2(centered, s=(FFT_GRID, FFT_GRID))
    )
    power_dense = np.abs(fft_dense) ** 2
    frequency = np.fft.fftshift(
        np.fft.fftfreq(FFT_GRID, d=1.0)
    )

    # The checker mode lies along fx = fy, so take that diagonal cut.
    diagonal_power = np.diag(power_dense)
    diagonal_power /= diagonal_power.max()

    # Include +0.5 explicitly so both Nyquist boundaries are visible.
    frequency_plot = np.append(frequency, 0.5)
    power_plot = np.append(diagonal_power, diagonal_power[0])

    # Exact 8-point FFT bins for reference markers.
    fft_exact = np.fft.fftshift(np.fft.fft2(centered))
    exact_power = np.diag(np.abs(fft_exact) ** 2)
    exact_power /= exact_power.max()
    exact_frequency = np.fft.fftshift(np.fft.fftfreq(N, d=1.0))

    return (
        centered,
        frequency_plot,
        power_plot,
        exact_frequency,
        exact_power,
    )


def make_figure(
    checker: np.ndarray,
    frequency: np.ndarray,
    power: np.ndarray,
    exact_frequency: np.ndarray,
    exact_power: np.ndarray,
) -> Path:
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10.2, 4.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 3.2], wspace=0.32)

    ax_image = fig.add_subplot(grid[0, 0])
    ax_curve = fig.add_subplot(grid[0, 1])

    ax_image.imshow(
        checker,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        origin="upper",
    )
    ax_image.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax_image.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax_image.grid(which="minor", linewidth=0.7, alpha=0.55)
    ax_image.tick_params(which="minor", bottom=False, left=False)
    ax_image.set_xticks([])
    ax_image.set_yticks([])
    ax_image.set_title("8 x 8 checkerboard\nblack = 0, white = 1", fontsize=11)

    ax_curve.plot(
        frequency,
        power,
        linewidth=2.0,
        label="Dense diagonal FFT power curve",
    )
    ax_curve.scatter(
        exact_frequency,
        exact_power,
        s=28,
        zorder=3,
        label="Exact 8 x 8 FFT bins",
    )
    ax_curve.set_xlim(-0.5, 0.5)
    ax_curve.set_ylim(-0.02, 1.05)
    ax_curve.set_xlabel("Spatial frequency along fx = fy [cycles/pixel]")
    ax_curve.set_ylabel("Normalized FFT power")
    ax_curve.set_title(
        "Single FFT spectrum curve\ncheckerboard diagonal frequency cut",
        fontsize=12,
    )
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(loc="upper center", fontsize=9)
    ax_curve.annotate(
        "checker mode\n2-pixel alternation",
        xy=(-0.5, 1.0),
        xytext=(-0.36, 0.73),
        arrowprops=dict(arrowstyle="->", linewidth=1.0),
        fontsize=9,
    )
    ax_curve.annotate(
        "zero-frequency power = 0\nafter mean subtraction",
        xy=(0.0, 0.0),
        xytext=(-0.18, 0.22),
        arrowprops=dict(arrowstyle="->", linewidth=1.0),
        fontsize=9,
    )

    fig.suptitle(
        "8 x 8 Checkerboard — One Continuous FFT Power Curve",
        fontsize=14,
    )
    output = PNG_DIR / f"{VERSION}_CHECKERBOARD_SINGLE_FFT_CURVE.png"
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def save_curve_csv(frequency: np.ndarray, power: np.ndarray) -> Path:
    output = CSV_DIR / f"{VERSION}_CHECKERBOARD_SINGLE_FFT_CURVE.csv"
    pd.DataFrame(
        {
            "frequency_cycles_per_pixel": frequency,
            "normalized_fft_power": power,
        }
    ).to_csv(output, index=False)
    return output


def main() -> None:
    checker = build_checkerboard()
    (
        centered,
        frequency,
        power,
        exact_frequency,
        exact_power,
    ) = compute_diagonal_curve(checker)

    png_path = make_figure(
        checker,
        frequency,
        power,
        exact_frequency,
        exact_power,
    )
    csv_path = save_curve_csv(frequency, power)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"INPUT              {N} x {N} binary checkerboard")
    print(f"MEAN REMOVED       {checker.mean():.6f}")
    print(f"DENSE FFT GRID     {FFT_GRID} x {FFT_GRID}")
    print("CURVE CUT          diagonal fx = fy")
    print("PEAK FREQUENCY     +/-0.500 cycles/pixel")
    print("NOTE               zero-padding smooths the display only; it adds no data")
    print(f"PLOT PNG           {png_path}")
    print(f"CSV                {csv_path}")
    print(f"Timestamp          {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
