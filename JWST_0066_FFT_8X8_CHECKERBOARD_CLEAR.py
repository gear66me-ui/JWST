#!/usr/bin/env python3
"""
JWST_0066_FFT_8X8_CHECKERBOARD_CLEAR.py

Compact educational 8x8 black/white checkerboard and its exact 2-D FFT.
The FFT is shown as percentage of total Fourier power, not raw power 1024.
No AI-generated imagery is used; NumPy and Matplotlib only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VERSION = "JWST_0066"
N = 8
ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
PNG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)


def build_checkerboard() -> np.ndarray:
    y, x = np.indices((N, N))
    return ((x + y) % 2).astype(float)


def compute_fft(checker: np.ndarray) -> dict[str, np.ndarray]:
    fft = np.fft.fftshift(np.fft.fft2(checker))
    power = np.abs(fft) ** 2
    total_power = float(power.sum())
    power_percent = 100.0 * power / total_power
    frequencies = np.fft.fftshift(np.fft.fftfreq(N, d=1.0))
    return {
        "fft": fft,
        "power": power,
        "power_percent": power_percent,
        "frequencies": frequencies,
    }


def format_frequency(value: float) -> str:
    if abs(value) < 1.0e-12:
        return "0"
    return f"{value:+.3f}".rstrip("0").rstrip(".")


def make_plot(checker: np.ndarray, result: dict[str, np.ndarray]) -> Path:
    frequencies = result["frequencies"]
    power_percent = result["power_percent"]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5), constrained_layout=True)

    # Left: literal 8x8 binary image.
    ax = axes[0]
    ax.imshow(checker, cmap="gray", vmin=0, vmax=1,
              origin="upper", interpolation="nearest")
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    ax.set_title("8 × 8 checkerboard\nblack = 0, white = 1", fontsize=12)
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which="minor", linewidth=0.7, alpha=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Right: exact shifted FFT power, normalized to percent of total power.
    ax = axes[1]
    ax.imshow(power_percent, cmap="gray", vmin=0, vmax=50,
              origin="lower", interpolation="nearest")
    labels = [format_frequency(v) for v in frequencies]
    ax.set_xticks(np.arange(N), labels=labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(N), labels=labels, fontsize=8)
    ax.set_xlabel("horizontal frequency fx [cycles/pixel]")
    ax.set_ylabel("vertical frequency fy [cycles/pixel]")
    ax.set_title("2-D FFT power\npercentage of total power", fontsize=12)
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which="minor", linewidth=0.7, alpha=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)

    nonzero = np.argwhere(power_percent > 1.0e-10)
    for iy, ix in nonzero:
        value = power_percent[iy, ix]
        text_color = "black" if value > 25 else "white"
        ax.text(ix, iy, f"{value:.0f}%", ha="center", va="center",
                fontsize=11, weight="bold", color=text_color)

    dc_index = int(np.where(np.isclose(frequencies, 0.0))[0][0])
    nyquist_index = int(np.where(np.isclose(frequencies, -0.5))[0][0])

    ax.annotate(
        "average brightness\n50% of total power",
        xy=(dc_index, dc_index),
        xytext=(4.8, 6.55),
        textcoords="data",
        ha="left",
        va="center",
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", linewidth=0.9),
    )
    ax.annotate(
        "checker alternation\n50% of total power\n2-pixel period in x and y",
        xy=(nyquist_index, nyquist_index),
        xytext=(1.25, 0.95),
        textcoords="data",
        ha="left",
        va="center",
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", linewidth=0.9),
    )

    fig.suptitle(
        "Exact FFT of an 8 × 8 checkerboard\n"
        "Only two Fourier bins are nonzero: the mean and the checker pattern",
        fontsize=14,
    )

    output = PNG_DIR / f"{VERSION}_8X8_CHECKERBOARD_CLEAR.png"
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def save_csv(result: dict[str, np.ndarray]) -> Path:
    frequencies = result["frequencies"]
    fft = result["fft"]
    power = result["power"]
    power_percent = result["power_percent"]

    rows: list[dict[str, float]] = []
    for iy, fy in enumerate(frequencies):
        for ix, fx in enumerate(frequencies):
            coefficient = fft[iy, ix]
            rows.append({
                "fx_cycles_per_pixel": float(fx),
                "fy_cycles_per_pixel": float(fy),
                "fft_real": float(coefficient.real),
                "fft_imag": float(coefficient.imag),
                "fft_magnitude": float(abs(coefficient)),
                "fft_power": float(power[iy, ix]),
                "percent_of_total_power": float(power_percent[iy, ix]),
            })

    output = CSV_DIR / f"{VERSION}_8X8_CHECKERBOARD_COEFFICIENTS.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def print_summary(checker: np.ndarray, result: dict[str, np.ndarray],
                  png_path: Path, csv_path: Path) -> None:
    frequencies = result["frequencies"]
    fft = result["fft"]
    power_percent = result["power_percent"]
    nonzero = np.argwhere(power_percent > 1.0e-10)

    print(f"CODE OUTPUT: {VERSION}")
    print("INPUT          8 x 8 binary checkerboard")
    print(f"MEAN VALUE     {checker.mean():.6f}")
    print("NONZERO FFT COMPONENTS")
    print("  fx [cyc/pix]  fy [cyc/pix]   coefficient      total power")
    for iy, ix in nonzero:
        coefficient = fft[iy, ix]
        print(
            f"  {frequencies[ix]:>+12.3f}  {frequencies[iy]:>+12.3f}"
            f"  {coefficient.real:>+9.3f}{coefficient.imag:>+9.3f}i"
            f"  {power_percent[iy, ix]:>10.3f}%"
        )
    print("INTERPRETATION (0,0) = average brightness; (-0.5,-0.5) = checker alternation")
    print("NYQUIST       For an 8-pixel even grid, -0.5 and +0.5 are the same endpoint bin")
    print(f"PLOT PNG      {png_path}")
    print(f"CSV           {csv_path}")
    print(f"Timestamp     {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


def main() -> None:
    checker = build_checkerboard()
    result = compute_fft(checker)
    png_path = make_plot(checker, result)
    csv_path = save_csv(result)
    print_summary(checker, result, png_path, csv_path)


if __name__ == "__main__":
    main()
