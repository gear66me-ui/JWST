#!/usr/bin/env python3
"""
JWST_0065_FFT_8X8_CHECKERBOARD.py

Educational 8 x 8 black/white checkerboard and its exact 2-D FFT.
All graphics are generated numerically with NumPy and Matplotlib.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

VERSION = "JWST_0065"
N = 8
ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
PNG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)


def build_checkerboard() -> np.ndarray:
    yy, xx = np.indices((N, N))
    return ((xx + yy) % 2).astype(float)


def fft_products(image: np.ndarray) -> dict[str, np.ndarray]:
    fft_raw = np.fft.fftshift(np.fft.fft2(image))
    centered = image - image.mean()
    fft_centered = np.fft.fftshift(np.fft.fft2(centered))
    frequencies = np.fft.fftshift(np.fft.fftfreq(N, d=1.0))
    return {
        "centered_image": centered,
        "fft_raw": fft_raw,
        "magnitude_raw": np.abs(fft_raw),
        "power_raw": np.abs(fft_raw) ** 2,
        "fft_centered": fft_centered,
        "magnitude_centered": np.abs(fft_centered),
        "power_centered": np.abs(fft_centered) ** 2,
        "frequencies": frequencies,
    }


def annotate_matrix(ax: plt.Axes, values: np.ndarray, threshold: float = 1.0e-9) -> None:
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = float(values[row, col])
            if abs(value) > threshold:
                ax.text(
                    col,
                    row,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    weight="bold",
                )


def draw_checker_grid(ax: plt.Axes, checker: np.ndarray) -> None:
    ax.imshow(checker, origin="upper", cmap="gray", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which="minor", linewidth=0.8, alpha=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    ax.set_title("8 × 8 black/white checkerboard\n0 and 1 alternate in both directions", fontsize=13)


def draw_fft_map(
    ax: plt.Axes,
    values: np.ndarray,
    frequencies: np.ndarray,
    title: str,
    log_scale: bool = False,
) -> None:
    extent = [frequencies[0] - 1/(2*N), frequencies[-1] + 1/(2*N),
              frequencies[0] - 1/(2*N), frequencies[-1] + 1/(2*N)]
    if log_scale:
        positive = values[values > 0]
        vmin = max(float(np.min(positive)), 1.0e-12) if positive.size else 1.0e-12
        vmax = float(np.max(positive)) if positive.size else 1.0
        image = ax.imshow(
            np.maximum(values, vmin),
            origin="lower",
            cmap="magma",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            extent=extent,
            interpolation="nearest",
            aspect="equal",
        )
    else:
        image = ax.imshow(
            values,
            origin="lower",
            cmap="magma",
            extent=extent,
            interpolation="nearest",
            aspect="equal",
        )
    ax.set_xticks(frequencies)
    ax.set_yticks(frequencies)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("fx [cycles/pixel]")
    ax.set_ylabel("fy [cycles/pixel]")
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.22)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def make_dashboard(checker: np.ndarray, products: dict[str, np.ndarray]) -> Path:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(15, 13), constrained_layout=True)

    draw_checker_grid(axes[0, 0], checker)

    draw_fft_map(
        axes[0, 1],
        products["magnitude_raw"],
        products["frequencies"],
        "Raw checkerboard FFT magnitude |F|\nDC center = 32; checker mode = 32",
    )
    annotate_matrix(axes[0, 1], products["magnitude_raw"])

    draw_fft_map(
        axes[1, 0],
        products["power_raw"],
        products["frequencies"],
        "Raw checkerboard FFT power |F|²\nTwo exact nonzero bins, each with power 1024",
        log_scale=True,
    )
    annotate_matrix(axes[1, 0], products["power_raw"])

    draw_fft_map(
        axes[1, 1],
        products["power_centered"],
        products["frequencies"],
        "Mean-subtracted FFT power\nDC removed; only the checkerboard Nyquist mode remains",
        log_scale=True,
    )
    annotate_matrix(axes[1, 1], products["power_centered"])

    axes[0, 1].annotate(
        "average brightness\n(fx, fy) = (0, 0)",
        xy=(0.0, 0.0),
        xytext=(0.12, 0.20),
        arrowprops=dict(arrowstyle="->", linewidth=1.0),
        fontsize=9,
    )
    axes[0, 1].annotate(
        "checker alternation\n(-0.5, -0.5) cyc/pix\n2-pixel period in x and y",
        xy=(-0.5, -0.5),
        xytext=(-0.16, -0.30),
        arrowprops=dict(arrowstyle="->", linewidth=1.0),
        fontsize=9,
        ha="left",
    )

    fig.suptitle(
        "Exact 2-D FFT of an 8 × 8 checkerboard\n"
        "The checker contains a constant mean plus one diagonal Nyquist-frequency mode",
        fontsize=18,
    )

    output = PNG_DIR / f"{VERSION}_8X8_CHECKERBOARD_FFT.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def save_coefficients(products: dict[str, np.ndarray]) -> Path:
    frequencies = products["frequencies"]
    fft_raw = products["fft_raw"]
    fft_centered = products["fft_centered"]
    rows: list[dict[str, float]] = []
    for iy, fy in enumerate(frequencies):
        for ix, fx in enumerate(frequencies):
            raw = fft_raw[iy, ix]
            centered = fft_centered[iy, ix]
            rows.append({
                "ix_shifted": ix,
                "iy_shifted": iy,
                "fx_cycles_per_pixel": float(fx),
                "fy_cycles_per_pixel": float(fy),
                "raw_real": float(raw.real),
                "raw_imag": float(raw.imag),
                "raw_magnitude": float(abs(raw)),
                "raw_power": float(abs(raw) ** 2),
                "mean_subtracted_real": float(centered.real),
                "mean_subtracted_imag": float(centered.imag),
                "mean_subtracted_magnitude": float(abs(centered)),
                "mean_subtracted_power": float(abs(centered) ** 2),
            })
    frame = pd.DataFrame(rows)
    output = CSV_DIR / f"{VERSION}_8X8_CHECKERBOARD_FFT_COEFFICIENTS.csv"
    frame.to_csv(output, index=False)
    return output


def print_summary(
    checker: np.ndarray,
    products: dict[str, np.ndarray],
    png_path: Path,
    csv_path: Path,
) -> None:
    frequency = products["frequencies"]
    raw_fft = products["fft_raw"]
    nonzero = np.argwhere(np.abs(raw_fft) > 1.0e-9)

    print(f"CODE OUTPUT: {VERSION}")
    print("INPUT          8 x 8 binary checkerboard")
    print(f"MEAN           {checker.mean():.6f}")
    print("NONZERO RAW FFT COEFFICIENTS")
    print("   fx [cyc/pix]   fy [cyc/pix]       Real       Imag      |F|      Power")
    for iy, ix in nonzero:
        value = raw_fft[iy, ix]
        print(
            f"   {frequency[ix]:>+12.3f}   {frequency[iy]:>+12.3f}"
            f"   {value.real:>9.3f}   {value.imag:>9.3f}"
            f"   {abs(value):>7.3f}   {abs(value)**2:>9.3f}"
        )
    print("INTERPRETATION  (0,0) is average brightness; (-0.5,-0.5) is the checker mode")
    print("NOTE            For an even grid, -0.5 and +0.5 are the same Nyquist bin")
    print(f"PLOT PNG        {png_path}")
    print(f"CSV             {csv_path}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


def main() -> None:
    checker = build_checkerboard()
    products = fft_products(checker)
    png_path = make_dashboard(checker, products)
    csv_path = save_coefficients(products)
    print_summary(checker, products, png_path, csv_path)


if __name__ == "__main__":
    main()
