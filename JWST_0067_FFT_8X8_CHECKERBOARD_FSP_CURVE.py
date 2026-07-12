#!/usr/bin/env python3
"""
JWST_0067_FFT_8X8_CHECKERBOARD_FSP_CURVE.py

Educational 8x8 checkerboard plus a clear one-dimensional FFT spatial-power
curve. The spectrum is averaged over all eight rows and all eight columns.
Because every row and column alternates 0,1,0,1..., the horizontal and vertical
spectra are identical.

No AI-generated imagery is used. NumPy + Matplotlib only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VERSION = "JWST_0067"
N = 8
ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
PNG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)


def checkerboard() -> np.ndarray:
    y, x = np.indices((N, N))
    return ((x + y) % 2).astype(float)


def one_sided_power(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coeff = np.fft.rfft(signal)
    power = np.abs(coeff) ** 2
    freq = np.fft.rfftfreq(signal.size, d=1.0)
    return freq, power


def averaged_axis_spectrum(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spectra = []
    frequency = None

    for row in image:
        frequency, power = one_sided_power(row)
        spectra.append(power)

    for column in image.T:
        _, power = one_sided_power(column)
        spectra.append(power)

    return frequency, np.mean(np.vstack(spectra), axis=0)


def percentage(power: np.ndarray) -> np.ndarray:
    total = float(np.sum(power))
    return 100.0 * power / total if total > 0 else np.zeros_like(power)


def period_labels(frequency: np.ndarray) -> list[str]:
    labels = []
    for f in frequency:
        if np.isclose(f, 0.0):
            labels.append("0\nconstant")
        else:
            labels.append(f"{f:.3f}\n{1.0/f:.2f} px/cycle")
    return labels


def make_plot(
    image: np.ndarray,
    frequency: np.ndarray,
    raw_share: np.ndarray,
    centered_share: np.ndarray,
) -> Path:
    plt.style.use("dark_background")
    fig, (ax_image, ax_curve) = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.4),
        gridspec_kw={"width_ratios": [0.78, 1.62]},
        constrained_layout=True,
    )

    ax_image.imshow(
        image,
        origin="upper",
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax_image.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax_image.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax_image.grid(which="minor", linewidth=0.7, alpha=0.45)
    ax_image.tick_params(which="minor", bottom=False, left=False)
    ax_image.set_xticks(np.arange(N))
    ax_image.set_yticks(np.arange(N))
    ax_image.set_xlabel("x pixel")
    ax_image.set_ylabel("y pixel")
    ax_image.set_title("8 × 8 checkerboard\nEvery row and column alternates 0, 1")

    ax_curve.plot(
        frequency,
        raw_share,
        marker="o",
        linewidth=2.1,
        markersize=7,
        label="Raw checkerboard",
    )
    ax_curve.plot(
        frequency,
        centered_share,
        marker="s",
        linewidth=2.1,
        markersize=7,
        label="After subtracting mean 0.5",
    )
    ax_curve.set_xlim(-0.015, 0.515)
    ax_curve.set_ylim(-3, 108)
    ax_curve.set_xticks(frequency)
    ax_curve.set_xticklabels(period_labels(frequency), fontsize=9)
    ax_curve.set_xlabel("Spatial frequency [cycles/pixel] and equivalent period")
    ax_curve.set_ylabel("Share of total one-dimensional FFT power [%]")
    ax_curve.set_title(
        "FFT spatial-power curve\n"
        "Average of all row spectra and all column spectra"
    )
    ax_curve.grid(alpha=0.22)
    ax_curve.legend(loc="center left")

    ax_curve.annotate(
        "50% constant mean brightness",
        xy=(0.0, raw_share[0]),
        xytext=(0.045, 64),
        arrowprops=dict(arrowstyle="->", linewidth=1.0),
        fontsize=9,
    )
    ax_curve.annotate(
        "Checker alternation:\n1 full cycle every 2 pixels",
        xy=(0.5, centered_share[-1]),
        xytext=(0.265, 88),
        arrowprops=dict(arrowstyle="->", linewidth=1.0),
        fontsize=9,
    )
    ax_curve.text(
        0.25,
        8,
        "No power at the intermediate frequency bins",
        ha="center",
        fontsize=9,
    )

    fig.suptitle(
        "8 × 8 checkerboard: where its FFT power actually goes",
        fontsize=16,
    )

    output = PNG_DIR / f"{VERSION}_8X8_CHECKERBOARD_FSP_CURVE.png"
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def main() -> None:
    image = checkerboard()
    frequency, raw_power = averaged_axis_spectrum(image)
    centered = image - image.mean()
    _, centered_power = averaged_axis_spectrum(centered)

    raw_share = percentage(raw_power)
    centered_share = percentage(centered_power)

    frame = pd.DataFrame({
        "frequency_cycles_per_pixel": frequency,
        "period_pixels": [np.inf if f == 0 else 1.0 / f for f in frequency],
        "raw_average_power": raw_power,
        "raw_power_share_percent": raw_share,
        "mean_subtracted_average_power": centered_power,
        "mean_subtracted_power_share_percent": centered_share,
    })

    csv_path = CSV_DIR / f"{VERSION}_8X8_CHECKERBOARD_FSP_CURVE.csv"
    frame.to_csv(csv_path, index=False)
    png_path = make_plot(image, frequency, raw_share, centered_share)

    print(f"CODE OUTPUT: {VERSION}")
    print("INPUT       8 x 8 binary checkerboard")
    print("CURVE       Average one-sided FFT power from all rows and columns")
    print("Frequency   Period [pix]   Raw share [%]   Mean-subtracted share [%]")
    for f, raw, centered_value in zip(frequency, raw_share, centered_share):
        period = "infinite" if f == 0 else f"{1.0/f:.4f}"
        print(f"{f:>9.3f}   {period:>12}   {raw:>13.3f}   {centered_value:>25.3f}")
    print("RESULT      Raw: 50% at f=0 and 50% at f=0.5 cycles/pixel")
    print("RESULT      Mean removed: 100% at f=0.5 cycles/pixel")
    print(f"PLOT PNG    {png_path}")
    print(f"CSV         {csv_path}")
    print(f"Timestamp   {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
