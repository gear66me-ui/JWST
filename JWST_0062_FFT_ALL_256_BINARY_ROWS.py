#!/usr/bin/env python3
"""
JWST_0062_FFT_ALL_256_BINARY_ROWS.py

Enumerate every possible 8-pixel binary row (2^8 = 256), compute the full
complex 1-D FFT for each row, and demonstrate which information is preserved
by the full FFT versus magnitude/power alone.

Outputs
-------
1. 256x8 black/white pattern atlas.
2. Stacked x-direction cross-sections for all 256 rows.
3. Full FFT atlas: magnitude, power, and phase for every row and every k-bin.
4. Styled summary table and full 2,048-row CSV of complex coefficients.

No AI-generated imagery is used. All figures are computed with NumPy and
rendered with Matplotlib.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def ensure_packages() -> None:
    required = {"numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib"}
    missing = [pip for module, pip in required.items() if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_packages()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.collections import LineCollection

VERSION = "JWST_0062"
ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
for directory in (PNG_DIR, CSV_DIR):
    directory.mkdir(parents=True, exist_ok=True)

N = 8
N_PATTERNS = 2**N
HIGHLIGHT_ROWS = [15, 85]


def build_patterns() -> tuple[np.ndarray, list[str]]:
    labels = [f"{value:08b}" for value in range(N_PATTERNS)]
    matrix = np.array([[int(bit) for bit in label] for label in labels], dtype=float)
    return matrix, labels


def compute_fft(patterns: np.ndarray) -> dict:
    coefficients = np.fft.fft(patterns, axis=1)
    magnitude = np.abs(coefficients)
    power = magnitude**2
    phase = np.angle(coefficients)
    frequencies = np.fft.fftfreq(N, d=1.0)
    reconstruction = np.fft.ifft(coefficients, axis=1).real
    max_error = float(np.max(np.abs(reconstruction - patterns)))

    complex_signature = np.round(
        np.concatenate([coefficients.real, coefficients.imag], axis=1), 12
    )
    magnitude_signature = np.round(magnitude, 12)
    power_signature = np.round(power, 12)

    return {
        "coefficients": coefficients,
        "magnitude": magnitude,
        "power": power,
        "phase": phase,
        "frequencies": frequencies,
        "reconstruction": reconstruction,
        "max_error": max_error,
        "unique_complex": int(np.unique(complex_signature, axis=0).shape[0]),
        "unique_magnitude": int(np.unique(magnitude_signature, axis=0).shape[0]),
        "unique_power": int(np.unique(power_signature, axis=0).shape[0]),
    }


def pattern_atlas(patterns: np.ndarray, labels: list[str]) -> Path:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 25), constrained_layout=True)
    ax.imshow(patterns, cmap="gray", vmin=0, vmax=1, origin="upper",
              aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(N), labels=[f"x{idx}" for idx in range(N)])
    major_rows = np.arange(0, N_PATTERNS, 8)
    ax.set_yticks(major_rows)
    ax.set_yticklabels([f"{row:3d}  {labels[row]}" for row in major_rows], fontsize=8)
    ax.set_xlabel("Pixel position")
    ax.set_ylabel("Decimal row index and 8-bit pattern")
    ax.set_title(
        "All 256 possible binary rows of eight pixels\n"
        "Black = 0, white = 1",
        fontsize=17,
    )

    for row in HIGHLIGHT_ROWS:
        ax.axhline(row - 0.5, linewidth=1.2, alpha=0.9)
        ax.axhline(row + 0.5, linewidth=1.2, alpha=0.9)
        ax.text(7.65, row, f"  {row}: {labels[row]}", va="center", ha="left", fontsize=9)

    ax.set_xlim(-0.5, 8.9)
    output = PNG_DIR / f"{VERSION}_01_ALL_256_BINARY_PATTERN_ATLAS.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def cross_section_catalog(patterns: np.ndarray, labels: list[str]) -> Path:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(13, 25), constrained_layout=True)

    x = np.arange(N, dtype=float)
    spacing = 1.35
    segments = []
    for row_index, values in enumerate(patterns):
        y = row_index * spacing + values
        segments.append(np.column_stack([x, y]))

    collection = LineCollection(segments, linewidths=0.38, alpha=0.58)
    ax.add_collection(collection)

    for row in HIGHLIGHT_ROWS:
        y = row * spacing + patterns[row]
        ax.plot(x, y, linewidth=2.0, marker="o", markersize=3.8,
                label=f"row {row}: {labels[row]}")

    ax.set_xlim(-0.25, 7.25)
    ax.set_ylim(-1.0, (N_PATTERNS - 1) * spacing + 2.0)
    ax.invert_yaxis()
    tick_rows = np.arange(0, N_PATTERNS, 8)
    ax.set_yticks(tick_rows * spacing + 0.5)
    ax.set_yticklabels([f"{row:3d}  {labels[row]}" for row in tick_rows], fontsize=8)
    ax.set_xticks(np.arange(N))
    ax.set_xlabel("Pixel position x")
    ax.set_ylabel("Each row offset vertically; step height is intensity 0 or 1")
    ax.set_title(
        "X-direction cross-section of every one of the 256 rows\n"
        "Each thin trace is one literal eight-pixel row",
        fontsize=17,
    )
    ax.grid(axis="x", alpha=0.16)
    ax.legend(loc="upper right")

    output = PNG_DIR / f"{VERSION}_02_ALL_256_CROSS_SECTIONS.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def fft_atlas(fft: dict, labels: list[str]) -> Path:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(22, 16), constrained_layout=True)

    datasets = [
        (np.log10(1.0 + fft["magnitude"]), "log10(1 + |Fₖ|)", "FFT magnitude"),
        (np.log10(1.0 + fft["power"]), "log10(1 + |Fₖ|²)", "FFT power"),
        (fft["phase"], "phase [radians]", "FFT phase"),
    ]

    frequencies = fft["frequencies"]
    xlabels = [f"k={k}\n{frequencies[k]:+.3f}" for k in range(N)]
    row_ticks = np.arange(0, N_PATTERNS, 16)

    for ax, (data, cbar_label, title) in zip(axes, datasets):
        if title == "FFT phase":
            image = ax.imshow(
                data, origin="upper", aspect="auto", interpolation="nearest",
                cmap="twilight", norm=TwoSlopeNorm(vmin=-np.pi, vcenter=0.0, vmax=np.pi),
            )
        else:
            image = ax.imshow(
                data, origin="upper", aspect="auto", interpolation="nearest",
                cmap="magma",
            )
        ax.set_xticks(np.arange(N), labels=xlabels, fontsize=8)
        ax.set_yticks(row_ticks)
        ax.set_yticklabels([f"{row:3d} {labels[row]}" for row in row_ticks], fontsize=8)
        ax.set_xlabel("Fourier bin k and frequency [cycles/pixel]")
        ax.set_ylabel("Input row")
        ax.set_title(title, fontsize=14)
        for row in HIGHLIGHT_ROWS:
            ax.axhline(row - 0.5, linewidth=1.0, alpha=0.85)
            ax.axhline(row + 0.5, linewidth=1.0, alpha=0.85)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
        colorbar.set_label(cbar_label)

    fig.suptitle(
        "Full FFT of all 256 binary eight-pixel rows\n"
        "Magnitude/power alone repeat; complex magnitude + phase uniquely identify all rows",
        fontsize=18,
    )

    output = PNG_DIR / f"{VERSION}_03_ALL_256_FULL_FFT_ATLAS.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def build_detailed_table(patterns: np.ndarray, labels: list[str], fft: dict) -> pd.DataFrame:
    rows = []
    for row_index in range(N_PATTERNS):
        for k in range(N):
            coefficient = fft["coefficients"][row_index, k]
            frequency = fft["frequencies"][k]
            rows.append({
                "row_index": row_index,
                "binary_pattern": labels[row_index],
                "ones_count": int(patterns[row_index].sum()),
                "k": k,
                "frequency_cycles_per_pixel": float(frequency),
                "period_pixels": np.inf if frequency == 0 else float(1.0 / abs(frequency)),
                "F_real": float(coefficient.real),
                "F_imag": float(coefficient.imag),
                "F_magnitude": float(fft["magnitude"][row_index, k]),
                "F_power": float(fft["power"][row_index, k]),
                "F_phase_radians": float(fft["phase"][row_index, k]),
            })
    return pd.DataFrame(rows)


def summary_table(fft: dict) -> tuple[Path, Path, Path]:
    summary = pd.DataFrame([
        ["Possible 8-bit rows", N_PATTERNS, "2^8"],
        ["Unique full complex FFT signatures", fft["unique_complex"], "Magnitude + phase retained"],
        ["Unique magnitude signatures", fft["unique_magnitude"], "Phase discarded"],
        ["Unique power signatures", fft["unique_power"], "Phase discarded"],
        ["Maximum inverse-FFT reconstruction error", fft["max_error"], "Numerical roundoff"],
    ], columns=["Metric", "Value", "Interpretation"])

    summary_csv = CSV_DIR / f"{VERSION}_SUMMARY.csv"
    summary.to_csv(summary_csv, index=False)

    shown = summary.copy()
    shown["Value"] = shown["Value"].map(
        lambda value: f"{value:.3e}" if isinstance(value, float) and value < 1e-6 else str(value)
    )

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 5.0), constrained_layout=True)
    ax.axis("off")
    table = ax.table(
        cellText=shown.values,
        colLabels=shown.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.42, 0.18, 0.40],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#536879")
        cell.set_linewidth(0.75)
        if row == 0:
            cell.set_facecolor("#18344f")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#111923" if row % 2 else "#17212c")
            cell.set_text_props(color="#edf3f8")

    ax.set_title(
        "FFT uniqueness audit for all 256 binary rows\n"
        "The full complex FFT is one-to-one; magnitude or power alone is not",
        fontsize=16,
        pad=16,
    )
    table_png = PNG_DIR / f"{VERSION}_04_FFT_UNIQUENESS_TABLE.png"
    fig.savefig(table_png, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return summary_csv, table_png, summary


def selected_derivation_table(detailed: pd.DataFrame) -> Path:
    selected = detailed[detailed["row_index"].isin(HIGHLIGHT_ROWS)].copy()
    selected["F_k"] = selected.apply(
        lambda row: f"{row['F_real']:+.3f}{row['F_imag']:+.3f}i", axis=1
    )
    selected["|F_k|"] = selected["F_magnitude"].map(lambda value: f"{value:.3f}")
    selected["|F_k|²"] = selected["F_power"].map(lambda value: f"{value:.3f}")
    selected["phase"] = selected["F_phase_radians"].map(lambda value: f"{value:+.3f}")
    shown = selected[[
        "row_index", "binary_pattern", "k", "frequency_cycles_per_pixel",
        "F_k", "|F_k|", "|F_k|²", "phase",
    ]]
    shown.columns = ["Row", "Pattern", "k", "f [cyc/pix]", "F_k", "|F_k|", "|F_k|²", "phase [rad]"]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(17, 9), constrained_layout=True)
    ax.axis("off")
    table = ax.table(
        cellText=shown.values,
        colLabels=shown.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.45)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#536879")
        cell.set_linewidth(0.65)
        if row == 0:
            cell.set_facecolor("#18344f")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#111923" if row % 2 else "#17212c")
            cell.set_text_props(color="#edf3f8")

    ax.set_title(
        "Exact complex FFT coefficients for the two highlighted rows\n"
        "The full 2,048-bin table is saved as CSV",
        fontsize=15,
        pad=16,
    )
    output = PNG_DIR / f"{VERSION}_05_SELECTED_COMPLEX_FFT_TABLE.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def main() -> None:
    patterns, labels = build_patterns()
    fft = compute_fft(patterns)

    atlas_png = pattern_atlas(patterns, labels)
    cross_png = cross_section_catalog(patterns, labels)
    fft_png = fft_atlas(fft, labels)

    detailed = build_detailed_table(patterns, labels, fft)
    detailed_csv = CSV_DIR / f"{VERSION}_ALL_256_COMPLEX_FFT_BINS.csv"
    detailed.to_csv(detailed_csv, index=False)

    summary_csv, summary_png, summary = summary_table(fft)
    selected_png = selected_derivation_table(detailed)

    print(f"CODE OUTPUT: {VERSION}")
    print("Input rows        All 256 possible 8-bit binary patterns")
    print(f"Complex FFT       {fft['unique_complex']} unique signatures")
    print(f"Magnitude only    {fft['unique_magnitude']} unique signatures")
    print(f"Power only        {fft['unique_power']} unique signatures")
    print(f"IFFT max error    {fft['max_error']:.3e}")
    print(f"Pattern PNG       {atlas_png}")
    print(f"Cross-section PNG {cross_png}")
    print(f"FFT atlas PNG     {fft_png}")
    print(f"Summary PNG       {summary_png}")
    print(f"Selected table    {selected_png}")
    print(f"Full FFT CSV      {detailed_csv}")
    print(f"Summary CSV       {summary_csv}")
    print(f"Timestamp         {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
