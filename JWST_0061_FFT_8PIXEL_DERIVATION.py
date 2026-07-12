#!/usr/bin/env python3
"""
JWST_0061 — exact mathematical derivation for two 1x8 strips.

Columns:
1) literal 1x8 strip,
2) literal intensity cross-section,
3) raw FFT coefficient magnitude |F_k| (the FFT),
4) one-sided FFT power w_k |F_k|^2 with exact percentage derivation.

No repeated rows and no 2-D FFT are used.
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
from matplotlib.patches import Rectangle

VERSION = "JWST_0061"
ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
for directory in (PNG_DIR, CSV_DIR):
    directory.mkdir(parents=True, exist_ok=True)

PATTERNS = [
    {
        "name": "BLOCK_STEP",
        "label": "00001111 — one broad step",
        "values": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float),
    },
    {
        "name": "ALTERNATING",
        "label": "01010101 — alternating every pixel",
        "values": np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float),
    },
]


def analyze(values: np.ndarray) -> dict:
    n = values.size
    k = np.arange(n // 2 + 1)
    frequency = np.fft.rfftfreq(n, d=1.0)
    coefficient = np.fft.rfft(values)
    magnitude = np.abs(coefficient)
    raw_power = magnitude**2

    weights = np.ones_like(raw_power)
    weights[1:-1] = 2.0  # positive-frequency bin plus its negative-frequency partner
    one_sided_power = weights * raw_power
    total_power = float(one_sided_power.sum())
    parseval_total = float(n * np.sum(values**2))
    share_percent = 100.0 * one_sided_power / total_power

    period = np.full_like(frequency, np.inf, dtype=float)
    period[frequency > 0] = 1.0 / frequency[frequency > 0]

    return {
        "n": n,
        "k": k,
        "frequency": frequency,
        "period": period,
        "coefficient": coefficient,
        "magnitude": magnitude,
        "raw_power": raw_power,
        "weights": weights,
        "one_sided_power": one_sided_power,
        "total_power": total_power,
        "parseval_total": parseval_total,
        "share_percent": share_percent,
    }


def draw_strip(ax: plt.Axes, values: np.ndarray, label: str) -> None:
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    for x, value in enumerate(values):
        face = "white" if value > 0.5 else "black"
        text = "black" if value > 0.5 else "white"
        ax.add_patch(Rectangle((x, 0), 1, 1, facecolor=face, edgecolor="0.55", linewidth=1.0))
        ax.text(x + 0.5, 0.5, str(int(value)), ha="center", va="center",
                fontsize=13, color=text, weight="bold")
    ax.set_xticks(np.arange(8) + 0.5, labels=[str(i) for i in range(8)])
    ax.set_yticks([])
    ax.set_xlabel("pixel index")
    ax.set_title(label, fontsize=13)


def draw_profile(ax: plt.Axes, values: np.ndarray) -> None:
    edges = np.arange(9, dtype=float)
    ax.step(edges, np.r_[values, values[-1]], where="post", linewidth=2.2)
    ax.scatter(np.arange(8) + 0.5, values, s=42, zorder=3)
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.12, 1.12)
    ax.set_xticks(np.arange(9))
    ax.set_yticks([0, 1])
    ax.set_xlabel("x position [pixels]")
    ax.set_ylabel("intensity")
    ax.set_title("Literal intensity cross-section", fontsize=11)
    ax.grid(alpha=0.22)


def draw_fft(ax: plt.Axes, result: dict) -> None:
    f = result["frequency"]
    mag = result["magnitude"]
    bars = ax.bar(f, mag, width=0.072, edgecolor="white", linewidth=0.7)
    ax.set_xlim(-0.04, 0.54)
    ax.set_ylim(0, max(4.8, float(mag.max()) * 1.28))
    ax.set_xticks(f)
    ax.set_xlabel("spatial frequency f = k/8 [cycles/pixel]")
    ax.set_ylabel("FFT magnitude |Fₖ|")
    ax.set_title("Raw 1-D FFT coefficients", fontsize=11)
    ax.grid(axis="y", alpha=0.22)

    for bar, k, coeff, value in zip(bars, result["k"], result["coefficient"], mag):
        if value < 1e-12:
            continue
        sign = "+" if coeff.imag >= 0 else "−"
        label = f"k={k}\nF={coeff.real:.3f}{sign}{abs(coeff.imag):.3f}i\n|F|={value:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.10,
                label, ha="center", va="bottom", fontsize=7.5)


def draw_power(ax: plt.Axes, result: dict) -> None:
    f = result["frequency"]
    power = result["one_sided_power"]
    share = result["share_percent"]
    bars = ax.bar(f, power, width=0.072, edgecolor="white", linewidth=0.7)
    ax.set_xlim(-0.04, 0.54)
    ax.set_ylim(0, max(19.0, float(power.max()) * 1.35))
    ax.set_xticks(f)
    ax.set_xlabel("spatial frequency [cycles/pixel]")
    ax.set_ylabel("one-sided FFT power wₖ|Fₖ|²")
    ax.set_title("Power derived from the FFT", fontsize=11)
    ax.grid(axis="y", alpha=0.22)

    total = result["total_power"]
    for bar, k, raw, weight, weighted, pct in zip(
        bars, result["k"], result["raw_power"], result["weights"], power, share
    ):
        if weighted < 1e-12:
            continue
        text = (
            f"k={k}\n"
            f"{weight:.0f}×{raw:.3f}={weighted:.3f}\n"
            f"{weighted:.3f}/{total:.3f}={pct:.1f}%"
        )
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.30,
                text, ha="center", va="bottom", fontsize=7.5)

    ax.text(
        0.02, 0.97,
        f"Total = Σ wₖ|Fₖ|² = {total:.3f}\n"
        f"Parseval check = NΣxₙ² = {result['parseval_total']:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="black", edgecolor="0.55", alpha=0.78),
    )


def make_dashboard(records: list[dict]) -> Path:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 4, figsize=(23, 9.5), constrained_layout=True)

    for row, record in enumerate(records):
        draw_strip(axes[row, 0], record["values"], record["label"])
        draw_profile(axes[row, 1], record["values"])
        draw_fft(axes[row, 2], record["fft"])
        draw_power(axes[row, 3], record["fft"])

    fig.suptitle(
        "Exact 1×8 FFT derivation\n"
        "The percentages are shares of Fourier power—not percentages of pixels",
        fontsize=18,
    )
    output = PNG_DIR / f"{VERSION}_8PIXEL_FFT_EXACT_DERIVATION.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def save_derivation(records: list[dict]) -> tuple[Path, Path]:
    rows = []
    for record in records:
        r = record["fft"]
        for k, f, period, coeff, mag, raw, weight, weighted, pct in zip(
            r["k"], r["frequency"], r["period"], r["coefficient"],
            r["magnitude"], r["raw_power"], r["weights"],
            r["one_sided_power"], r["share_percent"]
        ):
            rows.append({
                "pattern": record["name"],
                "k": int(k),
                "frequency_cycles_per_pixel": float(f),
                "period_pixels": float(period),
                "fft_real": float(coeff.real),
                "fft_imag": float(coeff.imag),
                "fft_magnitude": float(mag),
                "raw_power_absF_squared": float(raw),
                "one_sided_weight": float(weight),
                "one_sided_power": float(weighted),
                "power_share_percent": float(pct),
            })

    df = pd.DataFrame(rows)
    csv_path = CSV_DIR / f"{VERSION}_8PIXEL_FFT_EXACT_DERIVATION.csv"
    df.to_csv(csv_path, index=False)

    shown = df[df["one_sided_power"] > 1e-12].copy()
    shown["F_k"] = shown.apply(
        lambda row: f"{row['fft_real']:.3f} {row['fft_imag']:+.3f}i", axis=1
    )
    shown["|F_k|^2"] = shown["raw_power_absF_squared"].map(lambda v: f"{v:.3f}")
    shown["w_k"] = shown["one_sided_weight"].map(lambda v: f"{v:.0f}")
    shown["w_k|F_k|^2"] = shown["one_sided_power"].map(lambda v: f"{v:.3f}")
    shown["share"] = shown["power_share_percent"].map(lambda v: f"{v:.1f}%")
    table_df = shown[["pattern", "k", "frequency_cycles_per_pixel", "F_k", "|F_k|^2", "w_k", "w_k|F_k|^2", "share"]]
    table_df.columns = ["Pattern", "k", "f [cyc/pix]", "F_k", "|F_k|²", "w_k", "one-sided power", "share"]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(18, 5.3), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=table_df.values, colLabels=table_df.columns,
                     loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#536879")
        if row == 0:
            cell.set_facecolor("#18344f")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#111923" if row % 2 else "#17212c")
            cell.set_text_props(color="#edf3f8")
    ax.set_title(
        "Exact FFT derivation: share = w_k|F_k|² / Σ(w_j|F_j|²)\n"
        "w_k=2 combines ±frequency partners; DC and Nyquist use w_k=1",
        fontsize=15, pad=16,
    )
    table_path = PNG_DIR / f"{VERSION}_8PIXEL_FFT_DERIVATION_TABLE.png"
    fig.savefig(table_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return csv_path, table_path


def print_terminal_table(records: list[dict]) -> None:
    print("\nEXACT NONZERO FFT TERMS")
    print("PATTERN       k   f[cyc/pix]        F_k             |F_k|^2   w   ONE-SIDED   SHARE")
    print("-" * 92)
    for record in records:
        r = record["fft"]
        for k, f, coeff, raw, weight, weighted, pct in zip(
            r["k"], r["frequency"], r["coefficient"], r["raw_power"],
            r["weights"], r["one_sided_power"], r["share_percent"]
        ):
            if weighted <= 1e-12:
                continue
            ftxt = f"{coeff.real:+.6f}{coeff.imag:+.6f}i"
            print(f"{record['name']:<13} {k:>1d}   {f:>8.3f}   {ftxt:>22}   {raw:>9.6f}  {weight:>1.0f}   {weighted:>9.6f}  {pct:>6.2f}%")
        print(f"{'':13}     TOTAL POWER = {r['total_power']:.6f} = N*SUM(x^2) = {r['parseval_total']:.6f}\n")


def main() -> None:
    records = [{**pattern, "fft": analyze(pattern["values"])} for pattern in PATTERNS]
    dashboard = make_dashboard(records)
    csv_path, table_path = save_derivation(records)

    print(f"CODE OUTPUT: {VERSION}")
    print("Input A      0 0 0 0 1 1 1 1")
    print("Input B      0 1 0 1 0 1 0 1")
    print("Geometry     One literal 1x8 strip only")
    print("Column 3     Raw FFT coefficient magnitude |F_k|")
    print("Column 4     Derived one-sided power and exact percentage arithmetic")
    print_terminal_table(records)
    print(f"Plot PNG     {dashboard}")
    print(f"Table PNG    {table_path}")
    print(f"CSV          {csv_path}")
    print(f"Timestamp    {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
