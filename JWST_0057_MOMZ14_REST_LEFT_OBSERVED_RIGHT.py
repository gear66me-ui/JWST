# JWST_0057
# MoM-z14 side-by-side frequency comparison.
# LEFT  = rest-frame reconstruction from the real observed JWST samples.
# RIGHT = real observed JWST samples.
# Both panels use the SAME frequency-offset x-scale so the observed feature
# visibly appears compressed by the factor (1 + z).
# No AI images. Matplotlib only. No new MAST query.

from pathlib import Path
from datetime import datetime, timezone
import importlib
import subprocess
import sys

VERSION = "JWST_0057"
GALAXY = "MoM-z14"
C_UM_THz = 299.792458

OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"

BG = "#050712"
AX_BG = "#07101f"
GRID = "#1f6f8b"
TEXT = "#e6f4ff"
MUTED = "#8fb3c7"
REST_COLOR = "#18c7d8"
OBS_COLOR = "#ff9d2e"
CURVE = "#d9edf7"
POINT = "#58bfff"


def need(package):
    try:
        importlib.import_module(package)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


def newest(pattern):
    matches = sorted(CSV.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def locate_inputs():
    line_csv = CSV / "JWST_0052_MoM-z14_LINE_COMPLEX_REST_OBSERVED_FREQUENCY.csv"
    if not line_csv.exists():
        line_csv = newest("JWST_0052_*LINE_COMPLEX_REST_OBSERVED_FREQUENCY.csv")

    raw_csv = None
    for filename in [
        "JWST_0048_REAL_RAW_SPECTRUM.csv",
        "JWST_0047_REAL_RAW_SPECTRUM.csv",
        "JWST_0046_REAL_RAW_SPECTRUM.csv",
    ]:
        candidate = CSV / filename
        if candidate.exists() and candidate.stat().st_size > 100:
            raw_csv = candidate
            break
    if raw_csv is None:
        raw_csv = newest("JWST_*_REAL_RAW_SPECTRUM.csv")

    if line_csv is None or not line_csv.exists():
        raise FileNotFoundError("Run JWST_0052 first; its line-complex CSV is missing.")
    if raw_csv is None or not raw_csv.exists():
        raise FileNotFoundError("No cached real JWST spectrum CSV was found.")
    return line_csv, raw_csv


def find_column(frame, prefix):
    for column in frame.columns:
        if str(column).lower().startswith(prefix.lower()):
            return column
    return None


def load_data(line_csv, raw_csv):
    import numpy as np
    import pandas as pd

    lines = pd.read_csv(line_csv)
    required = [
        "n",
        "line_complex",
        "centroid_rest_um",
        "raw_local_peak_um_exploratory",
        "raw_local_peak_flux",
    ]
    missing = [column for column in required if column not in lines.columns]
    if missing:
        raise RuntimeError(f"Line CSV missing columns: {missing}")

    keep = (
        np.isfinite(lines["centroid_rest_um"])
        & np.isfinite(lines["raw_local_peak_um_exploratory"])
    )
    lines = lines[keep].copy().sort_values("n").reset_index(drop=True)
    lines["rest_frequency_THz"] = C_UM_THz / lines["centroid_rest_um"]
    lines["observed_peak_frequency_THz"] = C_UM_THz / lines["raw_local_peak_um_exploratory"]
    lines["stretch_factor"] = lines["rest_frequency_THz"] / lines["observed_peak_frequency_THz"]
    lines["z_pair"] = lines["stretch_factor"] - 1.0

    raw = pd.read_csv(raw_csv)
    wave_col = find_column(raw, "wavelength")
    flux_col = find_column(raw, "flux_raw_")
    if wave_col is None or flux_col is None:
        raise RuntimeError(f"Could not find wavelength/flux columns in {raw_csv.name}")

    wave = raw[wave_col].to_numpy(float)
    flux = raw[flux_col].to_numpy(float)
    finite = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    wave = wave[finite]
    flux = flux[finite]
    order = np.argsort(wave)
    return lines, wave[order], flux[order], str(flux_col)


def narrow_samples(wave, flux, center_um, minimum_points=10):
    import numpy as np

    half_widths = [0.003, 0.0045, 0.0065, 0.009, 0.013, 0.018, 0.026, 0.038]
    mask = None
    used = half_widths[-1]
    for half_width in half_widths:
        trial = (
            np.isfinite(wave)
            & np.isfinite(flux)
            & (wave >= center_um - half_width)
            & (wave <= center_um + half_width)
        )
        mask = trial
        used = half_width
        if int(trial.sum()) >= minimum_points:
            break
    return wave[mask], flux[mask], used


def style_axis(ax):
    ax.set_facecolor(AX_BG)
    ax.grid(True, color=GRID, linewidth=0.45, alpha=0.48)
    ax.tick_params(colors=TEXT, labelsize=7.2)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color("#4fa8c8")
        spine.set_linewidth(0.82)


def plot_profile(ax, offset_frequency, flux, center_frequency, frame_name, center_color, x_limit, line_name, sample_count):
    import numpy as np

    style_axis(ax)
    order = np.argsort(offset_frequency)
    x = np.asarray(offset_frequency)[order]
    y = np.asarray(flux)[order]

    ax.plot(x, y, color=CURVE, linewidth=0.85, alpha=0.92)
    ax.scatter(
        x,
        y,
        s=20,
        color=POINT,
        edgecolor=BG,
        linewidth=0.40,
        zorder=5,
    )
    ax.axvline(0.0, color=center_color, linewidth=1.75)
    ax.set_xlim(-x_limit, x_limit)

    if len(y) >= 2:
        ymin = float(np.nanmin(y))
        ymax = float(np.nanmax(y))
        pad = 0.10 * (ymax - ymin) if ymax > ymin else max(abs(ymax) * 0.10, 1.0e-8)
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_title(
        f"{frame_name} — center ν = {center_frequency:.6f} THz",
        fontsize=9.2,
        pad=6,
    )
    ax.set_xlabel("Frequency offset from line center, THz", fontsize=7.8)
    ax.set_ylabel("Raw flux", fontsize=7.8)
    ax.text(
        0.018,
        0.955,
        f"{line_name}\nreal JWST samples = {sample_count}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=TEXT,
        fontsize=6.9,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="#07111f",
            edgecolor=center_color,
            alpha=0.95,
        ),
    )


def build_dashboard(lines, wave, flux):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 2, figsize=(18.5, 20.5), facecolor=BG)
    sample_frames = []

    for row_index, row in enumerate(lines.itertuples()):
        local_wave, local_flux, half_width = narrow_samples(
            wave,
            flux,
            row.raw_local_peak_um_exploratory,
            minimum_points=10,
        )
        if len(local_wave) < 2:
            raise RuntimeError(f"Not enough samples for line {int(row.n)} {row.line_complex}")

        observed_frequency = C_UM_THz / local_wave
        observed_offset = observed_frequency - row.observed_peak_frequency_THz

        rest_frequency = observed_frequency * row.stretch_factor
        rest_offset = rest_frequency - row.rest_frequency_THz

        # One common offset scale for both panels. This is the critical fix:
        # the observed profile stays compressed by (1 + z) instead of auto-filling its panel.
        x_limit = float(np.nanmax(np.abs(rest_offset)))
        if not np.isfinite(x_limit) or x_limit <= 0:
            x_limit = 1.0
        x_limit *= 1.08

        plot_profile(
            axes[row_index, 0],
            rest_offset,
            local_flux,
            row.rest_frequency_THz,
            "REST FRAME",
            REST_COLOR,
            x_limit,
            f"{int(row.n)} {row.line_complex}",
            len(local_wave),
        )
        plot_profile(
            axes[row_index, 1],
            observed_offset,
            local_flux,
            row.observed_peak_frequency_THz,
            "OBSERVED FRAME",
            OBS_COLOR,
            x_limit,
            f"{int(row.n)} {row.line_complex}",
            len(local_wave),
        )

        axes[row_index, 0].text(
            0.985,
            0.955,
            (
                f"νrest / νobserved = {row.stretch_factor:.6f}×\n"
                f"z = {row.z_pair:.6f}"
            ),
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="top",
            color=TEXT,
            fontsize=7.0,
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor="#07111f",
                edgecolor=REST_COLOR,
                alpha=0.95,
            ),
        )
        axes[row_index, 1].text(
            0.985,
            0.955,
            f"same x-scale as left\nzoom = ±{half_width:.4f} µm",
            transform=axes[row_index, 1].transAxes,
            ha="right",
            va="top",
            color=TEXT,
            fontsize=7.0,
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor="#07111f",
                edgecolor=OBS_COLOR,
                alpha=0.95,
            ),
        )

        sample_frames.append(
            pd.DataFrame(
                {
                    "line_number": int(row.n),
                    "line_complex": row.line_complex,
                    "observed_wavelength_um": local_wave,
                    "raw_flux": local_flux,
                    "observed_frequency_THz": observed_frequency,
                    "observed_offset_THz": observed_offset,
                    "rest_reconstructed_frequency_THz": rest_frequency,
                    "rest_offset_THz": rest_offset,
                    "rest_center_frequency_THz": row.rest_frequency_THz,
                    "observed_center_frequency_THz": row.observed_peak_frequency_THz,
                    "stretch_factor": row.stretch_factor,
                    "z_pair": row.z_pair,
                    "zoom_half_width_um": half_width,
                }
            )
        )

    fig.suptitle(
        f"{VERSION} — {GALAXY}: REST LEFT, OBSERVED RIGHT",
        color=TEXT,
        fontsize=16.2,
        fontweight="bold",
        y=0.992,
    )
    fig.text(
        0.5,
        0.974,
        (
            "Both columns use the same frequency-offset scale. "
            "The observed profile is therefore visibly compressed by the measured (1+z) factor."
        ),
        color=MUTED,
        fontsize=9.5,
        ha="center",
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.955,
        bottom=0.035,
        hspace=0.34,
        wspace=0.16,
    )

    dashboard_path = PNG / f"{VERSION}_{GALAXY}_REST_LEFT_OBSERVED_RIGHT_COMMON_SCALE.png"
    fig.savefig(dashboard_path, dpi=235, facecolor=BG, edgecolor=BG)
    plt.show()
    plt.close(fig)

    samples = pd.concat(sample_frames, ignore_index=True, sort=False)
    samples_csv = CSV / f"{VERSION}_{GALAXY}_REST_LEFT_OBSERVED_RIGHT_SAMPLES.csv"
    samples.to_csv(samples_csv, index=False)
    return dashboard_path, samples_csv, samples


def print_table(rows, headers):
    widths = [len(str(header)) for header in headers]
    for row in rows:
        widths = [max(widths[index], len(str(row[index]))) for index in range(len(headers))]
    print(" | ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(str(row[index]).ljust(widths[index]) for index in range(len(headers))))


def main():
    for package in ["numpy", "pandas", "matplotlib"]:
        need(package)

    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)

    line_csv, raw_csv = locate_inputs()
    lines, wave, flux, flux_column = load_data(line_csv, raw_csv)
    dashboard_path, samples_csv, samples = build_dashboard(lines, wave, flux)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table(
        [
            (
                int(row.n),
                row.line_complex,
                f"{row.rest_frequency_THz:.6f}",
                f"{row.observed_peak_frequency_THz:.6f}",
                f"{row.stretch_factor:.6f}",
                f"{row.z_pair:.6f}",
                int((samples["line_number"] == int(row.n)).sum()),
            )
            for row in lines.itertuples()
        ],
        ["#", "Line complex", "Rest THz", "Observed THz", "Ratio x", "z", "Samples"],
    )
    print()
    print_table(
        [
            ("Galaxy", GALAXY),
            ("Layout", "REST LEFT | OBSERVED RIGHT"),
            ("Scale", "same frequency-offset scale in both columns"),
            ("Dashboard PNG", str(dashboard_path)),
            ("Samples CSV", str(samples_csv)),
            ("Line source CSV", str(line_csv)),
            ("Real spectrum CSV", str(raw_csv)),
            ("Flux column", flux_column),
        ],
        ["Field", "Value"],
    )
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
