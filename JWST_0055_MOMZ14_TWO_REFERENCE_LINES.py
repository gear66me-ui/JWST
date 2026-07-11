# JWST_0055
# MoM-z14 simplified spectrum plots in the fixed JWST_0050 presentation style.
# Exactly two reference lines per spectrum:
#   cyan  = rest wavelength carried to the observed frame at published z=14.44
#   orange = observed local peak used in the exploratory wavelength-pair calculation
# No AI images. Matplotlib only. No new MAST query.

from pathlib import Path
from datetime import datetime, timezone
import importlib
import subprocess
import sys

VERSION = "JWST_0055"
GALAXY = "MoM-z14"
PUBLISHED_Z = 14.44
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"

BG = "#050712"
AX_BG = "#07101f"
GRID = "#1f6f8b"
TEXT = "#e6f4ff"
MUTED = "#8fb3c7"
CYAN = "#18c7d8"
ORANGE = "#ff9d2e"
POINT_COLORS = ["#43b9ff", "#ff6b76", "#a78bfa", "#34d399", "#f6c453"]


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
        raise FileNotFoundError("No cached real spectrum CSV was found.")
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
        "centroid_expected_observed_um",
        "raw_local_peak_um_exploratory",
        "raw_local_peak_flux",
        "window_low_um",
        "window_high_um",
        "sample_count",
    ]
    missing = [column for column in required if column not in lines.columns]
    if missing:
        raise RuntimeError(f"Line CSV missing columns: {missing}")

    lines = lines[
        np.isfinite(lines["centroid_rest_um"])
        & np.isfinite(lines["centroid_expected_observed_um"])
        & np.isfinite(lines["raw_local_peak_um_exploratory"])
    ].copy()
    lines = lines.sort_values("n").reset_index(drop=True)
    lines["slope_m"] = lines["raw_local_peak_um_exploratory"] / lines["centroid_rest_um"]
    lines["z_pair"] = lines["slope_m"] - 1.0
    lines["delta_z_from_published"] = lines["z_pair"] - PUBLISHED_Z

    raw = pd.read_csv(raw_csv)
    wave_col = find_column(raw, "wavelength")
    flux_col = find_column(raw, "flux_raw_")
    if wave_col is None or flux_col is None:
        raise RuntimeError(f"Could not find spectrum columns in {raw_csv.name}")

    wave = raw[wave_col].to_numpy(float)
    flux = raw[flux_col].to_numpy(float)
    finite = np.isfinite(wave) & np.isfinite(flux)
    wave = wave[finite]
    flux = flux[finite]
    order = np.argsort(wave)
    return lines, wave[order], flux[order], str(flux_col)


def style_axis(ax, small=False):
    ax.set_facecolor(AX_BG)
    ax.grid(True, color=GRID, linewidth=0.45 if small else 0.58, alpha=0.48)
    ax.tick_params(colors=TEXT, labelsize=7.2 if small else 9.0)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color("#4fa8c8")
        spine.set_linewidth(0.82)


def dark_legend(ax, small=False):
    legend = ax.legend(
        loc="lower right",
        fontsize=6.2 if small else 7.6,
        facecolor="#07111f",
        edgecolor=GRID,
        framealpha=0.96,
    )
    for text in legend.get_texts():
        text.set_color(TEXT)


def draw_spectrum(ax, row, wave, flux, color, small=False):
    import numpy as np

    style_axis(ax, small=small)
    mask = (
        np.isfinite(wave)
        & np.isfinite(flux)
        & (wave >= row.window_low_um)
        & (wave <= row.window_high_um)
    )

    if int(mask.sum()) >= 2:
        local_wave = wave[mask]
        local_flux = flux[mask]
        ax.plot(
            local_wave,
            local_flux,
            color=TEXT,
            linewidth=0.70,
            alpha=0.88,
            label="real JWST/MAST spectrum",
        )

        # Exactly two reference lines.
        ax.axvline(
            row.centroid_expected_observed_um,
            color=CYAN,
            linestyle="--",
            linewidth=1.35,
            label="rest reference at published z=14.44",
        )
        ax.axvline(
            row.raw_local_peak_um_exploratory,
            color=ORANGE,
            linewidth=1.55,
            label="observed peak used",
        )
        ax.scatter(
            [row.raw_local_peak_um_exploratory],
            [row.raw_local_peak_flux],
            s=34 if small else 52,
            color=color,
            edgecolor="#f8fbff",
            linewidth=0.75,
            zorder=6,
        )

        ymin = float(np.nanmin(local_flux))
        ymax = float(np.nanmax(local_flux))
        span = ymax - ymin
        pad = 0.08 * span if span > 0 else max(abs(ymax) * 0.10, 1.0e-6)
        ax.set_ylim(ymin - pad, ymax + pad)
    else:
        ax.text(
            0.5,
            0.5,
            "NO FINITE DATA",
            transform=ax.transAxes,
            color=TEXT,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlim(row.window_low_um, row.window_high_um)
    ax.set_xlabel("Observed wavelength, µm", fontsize=7.7 if small else 9.5)
    ax.set_ylabel("Raw flux", fontsize=7.7 if small else 9.5)
    ax.set_title(
        f"{GALAXY} — {int(row.n)} {row.line_complex}",
        fontsize=8.8 if small else 11.5,
        pad=7,
    )

    calculation = (
        f"REST λ = {row.centroid_rest_um:.6f} µm\n"
        f"OBSERVED λ = {row.raw_local_peak_um_exploratory:.6f} µm\n"
        f"z = OBSERVED / REST − 1\n"
        f"z = {row.z_pair:.6f}"
    )
    ax.text(
        0.018,
        0.965,
        calculation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=TEXT,
        fontsize=7.0 if small else 9.0,
        linespacing=1.35,
        bbox=dict(
            boxstyle="round,pad=0.36",
            facecolor="#07111f",
            edgecolor=color,
            alpha=0.95,
        ),
    )
    dark_legend(ax, small=small)


def draw_z_summary(ax, lines, small=False):
    import numpy as np

    style_axis(ax, small=small)
    x = np.arange(len(lines))
    mean_z = float(lines["z_pair"].mean())

    ax.axhline(
        PUBLISHED_Z,
        color=ORANGE,
        linewidth=2.0,
        label=f"published joint-fit z = {PUBLISHED_Z:.2f}",
    )
    ax.axhline(
        mean_z,
        color=CYAN,
        linestyle="--",
        linewidth=1.35,
        label=f"mean pair z = {mean_z:.6f}",
    )
    ax.scatter(
        x,
        lines["z_pair"],
        s=60 if small else 95,
        color=POINT_COLORS[: len(lines)],
        edgecolor="#f8fbff",
        linewidth=0.8,
        zorder=6,
    )

    for index, row in enumerate(lines.itertuples()):
        ax.text(
            index,
            row.z_pair,
            f" {row.z_pair:.4f}",
            color=TEXT,
            fontsize=7.0 if small else 8.5,
            ha="center",
            va="bottom" if row.z_pair >= mean_z else "top",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{int(row.n)}\n{row.line_complex}" for row in lines.itertuples()],
        color=TEXT,
        fontsize=6.5 if small else 7.6,
    )
    ax.set_ylabel("z from wavelength pair", fontsize=7.8 if small else 9.6)
    ax.set_title("Five simple wavelength-pair results", fontsize=9.0 if small else 11.8, pad=8)
    dark_legend(ax, small=small)
    return mean_z


def create_individual_plots(lines, wave, flux):
    import matplotlib.pyplot as plt

    paths = []
    for index, row in enumerate(lines.itertuples()):
        fig = plt.figure(figsize=(13.8, 6.2), facecolor=BG)
        ax = fig.add_axes([0.075, 0.13, 0.89, 0.76])
        draw_spectrum(ax, row, wave, flux, POINT_COLORS[index], small=False)
        fig.suptitle(
            f"{VERSION} — {GALAXY}: two reference lines and one z calculation",
            color=TEXT,
            fontsize=13.0,
            y=0.975,
        )
        path = PNG / f"{VERSION}_{GALAXY}_TWO_LINES_{int(row.n)}.png"
        fig.savefig(path, dpi=245, facecolor=BG, edgecolor=BG)
        plt.show()
        plt.close(fig)
        paths.append(path)
    return paths


def create_dashboard(lines, wave, flux):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(18.0, 14.0), facecolor=BG)
    axes = axes.ravel()

    for index, row in enumerate(lines.itertuples()):
        draw_spectrum(axes[index], row, wave, flux, POINT_COLORS[index], small=True)

    mean_z = draw_z_summary(axes[5], lines, small=True)

    fig.suptitle(
        f"{VERSION} — {GALAXY}: five spectra, exactly two reference lines each",
        color=TEXT,
        fontsize=15.5,
        fontweight="bold",
        y=0.984,
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.947,
        bottom=0.055,
        hspace=0.36,
        wspace=0.18,
    )
    path = PNG / f"{VERSION}_{GALAXY}_TWO_REFERENCE_LINE_DASHBOARD.png"
    fig.savefig(path, dpi=230, facecolor=BG, edgecolor=BG)
    plt.show()
    plt.close(fig)
    return path, mean_z


def create_summary_table(lines, mean_z):
    import matplotlib.pyplot as plt

    table_rows = []
    for row in lines.itertuples():
        table_rows.append(
            [
                int(row.n),
                row.line_complex,
                f"{row.centroid_rest_um:.6f}",
                f"{row.raw_local_peak_um_exploratory:.6f}",
                f"{row.z_pair:.6f}",
            ]
        )

    table_rows.append(["μ", "MEAN OF FIVE PAIRS", "", "", f"{mean_z:.6f}"])
    table_rows.append(["★", "PUBLISHED JOINT FIT", "", "", f"{PUBLISHED_Z:.6f}"])

    fig = plt.figure(figsize=(14.8, 5.8), facecolor=BG)
    ax = fig.add_axes([0.035, 0.14, 0.93, 0.72])
    ax.axis("off")
    ax.set_title(
        f"{VERSION} — {GALAXY}: only the wavelength pair used for each z",
        color=TEXT,
        fontsize=13.8,
        fontweight="bold",
        pad=14,
    )

    table = ax.table(
        cellText=table_rows,
        colLabels=["#", "Line complex", "Rest λ µm", "Observed λ µm", "z"],
        loc="center",
        cellLoc="center",
        colWidths=[0.06, 0.36, 0.18, 0.20, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.7)
    table.scale(1.0, 1.55)

    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#29495f")
        cell.set_linewidth(0.50)
        cell.get_text().set_color(TEXT)
        if row_index == 0:
            cell.set_facecolor("#123149")
            cell.get_text().set_fontweight("bold")
        elif row_index <= len(lines):
            cell.set_facecolor("#081523")
            if col_index == 0:
                cell.get_text().set_color(POINT_COLORS[row_index - 1])
                cell.get_text().set_fontweight("bold")
        elif row_index == len(lines) + 1:
            cell.set_facecolor("#17314a")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#4a2e12")
            cell.get_text().set_color("#ffd39a")
            cell.get_text().set_fontweight("bold")

    fig.text(
        0.5,
        0.055,
        "No frequency table. No component list. Only rest wavelength, observed wavelength, and z.",
        color=MUTED,
        fontsize=9.0,
        ha="center",
    )

    table_path = PNG / f"{VERSION}_{GALAXY}_SIMPLE_PAIR_TABLE.png"
    fig.savefig(table_path, dpi=245, facecolor=BG, edgecolor=BG)
    plt.show()
    plt.close(fig)

    result_csv = CSV / f"{VERSION}_{GALAXY}_SIMPLE_PAIR_RESULTS.csv"
    lines.to_csv(result_csv, index=False)
    return table_path, result_csv


def print_table(rows, headers):
    widths = [len(str(header)) for header in headers]
    for row in rows:
        widths = [max(widths[i], len(str(row[i]))) for i in range(len(headers))]
    print(" | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def main():
    for package in ["numpy", "pandas", "matplotlib"]:
        need(package)

    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)

    line_csv, raw_csv = locate_inputs()
    lines, wave, flux, flux_column = load_data(line_csv, raw_csv)

    individual_paths = create_individual_plots(lines, wave, flux)
    dashboard_path, mean_z = create_dashboard(lines, wave, flux)
    table_path, result_csv = create_summary_table(lines, mean_z)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table(
        [
            (
                int(row.n),
                row.line_complex,
                f"{row.centroid_rest_um:.6f}",
                f"{row.raw_local_peak_um_exploratory:.6f}",
                f"{row.z_pair:.6f}",
            )
            for row in lines.itertuples()
        ],
        ["#", "Line complex", "Rest lambda um", "Observed lambda um", "z"],
    )
    print()
    print_table(
        [
            ("Galaxy", GALAXY),
            ("Published z", f"{PUBLISHED_Z:.6f}"),
            ("Mean pair z", f"{mean_z:.6f}"),
            ("Reference lines per spectrum", 2),
            ("Individual plots", len(individual_paths)),
            ("Dashboard PNG", str(dashboard_path)),
            ("Summary table PNG", str(table_path)),
            ("Results CSV", str(result_csv)),
            ("Line source CSV", str(line_csv)),
            ("Spectrum source CSV", str(raw_csv)),
            ("Flux column", flux_column),
        ],
        ["Field", "Value"],
    )
    print("\nINDIVIDUAL PLOTS")
    print_table([(index + 1, str(path)) for index, path in enumerate(individual_paths)], ["#", "Path"])
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
